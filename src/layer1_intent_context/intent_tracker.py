from __future__ import annotations

import argparse
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

from src.evaluation.metrics import multilabel_metrics, per_tag_metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import DataLoader, Dataset

from src.core.constants import ALL_TAGS, LAYER1_DATA_DIR


@dataclass
class IntentPrediction:
    text: str
    tag_scores: Dict[str, float]


class Layer1DLConfig:
    def __init__(self, payload: Dict) -> None:
        self.epochs = int(payload.get("epochs", 8))
        self.batch_size = int(payload.get("batch_size", 32))
        self.learning_rate = float(payload.get("learning_rate", 1e-3))
        self.weight_decay = float(payload.get("weight_decay", 1e-4))
        self.hidden_dim = int(payload.get("hidden_dim", 256))
        self.projection_dim = int(payload.get("projection_dim", 128))
        self.embedding_dim = int(payload.get("embedding_dim", 192))
        self.max_length = int(payload.get("max_length", 48))
        self.alpha_metric_loss = float(payload.get("alpha_metric_loss", 0.35))
        self.validation_ratio = float(payload.get("validation_ratio", 0.2))
        self.decision_threshold = float(payload.get("decision_threshold", 0.3))
        self.random_seed = int(payload.get("random_seed", 42))
        self.min_token_frequency = int(payload.get("min_token_frequency", 1))
        self.include_rl_samples = bool(payload.get("include_rl_samples", True))


class TextTensorDataset(Dataset):
    def __init__(self, input_ids: np.ndarray, attention_mask: np.ndarray, targets: np.ndarray) -> None:
        self.input_ids = torch.tensor(input_ids, dtype=torch.long)
        self.attention_mask = torch.tensor(attention_mask, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.input_ids.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.input_ids[idx], self.attention_mask[idx], self.targets[idx]


class Layer1DLModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        projection_dim: int,
        num_tags: int,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1),
        )
        self.projector = nn.Linear(hidden_dim, projection_dim)
        self.classifier = nn.Linear(hidden_dim, num_tags)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.embedding(input_ids)  # [B, L, D]
        masked = emb * attention_mask.unsqueeze(-1)
        pooled = masked.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True).clamp(min=1e-9)
        hidden = self.encoder(pooled)
        projection = F.normalize(self.projector(hidden), p=2, dim=1)
        logits = self.classifier(hidden)
        return projection, logits


class IntentTracker:
    """Layer1 DL-only model with metric learning on tag overlap."""

    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"

    def __init__(
        self,
        data_dir: Path | None = None,
        use_all_datasets: bool = False,
        include_rl_samples: bool | None = None,
    ) -> None:
        self.data_dir = data_dir or LAYER1_DATA_DIR
        self.use_all_datasets = use_all_datasets
        self.dataset_meta_file = self.data_dir / "datasets.json"
        self.tags_file = self.data_dir / "tags.json"
        self.dl_config_file = self.data_dir / "dl_config.json"
        self.dataset_dir = self._resolve_dataset_dir()
        self.train_file = self.dataset_dir / "intent_samples.csv"
        self.tags = self._load_tags()
        self.mlb = MultiLabelBinarizer(classes=self.tags)
        self.dl_config = self._load_dl_config()
        if include_rl_samples is not None:
            self.dl_config.include_rl_samples = bool(include_rl_samples)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._artifact_suffix = ""
        self.artifact_dir = self.data_dir / "model_artifacts_dl"
        self.model_file = self.artifact_dir / "intent_model.pt"
        self.meta_file = self.artifact_dir / "intent_model_meta.json"
        self.vocab_file = self.artifact_dir / "vocab.json"

        self.vocab: Dict[str, int] = {self.PAD_TOKEN: 0, self.UNK_TOKEN: 1}
        self.model: Layer1DLModel | None = None
        self.is_fitted = False
        self._load_artifacts_if_ready()

    def _resolve_dataset_dir(self) -> Path:
        if not self.dataset_meta_file.exists():
            return self.data_dir
        payload = json.loads(self.dataset_meta_file.read_text(encoding="utf-8"))
        active_dataset = payload.get("active_dataset")
        if not active_dataset:
            return self.data_dir
        return self.data_dir / active_dataset

    def _resolve_training_files(self) -> List[Path]:
        if not self.dataset_meta_file.exists():
            return [self.train_file]
        payload = json.loads(self.dataset_meta_file.read_text(encoding="utf-8"))
        if not self.use_all_datasets:
            return [self.train_file]
        training_files: List[Path] = []
        for dataset_name in payload.get("available_datasets", []):
            file_path = self.data_dir / dataset_name / "intent_samples.csv"
            if file_path.exists():
                training_files.append(file_path)
        return training_files or [self.train_file]

    def _load_tags(self) -> List[str]:
        if self.tags_file.exists():
            payload = json.loads(self.tags_file.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and isinstance(payload.get("tag_ids"), list):
                return payload["tag_ids"]
            if isinstance(payload, dict) and isinstance(payload.get("tags"), list):
                return payload["tags"]
        return ALL_TAGS

    def _load_dl_config(self) -> Layer1DLConfig:
        if not self.dl_config_file.exists():
            return Layer1DLConfig({})
        payload = json.loads(self.dl_config_file.read_text(encoding="utf-8"))
        return Layer1DLConfig(payload if isinstance(payload, dict) else {})

    @staticmethod
    def _normalize_vietnamese_text(text: str) -> str:
        normalized = text.lower().strip()
        normalized = normalized.replace("đ", "d").replace("Đ", "d")
        normalized = unicodedata.normalize("NFD", normalized)
        normalized = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    def _build_feature_text(self, text: str) -> str:
        raw = text.strip()
        vi_norm = self._normalize_vietnamese_text(raw)
        return f"{raw} || {vi_norm}"

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[a-zA-Z0-9_]+", text.lower())

    def _build_vocab(self, texts: List[str]) -> None:
        counts: Dict[str, int] = {}
        for text in texts:
            for token in self._tokenize(text):
                counts[token] = counts.get(token, 0) + 1
        self.vocab = {self.PAD_TOKEN: 0, self.UNK_TOKEN: 1}
        for token, freq in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            if freq < self.dl_config.min_token_frequency:
                continue
            self.vocab[token] = len(self.vocab)

    def _texts_to_tensors(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        input_ids: List[List[int]] = []
        masks: List[List[float]] = []
        for text in texts:
            tokens = self._tokenize(text)
            ids = [self.vocab.get(token, self.vocab[self.UNK_TOKEN]) for token in tokens][: self.dl_config.max_length]
            if not ids:
                ids = [self.vocab[self.UNK_TOKEN]]
            mask = [1.0] * len(ids)
            if len(ids) < self.dl_config.max_length:
                pad_len = self.dl_config.max_length - len(ids)
                ids.extend([self.vocab[self.PAD_TOKEN]] * pad_len)
                mask.extend([0.0] * pad_len)
            input_ids.append(ids)
            masks.append(mask)
        return np.array(input_ids, dtype=np.int64), np.array(masks, dtype=np.float32)

    @staticmethod
    def _overlap_similarity(label_a: np.ndarray, label_b: np.ndarray) -> float:
        overlap_count = float(np.logical_and(label_a > 0.5, label_b > 0.5).sum())
        max_tag_count = float(max((label_a > 0.5).sum(), (label_b > 0.5).sum(), 1))
        return overlap_count / max_tag_count

    def _pairwise_similarity_target(self, labels: torch.Tensor) -> torch.Tensor:
        labels_np = labels.detach().cpu().numpy()
        batch_size = labels_np.shape[0]
        matrix = np.zeros((batch_size, batch_size), dtype=np.float32)
        for i in range(batch_size):
            for j in range(batch_size):
                matrix[i, j] = self._overlap_similarity(labels_np[i], labels_np[j])
        return torch.tensor(matrix, dtype=torch.float32, device=labels.device)

    def _collect_base_records(self) -> List[dict]:
        training_files = self._resolve_training_files()
        frames = [pd.read_csv(file_path) for file_path in training_files]
        frame = pd.concat(frames, ignore_index=True)
        records: List[dict] = []
        for _, row in frame.iterrows():
            text = str(row.get("text", "")).strip()
            tags = [tag.strip() for tag in str(row.get("tags", "")).split("|") if tag.strip()]
            if text and tags:
                records.append({"text": text, "tags": tags})
        return records

    def _collect_rl_records(self) -> List[dict]:
        rl_file = self.data_dir / "rl_training" / "intent_train_data_rl.json"
        if not rl_file.exists():
            return []
        payload = json.loads(rl_file.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            return []
        records: List[dict] = []
        for item in payload:
            text = str(item.get("text", "")).strip()
            tags = item.get("tags", [])
            if text and isinstance(tags, list) and tags:
                records.append({"text": text, "tags": [str(tag) for tag in tags]})
        return records

    def _collect_train_records(self) -> List[dict]:
        records = self._collect_base_records()
        if self.dl_config.include_rl_samples:
            records.extend(self._collect_rl_records())
        return records

    def _artifact_paths(self) -> tuple[Path, Path, Path]:
        suffix = f"_{self._artifact_suffix}" if self._artifact_suffix else ""
        model_file = self.artifact_dir / f"intent_model{suffix}.pt"
        meta_file = self.artifact_dir / f"intent_model_meta{suffix}.json"
        vocab_file = self.artifact_dir / f"vocab{suffix}.json"
        return model_file, meta_file, vocab_file

    def _create_model(self) -> None:
        self.model = Layer1DLModel(
            vocab_size=len(self.vocab),
            embedding_dim=self.dl_config.embedding_dim,
            hidden_dim=self.dl_config.hidden_dim,
            projection_dim=self.dl_config.projection_dim,
            num_tags=len(self.tags),
        ).to(self.device)

    def _save_artifacts(self) -> None:
        if self.model is None:
            return
        model_file, meta_file, vocab_file = self._artifact_paths()
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), model_file)
        vocab_file.write_text(json.dumps(self.vocab, ensure_ascii=False, indent=2), encoding="utf-8")
        meta = {
            "tags": self.tags,
            "decision_threshold": self.dl_config.decision_threshold,
            "max_length": self.dl_config.max_length,
            "embedding_dim": self.dl_config.embedding_dim,
            "hidden_dim": self.dl_config.hidden_dim,
            "projection_dim": self.dl_config.projection_dim,
            "include_rl_samples": self.dl_config.include_rl_samples,
            "artifact_suffix": self._artifact_suffix,
        }
        meta_file.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        if not self._artifact_suffix:
            torch.save(self.model.state_dict(), self.model_file)
            self.vocab_file.write_text(json.dumps(self.vocab, ensure_ascii=False, indent=2), encoding="utf-8")
            self.meta_file.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load_artifacts_if_ready(self) -> None:
        if not self.model_file.exists() or not self.meta_file.exists() or not self.vocab_file.exists():
            return
        meta = json.loads(self.meta_file.read_text(encoding="utf-8"))
        if meta.get("tags") != self.tags:
            return
        self.vocab = json.loads(self.vocab_file.read_text(encoding="utf-8"))
        self._create_model()
        if self.model is None:
            return
        self.model.load_state_dict(torch.load(self.model_file, map_location=self.device))
        self.model.eval()
        self.is_fitted = True

    def _predict_binary(self, x_ids: np.ndarray, x_mask: np.ndarray) -> np.ndarray:
        if self.model is None or len(x_ids) == 0:
            return np.empty((0, len(self.tags)), dtype=np.int32)
        with torch.no_grad():
            ids = torch.tensor(x_ids, dtype=torch.long, device=self.device)
            mask = torch.tensor(x_mask, dtype=torch.float32, device=self.device)
            _, logits = self.model(ids, mask)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
        return (probs >= self.dl_config.decision_threshold).astype(np.int32)

    def _evaluate_f1(self, x_ids: np.ndarray, x_mask: np.ndarray, y_true: np.ndarray) -> tuple[float, float]:
        if self.model is None or len(x_ids) == 0:
            return 0.0, 0.0
        y_pred = self._predict_binary(x_ids, x_mask)
        micro = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
        macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        return micro, macro

    def evaluate_validation_split(self) -> Dict[str, object]:
        base_records = self._collect_base_records()
        if not base_records:
            raise ValueError("Khong co du lieu validation Layer1.")
        self.mlb.fit([item["tags"] for item in base_records])
        raw_texts = [item["text"] for item in base_records]
        tag_lists = [item["tags"] for item in base_records]
        indices = np.arange(len(base_records))
        _, val_idx = train_test_split(
            indices,
            test_size=self.dl_config.validation_ratio,
            random_state=self.dl_config.random_seed,
        )
        val_raw_texts = [raw_texts[int(i)] for i in val_idx]
        val_tag_lists = [tag_lists[int(i)] for i in val_idx]
        return self.evaluate(val_raw_texts, val_tag_lists)

    def evaluate(self, texts: List[str], tag_lists: List[List[str]]) -> Dict[str, object]:
        if self.model is None:
            raise RuntimeError("Model chua duoc train de evaluate.")
        feature_texts = [self._build_feature_text(text) for text in texts]
        input_ids, mask = self._texts_to_tensors(feature_texts)
        y_true = self.mlb.transform(tag_lists).astype(np.int32)
        y_pred = self._predict_binary(input_ids, mask)
        summary = multilabel_metrics(y_true, y_pred)
        summary["samples"] = len(texts)
        summary["threshold"] = self.dl_config.decision_threshold
        return {
            "summary": summary,
            "per_tag": per_tag_metrics(y_true, y_pred, self.tags),
        }

    def load_artifacts(self, suffix: str = "") -> None:
        self._artifact_suffix = suffix
        model_file, meta_file, vocab_file = self._artifact_paths()
        if not model_file.exists() or not meta_file.exists() or not vocab_file.exists():
            raise FileNotFoundError(f"Khong tim thay artifact suffix='{suffix}'")
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        if meta.get("tags") != self.tags:
            raise ValueError("Tag schema khong khop artifact.")
        self.vocab = json.loads(vocab_file.read_text(encoding="utf-8"))
        self._create_model()
        if self.model is None:
            raise RuntimeError("Khong khoi tao duoc model Layer1 DL")
        self.model.load_state_dict(torch.load(model_file, map_location=self.device))
        self.model.eval()
        self.is_fitted = True

    def fit(self, artifact_suffix: str = "") -> Dict[str, object]:
        self._artifact_suffix = artifact_suffix
        torch.manual_seed(self.dl_config.random_seed)
        np.random.seed(self.dl_config.random_seed)

        base_records = self._collect_base_records()
        if not base_records:
            raise ValueError("Khong co du lieu train Layer1.")

        raw_texts = [item["text"] for item in base_records]
        base_texts = np.array([self._build_feature_text(text) for text in raw_texts], dtype=object)
        base_labels = self.mlb.fit_transform([item["tags"] for item in base_records]).astype(np.float32)
        indices = np.arange(len(base_records))
        train_idx, val_idx = train_test_split(
            indices,
            test_size=self.dl_config.validation_ratio,
            random_state=self.dl_config.random_seed,
        )
        train_texts = base_texts[train_idx]
        val_texts = base_texts[val_idx]
        train_y = base_labels[train_idx]
        val_y = base_labels[val_idx]
        val_raw_texts = [raw_texts[int(i)] for i in val_idx]

        if self.dl_config.include_rl_samples:
            rl_records = self._collect_rl_records()
            if rl_records:
                rl_texts = [self._build_feature_text(item["text"]) for item in rl_records]
                rl_labels = self.mlb.transform([item["tags"] for item in rl_records]).astype(np.float32)
                train_texts = np.concatenate([train_texts, np.array(rl_texts, dtype=object)])
                train_y = np.vstack([train_y, rl_labels])

        self._build_vocab(train_texts.tolist())
        train_ids, train_mask = self._texts_to_tensors(train_texts.tolist())
        val_ids, val_mask = self._texts_to_tensors(val_texts.tolist()) if len(val_texts) > 0 else (np.empty((0, self.dl_config.max_length), dtype=np.int64), np.empty((0, self.dl_config.max_length), dtype=np.float32))

        self._create_model()
        if self.model is None:
            raise RuntimeError("Khong khoi tao duoc model Layer1 DL")

        train_loader = DataLoader(
            TextTensorDataset(train_ids, train_mask, train_y),
            batch_size=self.dl_config.batch_size,
            shuffle=True,
        )
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.dl_config.learning_rate,
            weight_decay=self.dl_config.weight_decay,
        )

        self.model.train()
        for epoch in range(self.dl_config.epochs):
            total_loss, total_cls, total_metric, steps = 0.0, 0.0, 0.0, 0
            for input_ids, attention_mask, target in train_loader:
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                target = target.to(self.device)

                projection, logits = self.model(input_ids, attention_mask)
                cls_loss = F.binary_cross_entropy_with_logits(logits, target)
                pred_sim = projection @ projection.t()
                target_sim = self._pairwise_similarity_target(target)
                if pred_sim.shape[0] > 1:
                    eye = torch.eye(pred_sim.size(0), device=self.device, dtype=torch.bool)
                    metric_loss = F.mse_loss(pred_sim[~eye], target_sim[~eye])
                else:
                    metric_loss = torch.tensor(0.0, device=self.device)

                loss = cls_loss + self.dl_config.alpha_metric_loss * metric_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += float(loss.item())
                total_cls += float(cls_loss.item())
                total_metric += float(metric_loss.item())
                steps += 1

            micro_f1, macro_f1 = self._evaluate_f1(val_ids, val_mask, val_y)
            denom = max(steps, 1)
            print(
                f"[Epoch {epoch + 1}/{self.dl_config.epochs}] "
                f"loss={total_loss / denom:.4f} "
                f"cls={total_cls / denom:.4f} "
                f"metric={total_metric / denom:.4f} "
                f"val_micro_f1={micro_f1:.4f} "
                f"val_macro_f1={macro_f1:.4f}"
            )

        self.model.eval()
        self.is_fitted = True
        self._save_artifacts()
        val_tag_lists = [
            [self.tags[idx] for idx, value in enumerate(row) if value > 0.5]
            for row in val_y
        ]
        return self.evaluate(val_raw_texts, val_tag_lists)

    def predict_tags(self, text: str) -> IntentPrediction:
        if not self.is_fitted or self.model is None:
            # Lazy-train de chat flow khong bi vo khi chua run lenh train truoc.
            self.fit()
        if self.model is None:
            raise RuntimeError("Khong khoi tao duoc model Layer1 DL sau khi train.")
        feature_text = self._build_feature_text(text)
        input_ids, mask = self._texts_to_tensors([feature_text])
        with torch.no_grad():
            ids_tensor = torch.tensor(input_ids, dtype=torch.long, device=self.device)
            mask_tensor = torch.tensor(mask, dtype=torch.float32, device=self.device)
            _, logits = self.model(ids_tensor, mask_tensor)
            probs = torch.sigmoid(logits).detach().cpu().numpy()[0]
        scores = {
            tag: float(max(0.0, min(1.0, score)))
            for tag, score in zip(self.tags, probs, strict=False)
        }
        return IntentPrediction(text=text, tag_scores=scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Layer1 IntentTracker DL")
    parser.add_argument("--all-datasets", action="store_true")
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()

    tracker = IntentTracker(use_all_datasets=args.all_datasets)
    if args.threshold is not None:
        tracker.dl_config.decision_threshold = float(args.threshold)
    tracker.fit()
    mode = "ALL_DATASETS" if args.all_datasets else "ACTIVE_DATASET_ONLY"
    print(f"Da train xong Layer1 DL ({mode}) voi so tag:", len(tracker.tags))
