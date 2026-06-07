from __future__ import annotations

import argparse
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

from src.core.constants import ALL_TAGS, LAYER1_DATA_DIR


@dataclass
class IntentPrediction:
    text: str
    tag_scores: Dict[str, float]


class IntentTracker:
    """Phan loai da nhan intent/slot bang TF-IDF + LogisticRegression."""

    def __init__(self, data_dir: Path | None = None, use_all_datasets: bool = False) -> None:
        self.data_dir = data_dir or LAYER1_DATA_DIR
        self.use_all_datasets = use_all_datasets
        self.dataset_meta_file = self.data_dir / "datasets.json"
        self.tags_file = self.data_dir / "tags.json"
        self.dataset_dir = self._resolve_dataset_dir()
        self.train_file = self.dataset_dir / "intent_samples.csv"
        self.tags = self._load_tags()
        self.mlb = MultiLabelBinarizer(classes=self.tags)
        self.model = self._build_model()
        self.is_fitted = False

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

        if not training_files:
            training_files.append(self.train_file)
        return training_files

    def _load_tags(self) -> List[str]:
        if self.tags_file.exists():
            payload = json.loads(self.tags_file.read_text(encoding="utf-8"))
            if "tag_ids" in payload and isinstance(payload["tag_ids"], list):
                return payload["tag_ids"]
            return payload.get("tags", ALL_TAGS)
        return ALL_TAGS

    @staticmethod
    def _build_model() -> Pipeline:
        return Pipeline(
            steps=[
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=3000)),
                (
                    "clf",
                    OneVsRestClassifier(
                        LogisticRegression(max_iter=300, solver="liblinear")
                    ),
                ),
            ]
        )

    @staticmethod
    def _normalize_vietnamese_text(text: str) -> str:
        normalized = text.lower().strip()
        # Chuyen rieng ky tu đ/Đ truoc khi bo dau.
        normalized = normalized.replace("đ", "d").replace("Đ", "d")
        normalized = unicodedata.normalize("NFD", normalized)
        normalized = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    def _build_feature_text(self, text: str) -> str:
        # Ghep cau goc + cau bo dau de model robust hon voi input khong dau.
        raw = text.strip()
        vi_norm = self._normalize_vietnamese_text(raw)
        return f"{raw} || {vi_norm}"

    def fit(self) -> None:
        training_files = self._resolve_training_files()
        frames = [pd.read_csv(file_path) for file_path in training_files]
        frame = pd.concat(frames, ignore_index=True)
        texts = (
            frame["text"]
            .fillna("")
            .astype(str)
            .apply(self._build_feature_text)
            .tolist()
        )
        raw_tags = (
            frame["tags"]
            .fillna("")
            .astype(str)
            .apply(lambda value: [tag.strip() for tag in value.split("|") if tag.strip()])
            .tolist()
        )
        target = self.mlb.fit_transform(raw_tags)
        self.model.fit(texts, target)
        self.is_fitted = True

    def predict_tags(self, text: str) -> IntentPrediction:
        if not self.is_fitted:
            self.fit()

        feature_text = self._build_feature_text(text)
        proba_list = self.model.predict_proba([feature_text])[0]
        scores = {
            tag: float(max(0.0, min(1.0, score)))
            for tag, score in zip(self.tags, proba_list, strict=False)
        }
        return IntentPrediction(text=text, tag_scores=scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Layer1 IntentTracker")
    parser.add_argument(
        "--all-datasets",
        action="store_true",
        help="Train gop tat ca dataset trong available_datasets",
    )
    args = parser.parse_args()

    tracker = IntentTracker(use_all_datasets=args.all_datasets)
    tracker.fit()
    mode = "ALL_DATASETS" if args.all_datasets else "ACTIVE_DATASET_ONLY"
    print(f"Da train xong IntentTracker ({mode}) voi so tag:", len(tracker.tags))
