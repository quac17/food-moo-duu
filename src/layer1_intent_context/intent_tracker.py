from __future__ import annotations

import json
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

    def __init__(self, data_dir: Path | None = None) -> None:
        self.data_dir = data_dir or LAYER1_DATA_DIR
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

    def fit(self) -> None:
        frame = pd.read_csv(self.train_file)
        texts = frame["text"].fillna("").astype(str).tolist()
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

        proba_list = self.model.predict_proba([text])[0]
        scores = {
            tag: float(max(0.0, min(1.0, score)))
            for tag, score in zip(self.tags, proba_list, strict=False)
        }
        return IntentPrediction(text=text, tag_scores=scores)


if __name__ == "__main__":
    tracker = IntentTracker()
    tracker.fit()
    print("Da train xong IntentTracker voi so tag:", len(tracker.tags))
