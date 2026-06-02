from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

from src.core.constants import ALL_TAGS, LAYER2_DATA_DIR


# Mapping de migrate tag legacy ve taxonomy chung cua Layer1.
LEGACY_TAG_ALIASES = {
    "time_late_night": "time_night",
    "time_weekday": "time_noon",
    "time_weekend": "time_snacks",
    "time_busy": "pref_convenient",
    "time_relaxed": "mood_normal",
    "time_quick_meal": "pref_instant",
    "weather_rainy": "weather_rain",
    "weather_sunny": "weather_normal",
    "weather_humid": "weather_normal",
    "weather_dry": "weather_normal",
    "weather_windy": "weather_normal",
    "weather_stormy": "weather_storm",
    "weather_cloudy": "weather_normal",
    "weather_mild": "weather_normal",
    "mood_sad": "mood_lonely",
    "mood_tired": "mood_exhausted",
    "mood_energetic": "mood_excited",
    "mood_adventurous": "mood_excited",
    "mood_social": "mood_gathering",
    "mood_comfort_seek": "mood_sluggish",
}


class RecommendationEngine:
    """Tinh diem goi y mon an bang Linear Weight Scoring."""

    def __init__(self, data_dir: Path | None = None) -> None:
        self.data_dir = data_dir or LAYER2_DATA_DIR
        self.dataset_meta_file = self.data_dir / "datasets.json"
        self.active_dataset = self._resolve_active_dataset()
        self.dataset_dir = self.data_dir / self.active_dataset
        self.manifest_file = self.dataset_dir / "dataset_manifest.json"
        self.canonical_file, self.legacy_file = self._resolve_dataset_files()
        self.runtime_dir = self.data_dir / "runtime"
        self.runtime_file = self.runtime_dir / f"{self.active_dataset}_dishes_runtime.json"
        self.source_file: Path = self.legacy_file
        self.source_kind: str = "legacy"
        self.dishes = self._load_dishes()

    def _resolve_active_dataset(self) -> str:
        if not self.dataset_meta_file.exists():
            return "dataset_v001"
        try:
            payload = json.loads(self.dataset_meta_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return "dataset_v001"

        active = payload.get("active_dataset")
        if isinstance(active, str) and active.strip():
            return active.strip()
        return "dataset_v001"

    def _resolve_dataset_files(self) -> Tuple[Path, Path]:
        canonical = self.dataset_dir / "food_weight_matrix.json"
        legacy = self.data_dir / "dishes_100.json"

        if not self.manifest_file.exists():
            return canonical, legacy

        try:
            payload = json.loads(self.manifest_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return canonical, legacy

        inputs = payload.get("inputs", {}) if isinstance(payload, dict) else {}
        if not isinstance(inputs, dict):
            return canonical, legacy

        canonical_rel = inputs.get("canonical_matrix")
        legacy_rel = inputs.get("legacy_matrix")

        if isinstance(canonical_rel, str) and canonical_rel.strip():
            canonical = (self.dataset_dir / canonical_rel).resolve()
        if isinstance(legacy_rel, str) and legacy_rel.strip():
            legacy = (self.dataset_dir / legacy_rel).resolve()
        return canonical, legacy

    @staticmethod
    def _normalize_tag_weights(raw_weights: Dict) -> Dict[str, float]:
        normalized = {tag: 0.0 for tag in ALL_TAGS}
        for raw_tag, raw_weight in raw_weights.items():
            target_tag = LEGACY_TAG_ALIASES.get(raw_tag, raw_tag)
            if target_tag not in normalized:
                continue
            try:
                value = float(raw_weight)
            except (TypeError, ValueError):
                value = 0.0
            normalized[target_tag] += value

        for tag, value in normalized.items():
            normalized[tag] = max(-1.0, min(1.0, value))
        return normalized

    def _canonical_key_to_dish_id(self, key: str) -> str:
        suffix = key.split("_", maxsplit=1)[-1] if "_" in key else key
        if suffix.isdigit():
            return f"dish_{int(suffix):03d}"
        return key.lower()

    def _load_legacy_dishes(self) -> List[Dict]:
        if not self.legacy_file.exists():
            return []
        try:
            payload = json.loads(self.legacy_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
        if not isinstance(payload, dict):
            return []
        dishes = payload.get("dishes", [])
        if not isinstance(dishes, list):
            return []
        return dishes

    def _load_canonical_dishes(self) -> List[Dict]:
        if not self.canonical_file.exists():
            return []
        try:
            payload = json.loads(self.canonical_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
        if not isinstance(payload, dict):
            return []

        if isinstance(payload.get("dishes"), list):
            return payload["dishes"]

        dishes: List[Dict] = []
        for key, value in payload.items():
            if not (isinstance(key, str) and key.startswith("FOOD_")):
                continue
            if not isinstance(value, dict):
                continue
            dishes.append(
                {
                    "id": self._canonical_key_to_dish_id(key),
                    "name": value.get("name", key),
                    "is_drink": bool(value.get("is_drink", False)),
                    "popularity_score": float(value.get("popularity_score", 1.0)),
                    "tag_weights": value.get("tag_weights", {}),
                }
            )
        return dishes

    def _load_runtime_dishes(self) -> List[Dict]:
        if not self.runtime_file.exists():
            return []
        try:
            payload = json.loads(self.runtime_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
        if not isinstance(payload, dict):
            return []
        dishes = payload.get("dishes", [])
        return dishes if isinstance(dishes, list) else []

    def _choose_source(self, canonical: List[Dict], legacy: List[Dict]) -> Tuple[List[Dict], Path, str]:
        runtime = self._load_runtime_dishes()
        if runtime:
            return runtime, self.runtime_file, "runtime"
        if canonical:
            return canonical, self.runtime_file, "canonical"
        if legacy:
            return legacy, self.runtime_file, "legacy"
        return [], self.legacy_file, "legacy"

    def _load_dishes(self) -> List[Dict]:
        canonical_dishes = self._load_canonical_dishes()
        legacy_dishes = self._load_legacy_dishes()
        dishes, source_file, source_kind = self._choose_source(canonical_dishes, legacy_dishes)
        self.source_file = source_file
        self.source_kind = source_kind

        normalized_dishes: List[Dict] = []
        for dish in dishes:
            if not isinstance(dish, dict):
                continue
            tag_weights = dish.get("tag_weights", {})
            if not isinstance(tag_weights, dict):
                tag_weights = {}
            normalized_dishes.append(
                {
                    "id": str(dish.get("id", "")),
                    "name": str(dish.get("name", "Unknown dish")),
                    "is_drink": bool(dish.get("is_drink", False)),
                    "popularity_score": float(dish.get("popularity_score", 1.0)),
                    "tag_weights": self._normalize_tag_weights(tag_weights),
                }
            )
        return normalized_dishes

    def save(self) -> None:
        payload = {"dishes": self.dishes}
        self.source_file.parent.mkdir(parents=True, exist_ok=True)
        self.source_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self.source_kind = "runtime"

    def score_dish(self, dish: Dict, context_scores: Dict[str, float]) -> float:
        # Cong thuc tuyen tinh:
        # score(dish) = sum_t activation(tag_t) * weight(dish, tag_t)
        return float(
            sum(
                context_scores.get(tag, 0.0) * dish["tag_weights"].get(tag, 0.0)
                for tag in ALL_TAGS
            )
        )

    def recommend(self, context_scores: Dict[str, float], top_k: int = 5) -> List[Dict]:
        ranked: List[Dict] = []
        for dish in self.dishes:
            score = self.score_dish(dish, context_scores)
            ranked.append(
                {
                    "id": dish["id"],
                    "name": dish["name"],
                    "score": round(score, 4),
                    "popularity_score": float(dish.get("popularity_score", 1.0)),
                }
            )
        ranked.sort(
            key=lambda item: (
                item["score"],
                item["popularity_score"],
                item["id"],
            ),
            reverse=True,
        )
        for item in ranked:
            item.pop("popularity_score", None)
        return ranked[:top_k]

    def get_dish_by_id(self, dish_id: str) -> Dict:
        for dish in self.dishes:
            if dish["id"] == dish_id:
                return dish
        raise ValueError(f"Khong tim thay mon co id={dish_id}")
