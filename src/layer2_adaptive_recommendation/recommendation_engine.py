from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

from src.core.constants import LAYER2_CONFIG, LAYER2_DATA_DIR


class RecommendationEngine:
    """Tinh diem goi y mon an bang Linear Weight Scoring."""

    def __init__(self, data_dir: Path | None = None, prefer_canonical: bool = False) -> None:
        self.data_dir = data_dir or LAYER2_DATA_DIR
        self.prefer_canonical = prefer_canonical
        self.dataset_meta_file = self.data_dir / "datasets.json"
        self.active_dataset = self._resolve_active_dataset()
        self.dataset_dir = self.data_dir / self.active_dataset
        self.manifest_file = self.dataset_dir / "dataset_manifest.json"
        self.canonical_file, self.legacy_file = self._resolve_dataset_files()
        self.runtime_dir = self.data_dir / "runtime"
        self.runtime_file = self.runtime_dir / f"{self.active_dataset}_dishes_runtime.json"
        self.similarity_tags = self._load_similarity_tags()
        self.source_file: Path = self.runtime_file
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

    def _load_similarity_tags(self) -> Dict[str, Dict[str, float]]:
        similarity = LAYER2_CONFIG.get("similarity", {})
        if not isinstance(similarity, dict):
            return {}
        normalized: Dict[str, Dict[str, float]] = {}
        for tag, mappings in similarity.items():
            if not isinstance(tag, str) or not isinstance(mappings, dict):
                continue
            normalized[tag] = {}
            for similar_tag, factor in mappings.items():
                try:
                    normalized[tag][str(similar_tag)] = float(factor)
                except (TypeError, ValueError):
                    continue
        return normalized

    @staticmethod
    def _coerce_float(value: object, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _add_weight(target: Dict[str, float], tag: str, amount: float) -> None:
        if amount <= 0.0:
            return
        target[tag] = min(1.0, target.get(tag, 0.0) + amount)

    def _get_source_mtime(self, file_path: Path) -> float:
        if not file_path.exists():
            return 0.0
        return file_path.stat().st_mtime

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
                    "popularity_score": self._coerce_float(value.get("popularity_score", 1.0), 1.0),
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
        if self.prefer_canonical and canonical:
            return canonical, self.canonical_file, "canonical"

        runtime = self._load_runtime_dishes()
        runtime_fresh = runtime and self._get_source_mtime(self.runtime_file) >= max(
            self._get_source_mtime(self.canonical_file),
            self._get_source_mtime(self.legacy_file),
        )
        if runtime_fresh:
            return runtime, self.runtime_file, "runtime"

        canonical_newer = self._get_source_mtime(self.canonical_file) >= self._get_source_mtime(self.legacy_file)
        if canonical and canonical_newer:
            return canonical, self.runtime_file, "canonical"
        if legacy:
            return legacy, self.runtime_file, "legacy"
        if canonical:
            return canonical, self.runtime_file, "canonical"
        return [], self.legacy_file, "legacy"

    def _normalize_tag_weights(self, raw_weights: Dict) -> Dict[str, float]:
        normalized: Dict[str, float] = {}
        for raw_tag, raw_weight in raw_weights.items():
            value = self._coerce_float(raw_weight, 0.0)
            normalized[raw_tag] = max(-1.0, min(1.0, normalized.get(raw_tag, 0.0) + value))
        return normalized

    def _blend_runtime_with_canonical(
        self,
        runtime_weights: Dict[str, float],
        canonical_weights: Dict[str, float],
        runtime_influence: float,
    ) -> Dict[str, float]:
        influence = max(0.0, min(1.0, runtime_influence))
        if influence == 0.0:
            return dict(canonical_weights)
        if influence == 1.0:
            return dict(runtime_weights)

        blended: Dict[str, float] = {}
        for tag in set(runtime_weights) | set(canonical_weights):
            runtime_value = self._coerce_float(runtime_weights.get(tag, 0.0), 0.0)
            canonical_value = self._coerce_float(canonical_weights.get(tag, 0.0), 0.0)
            mixed = (1.0 - influence) * canonical_value + influence * runtime_value
            blended[tag] = max(-1.0, min(1.0, mixed))
        return blended

    def _load_dishes(self) -> List[Dict]:
        canonical_dishes = self._load_canonical_dishes()
        legacy_dishes = self._load_legacy_dishes()
        dishes, source_file, source_kind = self._choose_source(canonical_dishes, legacy_dishes)
        self.source_file = source_file
        self.source_kind = source_kind

        canonical_by_id: Dict[str, Dict[str, float]] = {}
        for dish in canonical_dishes:
            if not isinstance(dish, dict):
                continue
            tag_weights = dish.get("tag_weights", {})
            if isinstance(tag_weights, dict):
                canonical_by_id[str(dish.get("id", ""))] = self._normalize_tag_weights(tag_weights)

        runtime_influence = self._coerce_float(
            LAYER2_CONFIG.get("learning", {}).get("runtime_influence", 0.35),
            0.35,
        )

        normalized_dishes: List[Dict] = []
        for dish in dishes:
            if not isinstance(dish, dict):
                continue
            tag_weights = dish.get("tag_weights", {})
            if not isinstance(tag_weights, dict):
                tag_weights = {}
            normalized_weights = self._normalize_tag_weights(tag_weights)
            dish_id = str(dish.get("id", ""))
            if source_kind == "runtime" and dish_id in canonical_by_id:
                normalized_weights = self._blend_runtime_with_canonical(
                    runtime_weights=normalized_weights,
                    canonical_weights=canonical_by_id[dish_id],
                    runtime_influence=runtime_influence,
                )
            normalized_dishes.append(
                {
                    "id": dish_id,
                    "name": str(dish.get("name", "Unknown dish")),
                    "is_drink": bool(dish.get("is_drink", False)),
                    "popularity_score": self._coerce_float(dish.get("popularity_score", 1.0), 1.0),
                    "tag_weights": normalized_weights,
                }
            )
        return normalized_dishes

    def _expand_context_scores(self, context_scores: Dict[str, float]) -> Dict[str, float]:
        expanded: Dict[str, float] = {}
        for tag, score in context_scores.items():
            base_score = self._coerce_float(score, 0.0)
            if base_score == 0.0:
                continue

            self._add_weight(expanded, tag, base_score)

            for similar_tag, factor in self.similarity_tags.get(tag, {}).items():
                self._add_weight(expanded, similar_tag, base_score * factor)
        return expanded

    def _expanded_context_for_update(self, context_scores: Dict[str, float]) -> Dict[str, float]:
        expanded = self._expand_context_scores(context_scores)
        for tag, score in context_scores.items():
            self._add_weight(expanded, tag, self._coerce_float(score, 0.0))
        return expanded

    def save(self) -> None:
        payload = {"dishes": self.dishes}
        self.source_file.parent.mkdir(parents=True, exist_ok=True)
        self.source_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self.source_kind = "runtime"

    def apply_context_to_dish(self, dish: Dict, context_scores: Dict[str, float], learning_rate: float, penalty_rate: float) -> None:
        expanded_context = self._expanded_context_for_update(context_scores)
        tag_weights = dish.setdefault("tag_weights", {})

        for tag in list(tag_weights.keys()):
            activation = self._coerce_float(expanded_context.get(tag, 0.0), 0.0)
            current_weight = self._coerce_float(tag_weights.get(tag, 0.0), 0.0)

            if activation >= 0.25:
                updated = current_weight + learning_rate * activation
            else:
                updated = current_weight - penalty_rate * (0.25 - activation)

            tag_weights[tag] = max(-1.0, min(1.0, updated))

    def apply_negative_feedback_to_dish(self, dish: Dict, context_scores: Dict[str, float], penalty_rate: float) -> None:
        expanded_context = self._expanded_context_for_update(context_scores)
        tag_weights = dish.setdefault("tag_weights", {})

        for tag, activation in expanded_context.items():
            current_weight = self._coerce_float(tag_weights.get(tag, 0.0), 0.0)
            updated = current_weight - penalty_rate * activation
            tag_weights[tag] = max(-1.0, min(1.0, updated))

    def score_dish(self, dish: Dict, context_scores: Dict[str, float]) -> float:
        # Cong thuc tuyen tinh:
        # score(dish) = sum_t activation(tag_t) * weight(dish, tag_t)
        expanded_context = self._expand_context_scores(context_scores)
        return float(
            sum(
                expanded_context.get(tag, 0.0) * weight
                for tag, weight in dish["tag_weights"].items()
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
                    "popularity_score": self._coerce_float(dish.get("popularity_score", 1.0), 1.0),
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
