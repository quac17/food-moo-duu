from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from src.core.constants import ALL_TAGS, LAYER2_DATA_DIR


class RecommendationEngine:
    """Tinh diem goi y mon an bang Linear Weight Scoring."""

    def __init__(self, data_dir: Path | None = None) -> None:
        self.data_dir = data_dir or LAYER2_DATA_DIR
        self.dishes_file = self.data_dir / "dishes_100.json"
        self.dishes = self._load_dishes()

    def _load_dishes(self) -> List[Dict]:
        payload = json.loads(self.dishes_file.read_text(encoding="utf-8"))
        dishes = payload.get("dishes", [])
        for dish in dishes:
            dish.setdefault("tag_weights", {})
            for tag in ALL_TAGS:
                dish["tag_weights"].setdefault(tag, 0.0)
        return dishes

    def save(self) -> None:
        payload = {"dishes": self.dishes}
        self.dishes_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

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
            ranked.append({"id": dish["id"], "name": dish["name"], "score": round(score, 4)})
        ranked.sort(key=lambda item: item["score"], reverse=True)
        return ranked[:top_k]

    def get_dish_by_id(self, dish_id: str) -> Dict:
        for dish in self.dishes:
            if dish["id"] == dish_id:
                return dish
        raise ValueError(f"Khong tim thay mon co id={dish_id}")
