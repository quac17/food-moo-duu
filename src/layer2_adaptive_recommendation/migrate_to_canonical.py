from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from src.core.constants import LAYER2_DATA_DIR
from src.layer2_adaptive_recommendation.recommendation_engine import RecommendationEngine


def _load_legacy_dishes(legacy_file: Path) -> List[Dict]:
    if not legacy_file.exists():
        return []
    payload = json.loads(legacy_file.read_text(encoding="utf-8"))
    dishes = payload.get("dishes", []) if isinstance(payload, dict) else []
    return dishes if isinstance(dishes, list) else []


def migrate_layer2_to_canonical(data_dir: Path = LAYER2_DATA_DIR) -> int:
    engine = RecommendationEngine(data_dir=data_dir)

    legacy_file = engine.legacy_file
    canonical_file = engine.canonical_file
    canonical_file.parent.mkdir(parents=True, exist_ok=True)

    legacy_dishes = _load_legacy_dishes(legacy_file)
    if not legacy_dishes:
        print("Khong tim thay legacy dishes de migrate.")
        return 0

    normalized: List[Dict] = []
    for dish in legacy_dishes:
        if not isinstance(dish, dict):
            continue
        tag_weights = dish.get("tag_weights", {})
        if not isinstance(tag_weights, dict):
            tag_weights = {}
        normalized.append(
            {
                "id": str(dish.get("id", "")),
                "name": str(dish.get("name", "Unknown dish")),
                "is_drink": bool(dish.get("is_drink", False)),
                "popularity_score": float(dish.get("popularity_score", 1.0)),
                "tag_weights": engine._normalize_tag_weights(tag_weights),
            }
        )

    canonical_file.write_text(
        json.dumps({"dishes": normalized}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Da migrate {len(normalized)} mon sang canonical: {canonical_file}")
    return len(normalized)


def main() -> None:
    migrate_layer2_to_canonical()


if __name__ == "__main__":
    main()
