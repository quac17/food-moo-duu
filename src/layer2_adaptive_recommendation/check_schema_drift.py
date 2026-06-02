from __future__ import annotations

import json
import sys
from pathlib import Path

from src.core.constants import ALL_TAGS, LAYER2_DATA_DIR
from src.layer2_adaptive_recommendation.recommendation_engine import RecommendationEngine


def check_layer2_schema_drift(data_dir: Path = LAYER2_DATA_DIR) -> int:
    engine = RecommendationEngine(data_dir=data_dir)
    tag_set = set(ALL_TAGS)

    missing_required_fields = 0
    invalid_tag_keys = set()
    empty_ids = 0

    for dish in engine.dishes:
        dish_id = str(dish.get("id", "")).strip()
        if not dish_id:
            empty_ids += 1

        if "name" not in dish or "tag_weights" not in dish:
            missing_required_fields += 1
            continue

        tag_weights = dish.get("tag_weights", {})
        if not isinstance(tag_weights, dict):
            missing_required_fields += 1
            continue

        for tag in tag_weights:
            if tag not in tag_set:
                invalid_tag_keys.add(tag)

    print(f"source_kind={engine.source_kind}")
    print(f"source_file={engine.source_file}")
    print(f"dish_count={len(engine.dishes)}")
    print(f"empty_ids={empty_ids}")
    print(f"missing_required_fields={missing_required_fields}")
    print(f"invalid_tag_key_count={len(invalid_tag_keys)}")

    if invalid_tag_keys:
        print("invalid_tags=" + ",".join(sorted(invalid_tag_keys)))

    has_error = bool(empty_ids or missing_required_fields or invalid_tag_keys)
    return 1 if has_error else 0


def main() -> None:
    code = check_layer2_schema_drift()
    if code != 0:
        print("Layer2 schema drift detected.")
    else:
        print("Layer2 schema drift check passed.")
    sys.exit(code)


if __name__ == "__main__":
    main()
