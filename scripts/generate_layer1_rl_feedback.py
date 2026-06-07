from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.constants import LAYER1_DATA_DIR, LAYER2_DATA_DIR
from src.layer1_intent_context.rl_feedback import (
    SIMULATED_EVENTS_FILE,
    append_layer1_rl_feedback,
)


def _load_layer1_train_records(dataset_id: str) -> List[Dict]:
    file_path = LAYER1_DATA_DIR / dataset_id / "intent_train_data.json"
    payload = json.loads(file_path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, list) else []


def _load_dishes() -> List[Dict]:
    payload = json.loads((LAYER2_DATA_DIR / "dishes_100.json").read_text(encoding="utf-8"))
    return payload.get("dishes", [])


def _tags_to_scores(tags: List[str], boost: float = 0.85) -> Dict[str, float]:
    return {tag: boost for tag in tags}


def _sample_recommendations(dishes: List[Dict], k: int = 5) -> List[Dict]:
    picks = random.sample(dishes, k=min(k, len(dishes)))
    recs: List[Dict] = []
    for idx, dish in enumerate(picks):
        recs.append(
            {
                "id": dish.get("id", f"dish_{idx}"),
                "name": dish.get("name", f"Dish {idx}"),
                "score": round(1.0 - idx * 0.1, 4),
            }
        )
    return recs


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate simulated Layer1 RL feedback")
    parser.add_argument("--dataset", default="dataset_v002")
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--reward-success", type=float, default=1.0)
    parser.add_argument("--reward-failure", type=float, default=-0.4)
    args = parser.parse_args()

    random.seed(42)
    records = _load_layer1_train_records(args.dataset)
    dishes = _load_dishes()
    if not records or not dishes:
        raise ValueError("Khong tim thay du lieu de sinh RL feedback gia lap")

    total = min(args.samples, len(records))
    for idx in range(total):
        row = records[idx]
        tags = row.get("tags", [])
        text = str(row.get("text", ""))
        recs = _sample_recommendations(dishes, k=5)
        chosen = recs[0]
        success = random.random() >= 0.2
        reward_signal = args.reward_success if success else args.reward_failure

        append_layer1_rl_feedback(
            user_text=text,
            turn_index=idx + 1,
            raw_scores=_tags_to_scores(tags, boost=0.78),
            context_scores=_tags_to_scores(tags, boost=0.9 if success else 0.65),
            chosen_dish_id=str(chosen["id"]),
            chosen_dish_name=str(chosen["name"]),
            recommended_candidates=recs,
            reward_signal=reward_signal,
            session_id=f"sim_{args.dataset}",
            export_mode="context",
            use_state=True,
            source="simulator_batch",
            output_file=SIMULATED_EVENTS_FILE,
        )

    print(f"Da sinh {total} mau RL feedback gia lap vao: {SIMULATED_EVENTS_FILE}")


if __name__ == "__main__":
    main()
