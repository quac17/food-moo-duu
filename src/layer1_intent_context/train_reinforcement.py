from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from src.layer1_intent_context.rl_feedback import (
    REAL_EVENTS_FILE,
    RL_TRAINING_DIR,
    SIMULATED_EVENTS_FILE,
    load_feedback_events,
)


def _convert_event_to_rl_row(event: Dict) -> Dict:
    user_text = str(event.get("user_text", ""))
    reward = float(event.get("reward_signal", 0.0))
    raw_tags = event.get("raw_tags", [])
    context_tags = event.get("context_tags", [])

    # Weighted merge:
    # context mang thong tin DST, raw mang thong tin cau hien tai.
    # Ta uu tien context 70%, raw 30% de tao nhan tang cuong cho Layer1.
    merged: Dict[str, float] = {}
    for item in context_tags:
        tag = str(item.get("tag", "")).strip()
        if not tag:
            continue
        merged[tag] = merged.get(tag, 0.0) + 0.7 * float(item.get("score", 0.0))
    for item in raw_tags:
        tag = str(item.get("tag", "")).strip()
        if not tag:
            continue
        merged[tag] = merged.get(tag, 0.0) + 0.3 * float(item.get("score", 0.0))

    adjusted = {tag: round(max(0.0, score + reward * 0.05), 4) for tag, score in merged.items()}
    return {
        "text": user_text,
        "reward_signal": reward,
        "chosen_dish_id": event.get("chosen_dish_id", ""),
        "tags_weighted": adjusted,
    }


def _to_intent_train_samples(rows: List[Dict], threshold: float) -> List[Dict]:
    samples: List[Dict] = []
    for row in rows:
        tags = [tag for tag, score in row["tags_weighted"].items() if score >= threshold]
        if not tags:
            continue
        samples.append({"text": row["text"], "tags": tags})
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline RL trainer for Layer1")
    parser.add_argument("--include-simulated", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.18)
    args = parser.parse_args()

    files = [REAL_EVENTS_FILE]
    if args.include_simulated:
        files.append(SIMULATED_EVENTS_FILE)
    events = load_feedback_events(files)
    if not events:
        raise ValueError("Khong co RL feedback events de train offline.")

    rl_rows = [_convert_event_to_rl_row(event) for event in events]
    train_samples = _to_intent_train_samples(rl_rows, threshold=args.threshold)

    RL_TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    rl_rows_path = RL_TRAINING_DIR / "reinforcement_rows.json"
    train_samples_path = RL_TRAINING_DIR / "intent_train_data_rl.json"
    stats_path = RL_TRAINING_DIR / "stats.json"

    rl_rows_path.write_text(json.dumps(rl_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    train_samples_path.write_text(
        json.dumps(train_samples, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    stats = {
        "events_total": len(events),
        "rows_total": len(rl_rows),
        "train_samples_total": len(train_samples),
        "threshold": args.threshold,
        "include_simulated": args.include_simulated,
    }
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"RL offline processing done. stats={stats}")
    print(f"- rows: {rl_rows_path}")
    print(f"- train_samples: {train_samples_path}")
    print(f"- stats: {stats_path}")


if __name__ == "__main__":
    main()
