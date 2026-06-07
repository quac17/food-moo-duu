from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from src.core.constants import LAYER1_DATA_DIR


RL_FEEDBACK_DIR = LAYER1_DATA_DIR / "rl_feedback"
REAL_EVENTS_FILE = RL_FEEDBACK_DIR / "selected_dish_events.jsonl"
SIMULATED_EVENTS_FILE = RL_FEEDBACK_DIR / "selected_dish_events_simulated.jsonl"
RL_TRAINING_DIR = LAYER1_DATA_DIR / "rl_training"


def _ensure_parent(file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)


def _to_tag_list(scores: Dict[str, float], top_k: int = 12, threshold: float = 0.05) -> List[Dict]:
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    filtered = [(tag, score) for tag, score in ranked if float(score) >= threshold][:top_k]
    return [{"tag": tag, "score": round(float(score), 4)} for tag, score in filtered]


def append_layer1_rl_feedback(
    user_text: str,
    turn_index: int,
    raw_scores: Dict[str, float],
    context_scores: Dict[str, float],
    chosen_dish_id: str,
    chosen_dish_name: str,
    recommended_candidates: List[Dict],
    reward_signal: float = 1.0,
    session_id: str | None = None,
    export_mode: str = "context",
    use_state: bool = True,
    source: str = "chatbot_runtime",
    output_file: Path = REAL_EVENTS_FILE,
) -> None:
    _ensure_parent(output_file)
    payload = {
        "event_id": str(uuid.uuid4()),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "session_id": session_id or "session_default",
        "turn_index": int(turn_index),
        "source": source,
        "user_text": user_text,
        "raw_tags": _to_tag_list(raw_scores),
        "context_tags": _to_tag_list(context_scores),
        "chosen_dish_id": chosen_dish_id,
        "chosen_dish_name": chosen_dish_name,
        "reward_signal": float(reward_signal),
        "recommended_candidates": [
            {
                "id": str(item.get("id", "")),
                "name": str(item.get("name", "")),
                "score": round(float(item.get("score", 0.0)), 4),
            }
            for item in recommended_candidates
        ],
        "export_mode": export_mode,
        "use_state": bool(use_state),
    }
    with output_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def load_feedback_events(files: List[Path]) -> List[Dict]:
    events: List[Dict] = []
    for file_path in files:
        if not file_path.exists():
            continue
        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return events
