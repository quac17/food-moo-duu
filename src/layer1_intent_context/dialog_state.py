from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

from src.core.constants import ALL_TAGS, LAYER1_DATA_DIR


@dataclass
class SessionState:
    """Luu activation score [0.0, 1.0] cho moi tag trong phien chat."""

    tag_scores: Dict[str, float] = field(
        default_factory=lambda: {tag: 0.0 for tag in ALL_TAGS}
    )
    turn_index: int = 0


class DialogStateTracker:
    def __init__(
        self,
        decay_rate: float = 0.92,
        accumulation_alpha: float = 0.65,
        conflict_beta: float = 0.35,
        data_dir: Path | None = None,
    ) -> None:
        self.decay_rate = decay_rate
        self.accumulation_alpha = accumulation_alpha
        self.conflict_beta = conflict_beta
        self.data_dir = data_dir or LAYER1_DATA_DIR
        self.state_file = self.data_dir / "session_state.json"
        self.conflict_file = self.data_dir / "conflict_pairs.json"
        self.conflict_pairs = self._load_conflict_pairs()
        self.state = self.load_state()

    def _load_conflict_pairs(self) -> List[Tuple[str, str]]:
        payload = json.loads(self.conflict_file.read_text(encoding="utf-8"))
        return [
            (pair["left"], pair["right"])
            for pair in payload.get("conflict_pairs", [])
            if pair.get("left") and pair.get("right")
        ]

    def load_state(self) -> SessionState:
        if not self.state_file.exists():
            return SessionState()
        payload = json.loads(self.state_file.read_text(encoding="utf-8"))
        return SessionState(
            tag_scores=payload.get("tag_scores", {tag: 0.0 for tag in ALL_TAGS}),
            turn_index=payload.get("turn_index", 0),
        )

    def save_state(self) -> None:
        payload = {"tag_scores": self.state.tag_scores, "turn_index": self.state.turn_index}
        self.state_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def update_context(self, new_scores: Dict[str, float]) -> Dict[str, float]:
        self.state.turn_index += 1

        for tag, old_score in self.state.tag_scores.items():
            # Quy luat 1 - Decay: diem cu se hao mon theo tung luot hoi thoai.
            decayed = old_score * self.decay_rate
            self.state.tag_scores[tag] = max(0.0, min(1.0, decayed))

        for tag, incoming in new_scores.items():
            current = self.state.tag_scores.get(tag, 0.0)
            # Quy luat 2 - Accumulation: cong don tin hieu moi theo he so alpha.
            # Cong thuc: score_t = score_t + alpha * confidence_intent
            accumulated = current + self.accumulation_alpha * incoming
            self.state.tag_scores[tag] = max(0.0, min(1.0, accumulated))

        for left, right in self.conflict_pairs:
            left_score = self.state.tag_scores.get(left, 0.0)
            right_score = self.state.tag_scores.get(right, 0.0)
            if left_score == right_score:
                continue

            stronger = left if left_score > right_score else right
            weaker = right if stronger == left else left
            # Quy luat 3 - Conflict Resolution:
            # Khi 2 tag doi nghich cung cao, giam ben yeu theo khoang cach.
            gap = abs(left_score - right_score)
            penalized = self.state.tag_scores[weaker] - self.conflict_beta * gap
            self.state.tag_scores[weaker] = max(0.0, min(1.0, penalized))

        self.save_state()
        return dict(self.state.tag_scores)
