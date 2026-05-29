from __future__ import annotations

from typing import Dict

from src.core.constants import ALL_TAGS
from src.layer2_adaptive_recommendation.recommendation_engine import RecommendationEngine


class HebbianLearner:
    """Cap nhat trong so mon-tag theo co che hoc truc tuyen Hebbian."""

    def __init__(
        self,
        engine: RecommendationEngine,
        lr_positive: float = 0.08,
        lr_negative: float = 0.02,
    ) -> None:
        self.engine = engine
        self.lr_positive = lr_positive
        self.lr_negative = lr_negative

    def update_after_choice(self, chosen_dish_id: str, context_scores: Dict[str, float]) -> None:
        dish = self.engine.get_dish_by_id(chosen_dish_id)
        tag_weights = dish["tag_weights"]

        for tag in ALL_TAGS:
            activation = context_scores.get(tag, 0.0)
            current_weight = tag_weights.get(tag, 0.0)

            # Hebbian positive: neuron-context va neuron-dish "ban cung nhau" thi lien ket tang.
            # Delta w+ = lr_positive * activation(tag)
            if activation >= 0.25:
                updated = current_weight + self.lr_positive * activation
            else:
                # Hebbian negative/forgetting nhe de tranh drift:
                # Delta w- = lr_negative * (nguong - activation), neu activation qua thap.
                updated = current_weight - self.lr_negative * (0.25 - activation)

            tag_weights[tag] = max(-1.0, min(1.0, updated))

        self.engine.save()
