from __future__ import annotations

from typing import Dict

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
        self.engine.apply_context_to_dish(
            dish,
            context_scores=context_scores,
            learning_rate=self.lr_positive,
            penalty_rate=self.lr_negative,
        )

        self.engine.save()
