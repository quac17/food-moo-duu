from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from src.layer1_intent_context.dialog_state import DialogStateTracker
from src.layer1_intent_context.intent_tracker import IntentTracker
from src.layer2_adaptive_recommendation.online_learning import HebbianLearner
from src.layer2_adaptive_recommendation.recommendation_engine import RecommendationEngine
from src.layer3_genetic_response.fitness_manager import FitnessManager
from src.layer3_genetic_response.genetic_generator import GeneticGenerator


@dataclass
class TurnResult:
    user_text: str
    context_scores: Dict[str, float]
    recommendations: List[Dict]
    response: str
    chromosome_key: str


class FoodSuggestionPipeline:
    def __init__(self) -> None:
        self.intent_tracker = IntentTracker()
        self.dst = DialogStateTracker()
        self.recommendation_engine = RecommendationEngine()
        self.hebbian = HebbianLearner(self.recommendation_engine)
        self.generator = GeneticGenerator()
        self.fitness_manager = FitnessManager(self.generator)
        self.last_chromosome_key = ""

    def process_turn(self, user_text: str, top_k: int = 5) -> TurnResult:
        prediction = self.intent_tracker.predict_tags(user_text)
        context_scores = self.dst.update_context(prediction.tag_scores)
        recommendations = self.recommendation_engine.recommend(context_scores, top_k=top_k)
        generated = self.generator.generate()
        self.last_chromosome_key = generated["chromosome_key"]
        return TurnResult(
            user_text=user_text,
            context_scores=context_scores,
            recommendations=recommendations,
            response=generated["response"],
            chromosome_key=generated["chromosome_key"],
        )

    def apply_feedback(self, chosen_dish_id: str, context_scores: Dict[str, float]) -> None:
        self.hebbian.update_after_choice(chosen_dish_id=chosen_dish_id, context_scores=context_scores)
        if self.last_chromosome_key:
            self.fitness_manager.update_from_implicit_feedback(
                chromosome_key=self.last_chromosome_key,
                user_chosen_dish=True,
            )

    def apply_abandon_feedback(self) -> None:
        if self.last_chromosome_key:
            self.fitness_manager.update_from_implicit_feedback(
                chromosome_key=self.last_chromosome_key,
                user_chosen_dish=False,
            )
