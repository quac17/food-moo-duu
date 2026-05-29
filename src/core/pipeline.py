from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from src.core.constants import HYPERPARAMS
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
        learning_rate = HYPERPARAMS["learning_rate"]
        punishment_rate = abs(HYPERPARAMS["punishment_rate"])
        context_decay = HYPERPARAMS["context_decay"]
        epsilon = HYPERPARAMS["epsilon"]

        self.intent_tracker = IntentTracker()
        self.dst = DialogStateTracker(decay_rate=context_decay)
        self.recommendation_engine = RecommendationEngine()
        self.hebbian = HebbianLearner(
            self.recommendation_engine,
            lr_positive=learning_rate,
            lr_negative=punishment_rate,
        )
        self.generator = None
        self.fitness_manager = None
        try:
            self.generator = GeneticGenerator(epsilon=epsilon)
            self.fitness_manager = FitnessManager(self.generator)
        except Exception:
            # Layer 3 co the chua dong bo schema; pipeline van phuc vu duoc layer1+2.
            self.generator = None
            self.fitness_manager = None
        self.last_chromosome_key = ""

    def process_turn(self, user_text: str, top_k: int = 5) -> TurnResult:
        prediction = self.intent_tracker.predict_tags(user_text)
        context_scores = self.dst.update_context(prediction.tag_scores)
        recommendations = self.recommendation_engine.recommend(context_scores, top_k=top_k)
        if self.generator is not None:
            try:
                generated = self.generator.generate()
                self.last_chromosome_key = generated["chromosome_key"]
                response = generated["response"]
                chromosome_key = generated["chromosome_key"]
            except Exception:
                # Fallback neu du lieu Layer3 chua dung schema ma generator mong doi.
                response = "Da cap nhat ngu canh va goi y mon an theo thong tin moi."
                chromosome_key = ""
                self.last_chromosome_key = ""
        else:
            response = "Da cap nhat ngu canh va goi y mon an theo thong tin moi."
            chromosome_key = ""
            self.last_chromosome_key = ""
        return TurnResult(
            user_text=user_text,
            context_scores=context_scores,
            recommendations=recommendations,
            response=response,
            chromosome_key=chromosome_key,
        )

    def apply_feedback(self, chosen_dish_id: str, context_scores: Dict[str, float]) -> None:
        self.hebbian.update_after_choice(chosen_dish_id=chosen_dish_id, context_scores=context_scores)
        if self.last_chromosome_key and self.fitness_manager is not None:
            self.fitness_manager.update_from_implicit_feedback(
                chromosome_key=self.last_chromosome_key,
                user_chosen_dish=True,
            )

    def apply_abandon_feedback(self) -> None:
        if self.last_chromosome_key and self.fitness_manager is not None:
            self.fitness_manager.update_from_implicit_feedback(
                chromosome_key=self.last_chromosome_key,
                user_chosen_dish=False,
            )
