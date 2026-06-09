from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from src.core.constants import HYPERPARAMS, LAYER2_CONFIG
from src.layer1_intent_context.dialog_state import DialogStateTracker
from src.layer1_intent_context.intent_tracker import IntentTracker
from src.layer2_adaptive_recommendation.online_learning import HebbianLearner
from src.layer2_adaptive_recommendation.recommendation_engine import RecommendationEngine
from src.layer3_genetic_response.fitness_manager import FitnessManager
from src.layer3_genetic_response.genetic_generator import GeneticGenerator


@dataclass
class TurnResult:
    user_text: str
    raw_scores: Dict[str, float]
    context_scores: Dict[str, float]
    recommendations: List[Dict]
    response: str
    chromosome_key: str


@dataclass
class FeedbackReport:
    chosen_dish_id: str
    chosen_dish_name: str
    score_before: float
    score_after: float
    delta: float


class FoodSuggestionPipeline:
    def __init__(self) -> None:
        learning_rate = LAYER2_CONFIG["learning"]["positive"]
        punishment_rate = abs(LAYER2_CONFIG["learning"]["negative"])
        epsilon = HYPERPARAMS["epsilon"]

        # Layer1 da dung DL vi-SBERT + metric learning, API predict_tags giu nguyen.
        self.intent_tracker = IntentTracker()
        self.dst = DialogStateTracker(
            decay_rate=HYPERPARAMS["context_decay"],
            time_decay_rate=HYPERPARAMS["context_decay_time"],
            accumulation_alpha=HYPERPARAMS["context_accumulation_alpha"],
            conflict_beta=HYPERPARAMS["context_conflict_beta"],
        )
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
        self.last_recommendations: List[Dict] = []

    def reset_session_state(self) -> None:
        self.dst.reset_state()

    def process_turn(self, user_text: str, top_k: int = 5) -> TurnResult:
        prediction = self.intent_tracker.predict_tags(user_text)
        context_scores = self.dst.update_context(prediction.tag_scores)
        recommendations = self.recommendation_engine.recommend(context_scores, top_k=top_k)
        self.last_recommendations = recommendations
        top_foods = ", ".join(item.get("name", "") for item in recommendations[:2] if item.get("name")) or "mot vai mon phu hop"
        if self.generator is not None:
            try:
                generated = self.generator.generate(context_scores=prediction.tag_scores)
                self.last_chromosome_key = generated["chromosome_key"]
                response = generated["response"].replace("{foods}", top_foods)
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
            raw_scores=prediction.tag_scores,
            context_scores=context_scores,
            recommendations=recommendations,
            response=response,
            chromosome_key=chromosome_key,
        )

    def apply_feedback(self, chosen_dish_id: str, context_scores: Dict[str, float]) -> FeedbackReport:
        dish = self.recommendation_engine.get_dish_by_id(chosen_dish_id)
        score_before = self.recommendation_engine.score_dish(dish, context_scores)

        self.hebbian.update_after_choice(chosen_dish_id=chosen_dish_id, context_scores=context_scores)

        for recommended in self.last_recommendations:
            recommended_id = str(recommended.get("id", ""))
            if not recommended_id or recommended_id == chosen_dish_id:
                continue

            try:
                non_selected_dish = self.recommendation_engine.get_dish_by_id(recommended_id)
            except ValueError:
                continue

            self.recommendation_engine.apply_negative_feedback_to_dish(
                non_selected_dish,
                context_scores=context_scores,
                penalty_rate=self.hebbian.lr_negative,
            )

        self.recommendation_engine.save()

        updated_dish = self.recommendation_engine.get_dish_by_id(chosen_dish_id)
        score_after = self.recommendation_engine.score_dish(updated_dish, context_scores)

        if self.last_chromosome_key and self.fitness_manager is not None:
            self.fitness_manager.update_from_implicit_feedback(
                chromosome_key=self.last_chromosome_key,
                user_chosen_dish=True,
            )

        return FeedbackReport(
            chosen_dish_id=chosen_dish_id,
            chosen_dish_name=str(updated_dish.get("name", chosen_dish_id)),
            score_before=round(score_before, 4),
            score_after=round(score_after, 4),
            delta=round(score_after - score_before, 4),
        )

    def apply_abandon_feedback(self) -> None:
        if self.last_chromosome_key and self.fitness_manager is not None:
            self.fitness_manager.update_from_implicit_feedback(
                chromosome_key=self.last_chromosome_key,
                user_chosen_dish=False,
            )
