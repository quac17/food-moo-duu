from __future__ import annotations

from src.layer3_genetic_response.genetic_generator import GeneticGenerator


class FitnessManager:
    """Quan ly cap nhat fitness dua tren implicit feedback."""

    def __init__(self, generator: GeneticGenerator) -> None:
        self.generator = generator

    def update_from_implicit_feedback(self, chromosome_key: str, user_chosen_dish: bool) -> None:
        self.generator.update_fitness(chromosome_key=chromosome_key, success=user_chosen_dish)
