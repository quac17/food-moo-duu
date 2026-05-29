from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

from src.core.constants import LAYER3_DATA_DIR


Chromosome = Tuple[str, str, str]


class GeneticGenerator:
    """Sinh cau thoai bang template + GA + epsilon-greedy."""

    def __init__(
        self,
        data_dir: Path | None = None,
        epsilon: float = 0.2,
        mutation_rate: float = 0.3,
    ) -> None:
        self.data_dir = data_dir or LAYER3_DATA_DIR
        self.gene_pool_file = self.data_dir / "gene_pool.json"
        self.fitness_file = self.data_dir / "fitness_history.json"
        self.epsilon = epsilon
        self.mutation_rate = mutation_rate
        self.pool = json.loads(self.gene_pool_file.read_text(encoding="utf-8"))
        self.fitness_payload = json.loads(self.fitness_file.read_text(encoding="utf-8"))

    def _key(self, chromosome: Chromosome) -> str:
        return " | ".join(chromosome)

    def _roulette_select(self, chromosomes: List[Chromosome]) -> Chromosome:
        scored = []
        total = 0.0
        for chromosome in chromosomes:
            key = self._key(chromosome)
            fit = float(self.fitness_payload["fitness"].get(key, 1.0))
            fit = max(0.01, fit)
            scored.append((chromosome, fit))
            total += fit

        pick = random.uniform(0.0, total)
        upto = 0.0
        for chromosome, fit in scored:
            upto += fit
            if upto >= pick:
                return chromosome
        return scored[-1][0]

    def _mutate_text(self, text: str) -> str:
        slang_tokens = self.pool.get("slang_mutations", [])
        if slang_tokens and random.random() < self.mutation_rate:
            return f"{text} {random.choice(slang_tokens)}"
        return text

    def _build_population(self, size: int = 8) -> List[Chromosome]:
        openings = self.pool["opening"]
        actions = self.pool["action"]
        closings = self.pool["closing"]
        population: List[Chromosome] = []
        for _ in range(size):
            population.append(
                (
                    random.choice(openings),
                    random.choice(actions),
                    random.choice(closings),
                )
            )
        return population

    def generate(self) -> Dict[str, str]:
        population = self._build_population()
        if random.random() < self.epsilon:
            chromosome = random.choice(population)
        else:
            chromosome = self._roulette_select(population)

        opening, action, closing = chromosome
        sentence = " ".join(
            [
                self._mutate_text(opening),
                self._mutate_text(action),
                self._mutate_text(closing),
            ]
        )
        return {"chromosome_key": self._key(chromosome), "response": sentence.strip()}

    def update_fitness(self, chromosome_key: str, success: bool) -> None:
        fitness = self.fitness_payload.setdefault("fitness", {})
        current = float(fitness.get(chromosome_key, 1.0))

        # Ham fitness toi gian theo phan hoi ngam:
        # success -> cong diem, fail -> tru diem va chan gia tri toi thieu de tranh =0.
        if success:
            updated = current + 0.25
        else:
            updated = max(0.05, current - 0.2)

        fitness[chromosome_key] = round(updated, 4)
        history = self.fitness_payload.setdefault("history", [])
        history.append(
            {"chromosome_key": chromosome_key, "success": success, "fitness": fitness[chromosome_key]}
        )
        self.fitness_file.write_text(
            json.dumps(self.fitness_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
