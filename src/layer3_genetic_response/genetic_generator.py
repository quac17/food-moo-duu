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
        self.dataset_meta_file = self.data_dir / "datasets.json"
        self.dataset_dir = self._resolve_dataset_dir()
        self.gene_pool_file = self._resolve_data_file("gene_pool.json")
        self.fitness_file = self._resolve_data_file("fitness_history.json")
        self.epsilon = epsilon
        self.mutation_rate = mutation_rate
        self.raw_pool = json.loads(self.gene_pool_file.read_text(encoding="utf-8"))
        self.pool = self._normalize_pool(self.raw_pool)
        self.fitness_payload = json.loads(self.fitness_file.read_text(encoding="utf-8"))
        self.fitness_map = self._normalize_fitness(self.fitness_payload)

    def _resolve_dataset_dir(self) -> Path:
        if not self.dataset_meta_file.exists():
            return self.data_dir
        payload = json.loads(self.dataset_meta_file.read_text(encoding="utf-8"))
        active_dataset = payload.get("active_dataset")
        if not active_dataset:
            return self.data_dir
        return self.data_dir / active_dataset

    def _resolve_data_file(self, file_name: str) -> Path:
        dataset_file = self.dataset_dir / file_name
        if dataset_file.exists():
            return dataset_file
        return self.data_dir / file_name

    def _normalize_pool(self, payload: Dict) -> Dict[str, List[str]]:
        # Ho tro ca schema cu (opening/action/closing) va schema common_v1 (theo mood).
        if {"opening", "action", "closing"}.issubset(payload.keys()):
            return {
                "opening": [str(item) for item in payload.get("opening", [])],
                "action": [str(item) for item in payload.get("action", [])],
                "closing": [str(item) for item in payload.get("closing", [])],
                "slang_mutations": [str(item) for item in payload.get("slang_mutations", [])],
            }

        openings: List[str] = []
        actions: List[str] = []
        closings: List[str] = []
        mutations: List[str] = []
        for key, value in payload.items():
            if key == "schema_version" or not isinstance(value, dict):
                continue

            for item in value.get("Opening", []):
                text = str(item.get("text", "")).strip()
                if text:
                    openings.append(text)
            for item in value.get("Action", []):
                text = str(item.get("text", "")).strip()
                if text:
                    actions.append(text)
            for item in value.get("Closing", []):
                text = str(item.get("text", "")).strip()
                if text:
                    closings.append(text)
            for item in value.get("Mutation", []):
                text = str(item).strip()
                if text:
                    mutations.append(text)

        if not openings or not actions or not closings:
            raise ValueError("Layer3 gene pool khong co du du lieu Opening/Action/Closing")

        return {
            "opening": openings,
            "action": actions,
            "closing": closings,
            "slang_mutations": mutations,
        }

    def _normalize_fitness(self, payload: Dict) -> Dict[str, float]:
        if isinstance(payload.get("fitness"), dict):
            return {
                str(key): float(value)
                for key, value in payload.get("fitness", {}).items()
                if isinstance(value, (int, float))
            }

        runtime_fitness = payload.get("runtime_fitness")
        if isinstance(runtime_fitness, dict):
            return {
                str(key): float(value)
                for key, value in runtime_fitness.items()
                if isinstance(value, (int, float))
            }

        fitness_map: Dict[str, float] = {}
        for key, value in payload.items():
            if key == "schema_version" or not isinstance(value, dict):
                continue
            fitness_score = value.get("fitness_score")
            if isinstance(fitness_score, (int, float)):
                fitness_map[str(key)] = float(fitness_score)
        return fitness_map

    def _key(self, chromosome: Chromosome) -> str:
        return " | ".join(chromosome)

    def _roulette_select(self, chromosomes: List[Chromosome]) -> Chromosome:
        scored = []
        total = 0.0
        for chromosome in chromosomes:
            key = self._key(chromosome)
            fit = float(self.fitness_map.get(key, 1.0))
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
        current = float(self.fitness_map.get(chromosome_key, 1.0))

        # Ham fitness toi gian theo phan hoi ngam:
        # success -> cong diem, fail -> tru diem va chan gia tri toi thieu de tranh =0.
        if success:
            updated = current + 0.25
        else:
            updated = max(0.05, current - 0.2)

        self.fitness_map[chromosome_key] = round(updated, 4)
        history = self.fitness_payload.setdefault("history", [])
        history.append(
            {
                "chromosome_key": chromosome_key,
                "success": success,
                "fitness": self.fitness_map[chromosome_key],
            }
        )
        # Luu runtime_fitness rieng de khong pha vo schema canonical ban dau.
        self.fitness_payload["runtime_fitness"] = self.fitness_map
        self.fitness_file.write_text(
            json.dumps(self.fitness_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
