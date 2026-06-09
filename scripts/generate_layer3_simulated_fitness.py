from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.constants import LAYER2_DATA_DIR, LAYER3_DATA_DIR
from src.layer3_genetic_response.genetic_generator import GeneticGenerator

Chromosome = Tuple[str, str, str]
SIMULATED_PAIRS_FILE = LAYER3_DATA_DIR / "dataset_v002" / "response_train_pairs_simulated.jsonl"


def _load_dishes() -> List[Dict[str, object]]:
    payload = json.loads((LAYER2_DATA_DIR / "dishes_100.json").read_text(encoding="utf-8"))
    dishes = payload.get("dishes", [])
    return dishes if isinstance(dishes, list) else []


def _chromosomes_for_mood(generator: GeneticGenerator, mood_key: str) -> List[Chromosome]:
    pool = generator.mood_pool_map.get(mood_key)
    if not pool:
        return []
    chromosomes: List[Chromosome] = []
    for opening in pool["opening"]:
        for action in pool["action"]:
            for closing in pool["closing"]:
                chromosomes.append((opening, action, closing))
    return chromosomes


def _template_from_chromosome(chromosome: Chromosome) -> str:
    return " ".join(part.strip() for part in chromosome if part.strip())


def _tags_for_mood(mood_key: str) -> Dict[str, float]:
    presets: Dict[str, Dict[str, float]] = {
        "mood_stressed": {"mood_stressed": 0.82, "time_evening": 0.68, "pref_soup": 0.55},
        "mood_sick": {"mood_sick": 0.86, "pref_soft": 0.72, "weather_rain": 0.48},
        "mood_gathering": {"mood_gathering": 0.8, "time_evening": 0.65, "pref_finger_food": 0.58},
    }
    return presets.get(mood_key, {mood_key: 0.75})


def generate_fitness_history(samples: int, success_rate: float, seed: int) -> int:
    random.seed(seed)
    generator = GeneticGenerator()
    mood_keys = list(generator.mood_pool_map.keys())
    if not mood_keys:
        raise ValueError("Khong tim thay mood pool trong gene_pool.json")

    before = len(generator.fitness_payload.get("history", []))
    for _ in range(samples):
        mood_key = random.choice(mood_keys)
        chromosomes = _chromosomes_for_mood(generator, mood_key)
        if not chromosomes:
            continue
        chromosome = random.choice(chromosomes)
        chromosome_key = generator._key(chromosome)
        success = random.random() < success_rate
        generator.update_fitness(chromosome_key=chromosome_key, success=success, source="simulated")

    after = len(generator.fitness_payload.get("history", []))
    return after - before


def generate_response_pairs(samples: int, seed: int) -> int:
    random.seed(seed + 7)
    dishes = _load_dishes()
    if not dishes:
        raise ValueError("Khong tim thay dishes_100.json")

    generator = GeneticGenerator()
    mood_keys = list(generator.mood_pool_map.keys())
    SIMULATED_PAIRS_FILE.parent.mkdir(parents=True, exist_ok=True)

    existing_ids = set()
    if SIMULATED_PAIRS_FILE.exists():
        for line in SIMULATED_PAIRS_FILE.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            existing_ids.add(str(row.get("sample_id", "")))

    start_idx = len(existing_ids) + 1
    rows: List[str] = []
    for offset in range(samples):
        mood_key = random.choice(mood_keys)
        chromosomes = _chromosomes_for_mood(generator, mood_key)
        if not chromosomes:
            continue
        chromosome = random.choice(chromosomes)
        dish_ids = [str(item.get("id", "")) for item in random.sample(dishes, k=min(3, len(dishes)))]
        success = random.random() < 0.58
        sample_id = f"l3s_{start_idx + offset:04d}"
        payload = {
            "sample_id": sample_id,
            "mood_tag": mood_key,
            "input_tags": _tags_for_mood(mood_key),
            "recommended_dishes": dish_ids,
            "response_template": _template_from_chromosome(chromosome),
            "chromosome_key": generator._key(chromosome),
            "feedback": "success" if success else "failure",
            "source": "simulated",
        }
        rows.append(json.dumps(payload, ensure_ascii=False))

    mode = "a" if SIMULATED_PAIRS_FILE.exists() else "w"
    with SIMULATED_PAIRS_FILE.open(mode, encoding="utf-8") as handle:
        if rows:
            handle.write("\n".join(rows) + "\n")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sinh du lieu gia lap fitness Layer 3")
    parser.add_argument("--fitness-samples", type=int, default=50, help="So luot cap nhat fitness moi")
    parser.add_argument("--pair-samples", type=int, default=40, help="So cap response_train_pairs gia lap")
    parser.add_argument("--success-rate", type=float, default=0.58, help="Ty le implicit success")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    fitness_added = generate_fitness_history(
        samples=max(0, args.fitness_samples),
        success_rate=args.success_rate,
        seed=args.seed,
    )
    pairs_added = generate_response_pairs(samples=max(0, args.pair_samples), seed=args.seed)

    fitness_file = LAYER3_DATA_DIR / "dataset_v001" / "fitness_history.json"
    print(f"Da them {fitness_added} mau fitness gia lap vao: {fitness_file}")
    print(f"Da them {pairs_added} cap response gia lap vao: {SIMULATED_PAIRS_FILE}")


if __name__ == "__main__":
    main()
