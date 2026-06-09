from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from src.core.constants import LAYER3_DATA_DIR


def _resolve_fitness_file() -> Path:
    meta_file = LAYER3_DATA_DIR / "datasets.json"
    if meta_file.exists():
        payload = json.loads(meta_file.read_text(encoding="utf-8"))
        active = payload.get("active_dataset")
        if active:
            dataset_file = LAYER3_DATA_DIR / active / "fitness_history.json"
            if dataset_file.exists():
                return dataset_file
    dataset_file = LAYER3_DATA_DIR / "dataset_v001" / "fitness_history.json"
    if dataset_file.exists():
        return dataset_file
    return LAYER3_DATA_DIR / "fitness_history.json"


def _history_source(item: Dict[str, object]) -> str:
    source = str(item.get("source", "runtime")).strip().lower()
    return source if source else "runtime"


def _split_history_metrics(history: List[Dict[str, object]]) -> Dict[str, object]:
    buckets: Dict[str, Dict[str, int]] = {
        "runtime": {"successes": 0, "failures": 0},
        "simulated": {"successes": 0, "failures": 0},
        "all": {"successes": 0, "failures": 0},
    }
    for item in history:
        if not isinstance(item, dict):
            continue
        source = _history_source(item)
        if source not in buckets:
            source = "runtime"
        bucket = buckets[source]
        all_bucket = buckets["all"]
        if bool(item.get("success")):
            bucket["successes"] += 1
            all_bucket["successes"] += 1
        else:
            bucket["failures"] += 1
            all_bucket["failures"] += 1

    def _rate(bucket: Dict[str, int]) -> float:
        total = bucket["successes"] + bucket["failures"]
        return round(bucket["successes"] / total, 4) if total else 0.0

    return {
        "runtime_updates": buckets["runtime"]["successes"] + buckets["runtime"]["failures"],
        "simulated_updates": buckets["simulated"]["successes"] + buckets["simulated"]["failures"],
        "runtime_success_rate": _rate(buckets["runtime"]),
        "simulated_success_rate": _rate(buckets["simulated"]),
    }


def evaluate_layer3_fitness() -> Dict[str, object]:
    fitness_file = _resolve_fitness_file()
    if not fitness_file.exists():
        return {"error": f"Khong tim thay fitness history: {fitness_file}"}

    payload = json.loads(fitness_file.read_text(encoding="utf-8"))
    history = [item for item in payload.get("history", []) if isinstance(item, dict)]
    runtime_fitness = payload.get("runtime_fitness", {})

    successes = 0
    failures = 0
    gains: List[float] = []
    for item in history:
        if bool(item.get("success")):
            successes += 1
            gains.append(0.25)
        else:
            failures += 1

    fitness_values = [float(value) for value in runtime_fitness.values()] if isinstance(runtime_fitness, dict) else []
    top_chromosomes = []
    if isinstance(runtime_fitness, dict):
        ranked = sorted(runtime_fitness.items(), key=lambda pair: pair[1], reverse=True)[:5]
        top_chromosomes = [
            {"chromosome_key": key, "fitness": round(float(value), 4)}
            for key, value in ranked
        ]

    total_updates = successes + failures
    split_metrics = _split_history_metrics(history)
    return {
        "source_file": str(fitness_file),
        "total_updates": total_updates,
        "success_rate": round(successes / total_updates, 4) if total_updates else 0.0,
        "failure_rate": round(failures / total_updates, 4) if total_updates else 0.0,
        "unique_chromosomes": len(runtime_fitness) if isinstance(runtime_fitness, dict) else 0,
        "avg_fitness": round(sum(fitness_values) / len(fitness_values), 4) if fitness_values else 0.0,
        "max_fitness": round(max(fitness_values), 4) if fitness_values else 0.0,
        "min_fitness": round(min(fitness_values), 4) if fitness_values else 0.0,
        "avg_fitness_gain_per_success": round(sum(gains) / len(gains), 4) if gains else 0.0,
        "top_chromosomes": top_chromosomes,
        **split_metrics,
    }


def save_layer3_outputs(output_dir: Path, metrics: Dict[str, object]) -> None:
    layer_dir = output_dir / "layer3"
    layer_dir.mkdir(parents=True, exist_ok=True)
    (layer_dir / "fitness_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
