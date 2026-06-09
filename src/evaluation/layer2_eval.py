from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.core.constants import LAYER1_DATA_DIR, LAYER2_DATA_DIR
from src.evaluation.metrics import aggregate_ranking_metrics
from src.layer1_intent_context.rl_feedback import (
    REAL_EVENTS_FILE,
    SIMULATED_EVENTS_FILE,
    load_feedback_events,
)
from src.layer2_adaptive_recommendation.recommendation_engine import RecommendationEngine


def _load_intent_samples(use_all_datasets: bool) -> List[Dict[str, object]]:
    meta_file = LAYER1_DATA_DIR / "datasets.json"
    payload = json.loads(meta_file.read_text(encoding="utf-8"))
    datasets = payload.get("available_datasets", []) if use_all_datasets else [payload.get("active_dataset", "dataset_v001")]
    records: List[Dict[str, object]] = []
    for dataset_name in datasets:
        csv_file = LAYER1_DATA_DIR / dataset_name / "intent_samples.csv"
        if not csv_file.exists():
            continue
        frame = pd.read_csv(csv_file)
        for _, row in frame.iterrows():
            text = str(row.get("text", "")).strip()
            tags = [tag.strip() for tag in str(row.get("tags", "")).split("|") if tag.strip()]
            if text and tags:
                records.append({"text": text, "tags": tags})
    return records


def _context_from_tags(tags: List[str], score: float = 1.0) -> Dict[str, float]:
    return {tag: score for tag in tags}


def _oracle_relevant_dishes(engine: RecommendationEngine, context_scores: Dict[str, float], top_n: int = 3) -> List[str]:
    ranked = []
    for dish in engine.dishes:
        ranked.append((dish["id"], engine.score_dish(dish, context_scores)))
    ranked.sort(key=lambda item: item[1], reverse=True)
    return [dish_id for dish_id, _ in ranked[:top_n]]


def evaluate_layer2_oracle(use_all_datasets: bool, k_values: List[int] | None = None) -> Dict[str, object]:
    k_values = k_values or [3, 5]
    engine = RecommendationEngine(prefer_canonical=True)
    samples = _load_intent_samples(use_all_datasets)

    ranked_lists: List[List[str]] = []
    relevant_lists: List[List[str]] = []
    max_k = max(k_values)

    for sample in samples:
        tags = sample["tags"]  # type: ignore[index]
        context_scores = _context_from_tags(tags)  # type: ignore[arg-type]
        relevant = _oracle_relevant_dishes(engine, context_scores, top_n=3)
        recommendations = engine.recommend(context_scores, top_k=max_k)
        ranked_lists.append([str(item["id"]) for item in recommendations])
        relevant_lists.append(relevant)

    metrics = aggregate_ranking_metrics(ranked_lists, relevant_lists, k_values=k_values)
    return {
        "mode": "oracle_tags",
        "source": "intent_samples.csv",
        "matrix": "canonical",
        "samples": len(samples),
        "relevant_top_n": 3,
        "metrics": metrics,
    }


def _event_context_scores(event: Dict[str, object]) -> Dict[str, float]:
    context_tags = event.get("context_tags", [])
    if not isinstance(context_tags, list):
        return {}
    scores: Dict[str, float] = {}
    for item in context_tags:
        if not isinstance(item, dict):
            continue
        tag = str(item.get("tag", "")).strip()
        if tag:
            scores[tag] = float(item.get("score", 0.0))
    return scores


def evaluate_layer2_behavioral(include_simulated: bool, top_k: int = 5) -> Dict[str, object]:
    files = [REAL_EVENTS_FILE]
    if include_simulated:
        files.append(SIMULATED_EVENTS_FILE)
    events = load_feedback_events(files)
    engine = RecommendationEngine(prefer_canonical=True)

    ranked_lists: List[List[str]] = []
    relevant_lists: List[List[str]] = []
    chosen_scores: List[float] = []

    for event in events:
        chosen_id = str(event.get("chosen_dish_id", "")).strip()
        if not chosen_id:
            continue
        context_scores = _event_context_scores(event)
        if not context_scores:
            continue
        recommendations = engine.recommend(context_scores, top_k=top_k)
        ranked = [str(item["id"]) for item in recommendations]
        ranked_lists.append(ranked)
        relevant_lists.append([chosen_id])
        for item in recommendations:
            if str(item.get("id", "")) == chosen_id:
                chosen_scores.append(float(item.get("score", 0.0)))
                break

    metrics = aggregate_ranking_metrics(ranked_lists, relevant_lists, k_values=[3, top_k])
    if chosen_scores:
        metrics["avg_chosen_score"] = round(sum(chosen_scores) / len(chosen_scores), 4)
    return {
        "mode": "behavioral_feedback",
        "source": [str(path.name) for path in files if path.exists()],
        "matrix": "canonical",
        "samples": len(ranked_lists),
        "metrics": metrics,
    }


def save_layer2_outputs(output_dir: Path, oracle: Dict[str, object], behavioral: Dict[str, object]) -> None:
    layer_dir = output_dir / "layer2"
    layer_dir.mkdir(parents=True, exist_ok=True)
    (layer_dir / "oracle_metrics.json").write_text(
        json.dumps(oracle, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (layer_dir / "behavioral_metrics.json").write_text(
        json.dumps(behavioral, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
