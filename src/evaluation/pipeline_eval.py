from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from src.core.constants import LAYER2_DATA_DIR
from src.evaluation.metrics import aggregate_ranking_metrics, tag_overlap_ratio
from src.layer1_intent_context.intent_tracker import IntentTracker
from src.layer1_intent_context.rl_feedback import (
    REAL_EVENTS_FILE,
    SIMULATED_EVENTS_FILE,
    load_feedback_events,
)
from src.layer2_adaptive_recommendation.recommendation_engine import RecommendationEngine


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


def _reference_tags_from_event(event: Dict[str, object], threshold: float = 0.18) -> List[str]:
    context_tags = event.get("context_tags", [])
    if not isinstance(context_tags, list):
        return []
    return [
        str(item.get("tag", "")).strip()
        for item in context_tags
        if isinstance(item, dict) and float(item.get("score", 0.0)) >= threshold
    ]


def _predicted_tags(tracker: IntentTracker, text: str, threshold: float) -> List[str]:
    prediction = tracker.predict_tags(text)
    return [tag for tag, score in prediction.tag_scores.items() if score >= threshold]


def evaluate_end_to_end(
    tracker: IntentTracker,
    include_simulated: bool,
    top_k: int = 5,
    tag_threshold: float = 0.3,
) -> Dict[str, object]:
    files = [REAL_EVENTS_FILE]
    if include_simulated:
        files.append(SIMULATED_EVENTS_FILE)
    events = load_feedback_events(files)
    engine = RecommendationEngine(prefer_canonical=True)

    ranked_lists: List[List[str]] = []
    relevant_lists: List[List[str]] = []
    overlap_scores: List[float] = []
    chosen_scores: List[float] = []

    for event in events:
        chosen_id = str(event.get("chosen_dish_id", "")).strip()
        user_text = str(event.get("user_text", "")).strip()
        if not chosen_id or not user_text:
            continue

        predicted = _predicted_tags(tracker, user_text, threshold=tag_threshold)
        reference = _reference_tags_from_event(event, threshold=0.18)
        overlap_scores.append(tag_overlap_ratio(predicted, reference))

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
    if overlap_scores:
        metrics["tag_overlap_ratio_mean"] = round(sum(overlap_scores) / len(overlap_scores), 4)
    if chosen_scores:
        metrics["avg_chosen_score"] = round(sum(chosen_scores) / len(chosen_scores), 4)

    return {
        "variant": tracker._artifact_suffix or "default",
        "samples": len(ranked_lists),
        "tag_threshold": tag_threshold,
        "metrics": metrics,
    }


def evaluate_feedback_delta() -> Dict[str, object]:
    feedback_file = LAYER2_DATA_DIR / "runtime" / "feedback_reports.jsonl"
    if not feedback_file.exists():
        return {"samples": 0, "feedback_delta_mean": 0.0}

    deltas: List[float] = []
    with feedback_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            before = float(payload.get("score_before", 0.0))
            after = float(payload.get("score_after", 0.0))
            deltas.append(after - before)
    return {
        "samples": len(deltas),
        "feedback_delta_mean": round(sum(deltas) / len(deltas), 4) if deltas else 0.0,
    }


def save_pipeline_outputs(
    output_dir: Path,
    without_rl: Dict[str, object],
    with_rl: Dict[str, object],
    feedback_delta: Dict[str, object],
) -> None:
    pipeline_dir = output_dir / "pipeline"
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "without_rl": without_rl,
        "with_rl": with_rl,
        "feedback_delta": feedback_delta,
    }
    (pipeline_dir / "end_to_end_metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
