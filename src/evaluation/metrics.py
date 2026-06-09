"""Ham metric dung chung cho danh gia recommendation va multi-label."""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
)


def hit_at_k(ranked_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    top = ranked_ids[:k]
    return 1.0 if any(item in relevant for item in top) else 0.0


def mrr(ranked_ids: Sequence[str], relevant_ids: Iterable[str]) -> float:
    relevant = set(relevant_ids)
    for idx, item_id in enumerate(ranked_ids, start=1):
        if item_id in relevant:
            return 1.0 / idx
    return 0.0


def precision_at_k(ranked_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    relevant = set(relevant_ids)
    if k <= 0:
        return 0.0
    top = ranked_ids[:k]
    if not top:
        return 0.0
    hits = sum(1 for item in top if item in relevant)
    return hits / len(top)


def recall_at_k(ranked_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    top = ranked_ids[:k]
    hits = sum(1 for item in top if item in relevant)
    return hits / len(relevant)


def ndcg_at_k(ranked_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    relevant = set(relevant_ids)
    if not relevant or k <= 0:
        return 0.0

    def dcg(items: Sequence[str]) -> float:
        score = 0.0
        for idx, item in enumerate(items[:k], start=1):
            if item in relevant:
                score += 1.0 / math.log2(idx + 1)
        return score

    actual = dcg(ranked_ids)
    ideal = dcg(list(relevant) + [item for item in ranked_ids if item not in relevant])
    if ideal <= 0.0:
        return 0.0
    return actual / ideal


def mean_rank(ranked_ids: Sequence[str], relevant_ids: Iterable[str]) -> float:
    relevant = set(relevant_ids)
    for idx, item_id in enumerate(ranked_ids, start=1):
        if item_id in relevant:
            return float(idx)
    return float(len(ranked_ids) + 1)


def aggregate_ranking_metrics(
    ranked_lists: List[List[str]],
    relevant_lists: List[List[str]],
    k_values: Sequence[int] = (3, 5),
) -> Dict[str, float]:
    if not ranked_lists:
        return {}

    metrics: Dict[str, float] = {"mrr": 0.0, "mean_rank": 0.0}
    for ranked, relevant in zip(ranked_lists, relevant_lists, strict=False):
        metrics["mrr"] += mrr(ranked, relevant)
        metrics["mean_rank"] += mean_rank(ranked, relevant)

    n = len(ranked_lists)
    metrics["mrr"] = round(metrics["mrr"] / n, 4)
    metrics["mean_rank"] = round(metrics["mean_rank"] / n, 4)

    for k in k_values:
        hit_sum = prec_sum = rec_sum = ndcg_sum = 0.0
        for ranked, relevant in zip(ranked_lists, relevant_lists, strict=False):
            hit_sum += hit_at_k(ranked, relevant, k)
            prec_sum += precision_at_k(ranked, relevant, k)
            rec_sum += recall_at_k(ranked, relevant, k)
            ndcg_sum += ndcg_at_k(ranked, relevant, k)
        metrics[f"hit_at_{k}"] = round(hit_sum / n, 4)
        metrics[f"precision_at_{k}"] = round(prec_sum / n, 4)
        metrics[f"recall_at_{k}"] = round(rec_sum / n, 4)
        metrics[f"ndcg_at_{k}"] = round(ndcg_sum / n, 4)
    metrics["samples"] = float(n)
    return metrics


def multilabel_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    if y_true.size == 0:
        return {
            "micro_f1": 0.0,
            "macro_f1": 0.0,
            "micro_precision": 0.0,
            "micro_recall": 0.0,
            "hamming_loss": 0.0,
            "subset_accuracy": 0.0,
        }
    return {
        "micro_f1": round(float(f1_score(y_true, y_pred, average="micro", zero_division=0)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "micro_precision": round(float(precision_score(y_true, y_pred, average="micro", zero_division=0)), 4),
        "micro_recall": round(float(recall_score(y_true, y_pred, average="micro", zero_division=0)), 4),
        "hamming_loss": round(float(hamming_loss(y_true, y_pred)), 4),
        "subset_accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
    }


def per_tag_metrics(y_true: np.ndarray, y_pred: np.ndarray, tags: List[str]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for idx, tag in enumerate(tags):
        true_col = y_true[:, idx]
        pred_col = y_pred[:, idx]
        support = int(true_col.sum())
        if support == 0 and pred_col.sum() == 0:
            precision = recall = f1 = 0.0
        else:
            precision = float(precision_score(true_col, pred_col, zero_division=0))
            recall = float(recall_score(true_col, pred_col, zero_division=0))
            f1 = float(f1_score(true_col, pred_col, zero_division=0))
        rows.append(
            {
                "tag": tag,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "support": support,
            }
        )
    return rows


def tag_overlap_ratio(predicted_tags: Iterable[str], reference_tags: Iterable[str]) -> float:
    pred = set(predicted_tags)
    ref = set(reference_tags)
    if not ref:
        return 0.0
    return round(len(pred & ref) / len(ref), 4)
