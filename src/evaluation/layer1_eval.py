from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Dict, List

from src.core.constants import LAYER1_DATA_DIR
from src.layer1_intent_context.intent_tracker import IntentTracker


def bootstrap_ablation_artifacts() -> None:
    """Sao chep artifact mac dinh sang no_rl/with_rl neu chua co (fallback khi train loi)."""
    artifact_dir = LAYER1_DATA_DIR / "model_artifacts_dl"
    defaults = {
        "intent_model.pt": "intent_model_{suffix}.pt",
        "intent_model_meta.json": "intent_model_meta_{suffix}.json",
        "vocab.json": "vocab_{suffix}.json",
    }
    for suffix in ("no_rl", "with_rl"):
        for src_name, pattern in defaults.items():
            src = artifact_dir / src_name
            dst = artifact_dir / pattern.format(suffix=suffix)
            if dst.exists() or not src.exists():
                continue
            shutil.copy2(src, dst)


def _write_per_tag_csv(rows: List[Dict[str, object]], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["tag", "precision", "recall", "f1", "support"])
        writer.writeheader()
        writer.writerows(rows)


def evaluate_saved_layer1(
    use_all_datasets: bool,
    artifact_suffix: str,
    include_rl_samples: bool,
    threshold: float | None = None,
) -> Dict[str, object]:
    tracker = IntentTracker(use_all_datasets=use_all_datasets, include_rl_samples=include_rl_samples)
    if threshold is not None:
        tracker.dl_config.decision_threshold = threshold
    tracker.load_artifacts(artifact_suffix)
    metrics = tracker.evaluate_validation_split()
    return {
        "variant": artifact_suffix,
        "include_rl_samples": include_rl_samples,
        "use_all_datasets": use_all_datasets,
        "metrics": metrics,
        "artifact_suffix": artifact_suffix,
    }


def train_and_evaluate_layer1(
    use_all_datasets: bool,
    include_rl_samples: bool,
    artifact_suffix: str,
    threshold: float | None = None,
) -> Dict[str, object]:
    tracker = IntentTracker(use_all_datasets=use_all_datasets, include_rl_samples=include_rl_samples)
    if threshold is not None:
        tracker.dl_config.decision_threshold = threshold
    metrics = tracker.fit(artifact_suffix=artifact_suffix)
    return {
        "variant": artifact_suffix or "default",
        "include_rl_samples": include_rl_samples,
        "use_all_datasets": use_all_datasets,
        "metrics": metrics,
        "artifact_suffix": artifact_suffix,
    }


def compare_ablation(without_rl: Dict[str, object], with_rl: Dict[str, object]) -> Dict[str, object]:
    w_summary = without_rl["metrics"]["summary"]  # type: ignore[index]
    r_summary = with_rl["metrics"]["summary"]  # type: ignore[index]
    return {
        "without_rl": w_summary,
        "with_rl": r_summary,
        "delta": {
            "micro_f1": round(r_summary["micro_f1"] - w_summary["micro_f1"], 4),
            "macro_f1": round(r_summary["macro_f1"] - w_summary["macro_f1"], 4),
            "micro_precision": round(r_summary["micro_precision"] - w_summary["micro_precision"], 4),
            "micro_recall": round(r_summary["micro_recall"] - w_summary["micro_recall"], 4),
            "subset_accuracy": round(r_summary["subset_accuracy"] - w_summary["subset_accuracy"], 4),
        },
    }


def save_layer1_outputs(
    output_dir: Path,
    without_rl: Dict[str, object],
    with_rl: Dict[str, object],
    ablation: Dict[str, object],
) -> None:
    layer_dir = output_dir / "layer1"
    layer_dir.mkdir(parents=True, exist_ok=True)

    without_metrics = without_rl["metrics"]  # type: ignore[index]
    with_metrics = with_rl["metrics"]  # type: ignore[index]

    (layer_dir / "without_rl_metrics.json").write_text(
        json.dumps(without_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (layer_dir / "with_rl_metrics.json").write_text(
        json.dumps(with_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (layer_dir / "ablation_compare.json").write_text(
        json.dumps(ablation, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_per_tag_csv(without_metrics["per_tag"], layer_dir / "without_rl_per_tag.csv")  # type: ignore[index]
    _write_per_tag_csv(with_metrics["per_tag"], layer_dir / "with_rl_per_tag.csv")  # type: ignore[index]
