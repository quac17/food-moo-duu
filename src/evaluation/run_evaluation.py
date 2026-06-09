from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict

from src.core.constants import HYPERPARAMS, LAYER1_DATA_DIR
from src.evaluation.layer1_eval import (
    bootstrap_ablation_artifacts,
    compare_ablation,
    evaluate_saved_layer1,
    save_layer1_outputs,
    train_and_evaluate_layer1,
)
from src.evaluation.layer2_eval import (
    evaluate_layer2_behavioral,
    evaluate_layer2_oracle,
    save_layer2_outputs,
)
from src.evaluation.layer3_eval import evaluate_layer3_fitness, save_layer3_outputs
from src.evaluation.pipeline_eval import (
    evaluate_end_to_end,
    evaluate_feedback_delta,
    save_pipeline_outputs,
)


def _train_layer1_subprocess(
    use_all_datasets: bool,
    include_rl_samples: bool,
    artifact_suffix: str,
    threshold: float | None,
) -> None:
    threshold_expr = "None" if threshold is None else str(threshold)
    code = f"""
from src.evaluation.layer1_eval import train_and_evaluate_layer1
train_and_evaluate_layer1(
    use_all_datasets={use_all_datasets},
    include_rl_samples={include_rl_samples},
    artifact_suffix={artifact_suffix!r},
    threshold={threshold_expr},
)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(Path(__file__).resolve().parents[2]),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"Train Layer1 '{artifact_suffix}' that bai")


def _git_commit_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except OSError:
        pass
    return ""


def _build_summary(
    ablation: Dict[str, object],
    oracle: Dict[str, object],
    behavioral: Dict[str, object],
    layer3: Dict[str, object],
    pipeline: Dict[str, object],
) -> Dict[str, object]:
    without_summary = ablation["without_rl"]  # type: ignore[index]
    with_summary = ablation["with_rl"]  # type: ignore[index]
    delta = ablation["delta"]  # type: ignore[index]
    return {
        "layer1_without_rl_macro_f1": without_summary["macro_f1"],
        "layer1_with_rl_macro_f1": with_summary["macro_f1"],
        "layer1_delta_macro_f1": delta["macro_f1"],
        "layer1_delta_micro_f1": delta["micro_f1"],
        "layer2_oracle_hit_at_5": oracle["metrics"]["hit_at_5"],  # type: ignore[index]
        "layer2_oracle_mrr": oracle["metrics"]["mrr"],  # type: ignore[index]
        "layer2_behavioral_hit_at_5": behavioral["metrics"].get("hit_at_5", 0.0),  # type: ignore[index]
        "layer2_behavioral_mrr": behavioral["metrics"]["mrr"],  # type: ignore[index]
        "layer3_success_rate": layer3.get("success_rate", 0.0),
        "layer3_avg_fitness": layer3.get("avg_fitness", 0.0),
        "pipeline_e2e_without_rl_hit_at_5": pipeline["without_rl"]["metrics"].get("hit_at_5", 0.0),  # type: ignore[index]
        "pipeline_e2e_with_rl_hit_at_5": pipeline["with_rl"]["metrics"].get("hit_at_5", 0.0),  # type: ignore[index]
        "pipeline_feedback_delta_mean": pipeline["feedback_delta"]["feedback_delta_mean"],  # type: ignore[index]
    }


def _write_summary_md(summary: Dict[str, object], output_file: Path) -> None:
    lines = [
        "# Bao cao danh gia Food Moo Duu",
        "",
        "## Layer 1 (Intent)",
        f"- Macro F1 (khong RL): **{summary['layer1_without_rl_macro_f1']}**",
        f"- Macro F1 (co RL): **{summary['layer1_with_rl_macro_f1']}**",
        f"- Delta macro F1: **{summary['layer1_delta_macro_f1']}**",
        "",
        "## Layer 2 (Recommendation)",
        f"- Oracle Hit@5: **{summary['layer2_oracle_hit_at_5']}** | MRR: **{summary['layer2_oracle_mrr']}**",
        f"- Behavioral Hit@5: **{summary['layer2_behavioral_hit_at_5']}** | MRR: **{summary['layer2_behavioral_mrr']}**",
        "",
        "## Layer 3 (Genetic Response)",
        f"- Success rate: **{summary['layer3_success_rate']}**",
        f"- Avg fitness: **{summary['layer3_avg_fitness']}**",
        "",
        "## Pipeline end-to-end",
        f"- E2E Hit@5 (khong RL): **{summary['pipeline_e2e_without_rl_hit_at_5']}**",
        f"- E2E Hit@5 (co RL): **{summary['pipeline_e2e_with_rl_hit_at_5']}**",
        f"- Feedback delta mean: **{summary['pipeline_feedback_delta_mean']}**",
    ]
    output_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_evaluation(
    use_all_datasets: bool,
    top_k: int,
    include_simulated: bool,
    threshold: float | None,
    train_layer1: bool,
    output_root: Path,
) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "timestamp": timestamp,
        "use_all_datasets": use_all_datasets,
        "top_k": top_k,
        "include_simulated": include_simulated,
        "threshold": threshold,
        "train_layer1": train_layer1,
        "git_commit": _git_commit_hash(),
        "active_dataset": json.loads((LAYER1_DATA_DIR / "datasets.json").read_text(encoding="utf-8")).get(
            "active_dataset", "dataset_v001"
        ),
        "dst_hyperparameters": {
            "context_decay": HYPERPARAMS["context_decay"],
            "context_accumulation_alpha": HYPERPARAMS["context_accumulation_alpha"],
            "context_conflict_beta": HYPERPARAMS["context_conflict_beta"],
        },
    }
    train_fallback = False
    if train_layer1:
        try:
            _train_layer1_subprocess(use_all_datasets, False, "no_rl", threshold)
            rl_cmd = [sys.executable, "-m", "src.layer1_intent_context.train_reinforcement"]
            if include_simulated:
                rl_cmd.append("--include-simulated")
            subprocess.run(rl_cmd, check=False)
            _train_layer1_subprocess(use_all_datasets, True, "with_rl", threshold)
            without_rl = evaluate_saved_layer1(
                use_all_datasets=use_all_datasets,
                include_rl_samples=False,
                artifact_suffix="no_rl",
                threshold=threshold,
            )
            with_rl = evaluate_saved_layer1(
                use_all_datasets=use_all_datasets,
                include_rl_samples=True,
                artifact_suffix="with_rl",
                threshold=threshold,
            )
        except Exception as exc:
            train_fallback = True
            manifest["train_fallback"] = True
            manifest["train_error"] = str(exc)
            bootstrap_ablation_artifacts()
            without_rl = evaluate_saved_layer1(
                use_all_datasets=use_all_datasets,
                include_rl_samples=False,
                artifact_suffix="no_rl",
                threshold=threshold,
            )
            with_rl = evaluate_saved_layer1(
                use_all_datasets=use_all_datasets,
                include_rl_samples=True,
                artifact_suffix="with_rl",
                threshold=threshold,
            )
    else:
        bootstrap_ablation_artifacts()
        without_rl = evaluate_saved_layer1(
            use_all_datasets=use_all_datasets,
            include_rl_samples=False,
            artifact_suffix="no_rl",
            threshold=threshold,
        )
        with_rl = evaluate_saved_layer1(
            use_all_datasets=use_all_datasets,
            include_rl_samples=True,
            artifact_suffix="with_rl",
            threshold=threshold,
        )

    ablation = compare_ablation(without_rl, with_rl)
    save_layer1_outputs(run_dir, without_rl, with_rl, ablation)

    oracle = evaluate_layer2_oracle(use_all_datasets=use_all_datasets, k_values=[3, top_k])
    behavioral = evaluate_layer2_behavioral(include_simulated=include_simulated, top_k=top_k)
    save_layer2_outputs(run_dir, oracle, behavioral)

    layer3 = evaluate_layer3_fitness()
    save_layer3_outputs(run_dir, layer3)

    from src.layer1_intent_context.intent_tracker import IntentTracker

    tracker_without = IntentTracker(use_all_datasets=use_all_datasets, include_rl_samples=False)
    tracker_without.load_artifacts("no_rl")
    tracker_with = IntentTracker(use_all_datasets=use_all_datasets, include_rl_samples=True)
    tracker_with.load_artifacts("with_rl")

    e2e_without = evaluate_end_to_end(
        tracker_without,
        include_simulated=include_simulated,
        top_k=top_k,
        tag_threshold=threshold or tracker_without.dl_config.decision_threshold,
    )
    e2e_with = evaluate_end_to_end(
        tracker_with,
        include_simulated=include_simulated,
        top_k=top_k,
        tag_threshold=threshold or tracker_with.dl_config.decision_threshold,
    )
    feedback_delta = evaluate_feedback_delta()
    save_pipeline_outputs(run_dir, e2e_without, e2e_with, feedback_delta)

    (run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = _build_summary(ablation, oracle, behavioral, layer3, {
        "without_rl": e2e_without,
        "with_rl": e2e_with,
        "feedback_delta": feedback_delta,
    })
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_summary_md(summary, run_dir / "summary.md")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Chay danh gia toan bo he thong Food Moo Duu")
    parser.add_argument("--all-datasets", action="store_true")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--include-simulated", action="store_true")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--skip-train", action="store_true", help="Chi danh gia, khong train lai Layer1")
    parser.add_argument("--output-dir", default="data/evaluation/runs")
    args = parser.parse_args()

    run_dir = run_evaluation(
        use_all_datasets=args.all_datasets,
        top_k=args.top_k,
        include_simulated=args.include_simulated,
        threshold=args.threshold,
        train_layer1=not args.skip_train,
        output_root=Path(args.output_dir),
    )
    print(f"Hoan tat danh gia: {run_dir}")


if __name__ == "__main__":
    main()
