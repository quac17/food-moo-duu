"""Doc ket qua danh gia moi nhat de cap nhat slide/diagram."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "data" / "evaluation" / "runs"


def latest_run_dir() -> Path | None:
    if not RUNS_DIR.exists():
        return None
    candidates = sorted(
        [p for p in RUNS_DIR.iterdir() if p.is_dir() and p.name.startswith("run_")],
        key=lambda p: p.name,
        reverse=True,
    )
    return candidates[0] if candidates else None


def load_eval_bundle() -> Tuple[str, Dict[str, object], Dict[str, object]]:
    run_dir = latest_run_dir()
    if run_dir is None:
        raise FileNotFoundError("Khong tim thay thu muc evaluation run nao.")

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))

    layer1_no_rl = json.loads((run_dir / "layer1" / "without_rl_metrics.json").read_text(encoding="utf-8"))
    layer1_with_rl = json.loads((run_dir / "layer1" / "with_rl_metrics.json").read_text(encoding="utf-8"))
    layer2_oracle = json.loads((run_dir / "layer2" / "oracle_metrics.json").read_text(encoding="utf-8"))
    layer2_behavioral = json.loads((run_dir / "layer2" / "behavioral_metrics.json").read_text(encoding="utf-8"))
    layer3 = json.loads((run_dir / "layer3" / "fitness_metrics.json").read_text(encoding="utf-8"))

    run_id = run_dir.name.replace("run_", "")
    bundle = {
        "summary": summary,
        "manifest": manifest,
        "layer1_no_rl": layer1_no_rl,
        "layer1_with_rl": layer1_with_rl,
        "layer2_oracle": layer2_oracle,
        "layer2_behavioral": layer2_behavioral,
        "layer3": layer3,
    }
    return run_id, bundle, {"run_dir": str(run_dir)}
