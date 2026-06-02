from __future__ import annotations

import sys

from src.layer2_adaptive_recommendation.recommendation_engine import RecommendationEngine


def reset_layer2_runtime() -> int:
    engine = RecommendationEngine()
    runtime_file = engine.runtime_file
    feedback_log_file = engine.runtime_dir / "feedback_reports.jsonl"

    if runtime_file.exists():
        runtime_file.unlink()
        print(f"Da xoa runtime file: {runtime_file}")
    else:
        print(f"Khong co runtime file de xoa: {runtime_file}")

    if feedback_log_file.exists():
        feedback_log_file.unlink()
        print(f"Da xoa feedback log file: {feedback_log_file}")
    else:
        print(f"Khong co feedback log file de xoa: {feedback_log_file}")

    return 0


def main() -> None:
    sys.exit(reset_layer2_runtime())


if __name__ == "__main__":
    main()
