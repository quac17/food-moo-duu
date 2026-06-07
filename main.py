from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from src.core.pipeline import FoodSuggestionPipeline
from src.layer1_intent_context.rl_feedback import append_layer1_rl_feedback


def safe_text(text: str) -> str:
    encoding = sys.stdout.encoding or "utf-8"
    return text.encode(encoding, errors="replace").decode(encoding, errors="replace")


def print_recommendations(items: list[dict]) -> None:
    print("\nTop goi y mon an:")
    for idx, item in enumerate(items, start=1):
        print(f"  {idx}. {item['name']} ({item['id']}) - score={item['score']}")


def print_feedback_report(report: dict | object) -> None:
    print("\nImpact sau feedback:")
    print(f"  Mon: {report.chosen_dish_name} ({report.chosen_dish_id})")
    print(f"  Score truoc update: {report.score_before}")
    print(f"  Score sau update : {report.score_after}")
    print(f"  Delta            : {report.delta}")


def append_feedback_report_log(report: object, context_scores: dict[str, float], log_file: Path) -> None:
    top_context = sorted(
        context_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:5]
    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "dish_id": report.chosen_dish_id,
        "dish_name": report.chosen_dish_name,
        "score_before": report.score_before,
        "score_after": report.score_after,
        "delta": report.delta,
        "top_context": [
            {"tag": tag, "score": round(score, 4)}
            for tag, score in top_context
        ],
    }
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def get_feedback_log_file(pipeline: FoodSuggestionPipeline) -> Path:
    return pipeline.recommendation_engine.runtime_dir / "feedback_reports.jsonl"


def run_interactive(top_k: int) -> None:
    pipeline = FoodSuggestionPipeline()
    session_id = str(uuid4())
    turn_idx = 0
    print("=== FOOD MOO DUU - OFFLINE SMART RECOMMENDER ===")
    print("Kich ban demo: chat 2 luot -> chon mon -> he thong tu hoc.")

    user_turn_1 = input("\nUser chat #1: ").strip()
    turn_idx += 1
    turn1 = pipeline.process_turn(user_turn_1, top_k=top_k)
    print(f"Bot: {safe_text(turn1.response)}")
    print_recommendations(turn1.recommendations)

    user_turn_2 = input("\nUser chat #2 (co the quay xe): ").strip()
    turn_idx += 1
    turn2 = pipeline.process_turn(user_turn_2, top_k=top_k)
    print(f"Bot: {safe_text(turn2.response)}")
    print_recommendations(turn2.recommendations)

    choice = input(
        "\nNhap ID mon ban chon de xac nhan (de trong neu thoat app/khong chon): "
    ).strip()
    if choice:
        report = pipeline.apply_feedback(chosen_dish_id=choice, context_scores=turn2.context_scores)
        print("Da cap nhat Hebbian matrix va fitness response (success).")
        print_feedback_report(report)
        append_feedback_report_log(report, turn2.context_scores, get_feedback_log_file(pipeline))
        # RL feedback Layer1: chi log su kien chon mon, khong train realtime trong flow chatbot.
        append_layer1_rl_feedback(
            user_text=turn2.user_text,
            turn_index=turn_idx,
            raw_scores=turn2.raw_scores,
            context_scores=turn2.context_scores,
            chosen_dish_id=report.chosen_dish_id,
            chosen_dish_name=report.chosen_dish_name,
            recommended_candidates=turn2.recommendations,
            reward_signal=1.0,
            session_id=session_id,
            export_mode="context",
            use_state=True,
            source="chatbot_runtime",
        )
    else:
        pipeline.apply_abandon_feedback()
        print("Da cap nhat fitness response (failure do khong chon mon).")

    print("Hoan tat workflow offline.")


def run_non_interactive(chat1: str, chat2: str, choice: str, top_k: int) -> None:
    pipeline = FoodSuggestionPipeline()
    session_id = str(uuid4())
    turn_idx = 0
    print("=== FOOD MOO DUU - NON INTERACTIVE RUN ===")

    turn_idx += 1
    turn1 = pipeline.process_turn(chat1, top_k=top_k)
    print(f"Chat1: {chat1}")
    print(f"Bot1: {safe_text(turn1.response)}")
    print_recommendations(turn1.recommendations)

    turn_idx += 1
    turn2 = pipeline.process_turn(chat2, top_k=top_k)
    print(f"\nChat2: {chat2}")
    print(f"Bot2: {safe_text(turn2.response)}")
    print_recommendations(turn2.recommendations)

    if choice:
        report = pipeline.apply_feedback(chosen_dish_id=choice, context_scores=turn2.context_scores)
        print(f"\nFeedback: chosen={choice} -> update success")
        print_feedback_report(report)
        append_feedback_report_log(report, turn2.context_scores, get_feedback_log_file(pipeline))
        # RL feedback Layer1: chi ghi file de train offline sau do.
        append_layer1_rl_feedback(
            user_text=turn2.user_text,
            turn_index=turn_idx,
            raw_scores=turn2.raw_scores,
            context_scores=turn2.context_scores,
            chosen_dish_id=report.chosen_dish_id,
            chosen_dish_name=report.chosen_dish_name,
            recommended_candidates=turn2.recommendations,
            reward_signal=1.0,
            session_id=session_id,
            export_mode="context",
            use_state=True,
            source="chatbot_runtime",
        )
    else:
        pipeline.apply_abandon_feedback()
        print("\nFeedback: no-choice -> abandon update")

    print("Run hoan tat.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Food Moo Duu CLI")
    parser.add_argument("--non-interactive", action="store_true", help="Chay khong can input tu terminal")
    parser.add_argument("--chat1", default="Toi nay troi lanh, toi muon mon nuoc nong.")
    parser.add_argument("--chat2", default="Quay xe, gio toi muon mon nhanh va tien loi.")
    parser.add_argument("--choice", default="dish_001")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    if args.non_interactive:
        run_non_interactive(
            chat1=args.chat1,
            chat2=args.chat2,
            choice=args.choice,
            top_k=args.top_k,
        )
    else:
        run_interactive(top_k=args.top_k)


if __name__ == "__main__":
    main()
