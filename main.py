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


def resolve_choice_to_dish_id(choice: str, recommendations: list[dict]) -> str:
    normalized = choice.strip()
    if not normalized:
        return ""
    if normalized.isdigit():
        idx = int(normalized)
        if 1 <= idx <= len(recommendations):
            return str(recommendations[idx - 1].get("id", normalized))
    return normalized


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


def run_interactive(top_k: int, max_turns: int = 5) -> None:
    pipeline = FoodSuggestionPipeline()
    pipeline.reset_session_state()
    session_id = str(uuid4())
    print("=== FOOD MOO DUU - OFFLINE SMART RECOMMENDER ===")
    print(f"Kich ban demo: chat toi da {max_turns} luot -> chon mon -> he thong tu hoc.")
    print("Sau moi luot: nhap ID/so mon de chon (ket thuc), hoac de trong de chat tiep.")

    chosen = False
    for turn_idx in range(1, max_turns + 1):
        user_turn = input(f"\nUser chat #{turn_idx}/{max_turns}: ").strip()
        turn = pipeline.process_turn(user_turn, top_k=top_k)
        print(f"Bot: {safe_text(turn.response)}")
        print_recommendations(turn.recommendations)

        is_last_turn = turn_idx >= max_turns
        if is_last_turn:
            prompt = "\nLuot cuoi - nhap ID/so mon de chon (de trong neu khong chon): "
        else:
            prompt = "\nNhap ID/so mon de chon (de trong de chat tiep): "
        choice_input = input(prompt).strip()
        choice = resolve_choice_to_dish_id(choice_input, turn.recommendations)
        if choice:
            report = pipeline.apply_feedback(chosen_dish_id=choice, context_scores=turn.context_scores)
            print("Da cap nhat Hebbian matrix va fitness response (success).")
            print_feedback_report(report)
            append_feedback_report_log(report, turn.context_scores, get_feedback_log_file(pipeline))
            # RL feedback Layer1: chi log su kien chon mon, khong train realtime trong flow chatbot.
            append_layer1_rl_feedback(
                user_text=turn.user_text,
                turn_index=turn_idx,
                raw_scores=turn.raw_scores,
                context_scores=turn.context_scores,
                chosen_dish_id=report.chosen_dish_id,
                chosen_dish_name=report.chosen_dish_name,
                recommended_candidates=turn.recommendations,
                reward_signal=1.0,
                session_id=session_id,
                export_mode="context",
                use_state=True,
                source="chatbot_runtime",
            )
            chosen = True
            break

    if not chosen:
        pipeline.apply_abandon_feedback()
        print("Da cap nhat fitness response (failure do khong chon mon).")

    print("Hoan tat workflow offline.")


def run_non_interactive(chats: list[str], choice: str, top_k: int, max_turns: int = 5) -> None:
    pipeline = FoodSuggestionPipeline()
    pipeline.reset_session_state()
    session_id = str(uuid4())
    print("=== FOOD MOO DUU - NON INTERACTIVE RUN ===")

    effective_chats = [chat for chat in chats if chat is not None][:max_turns]
    if not effective_chats:
        print("Khong co cau chat dau vao.")
        return

    last_turn = None
    last_turn_idx = 0
    for turn_idx, chat in enumerate(effective_chats, start=1):
        turn = pipeline.process_turn(chat, top_k=top_k)
        print(f"\nChat{turn_idx}: {chat}")
        print(f"Bot{turn_idx}: {safe_text(turn.response)}")
        print_recommendations(turn.recommendations)
        last_turn = turn
        last_turn_idx = turn_idx

    if choice and last_turn is not None:
        chosen_dish_id = resolve_choice_to_dish_id(choice, last_turn.recommendations)
        report = pipeline.apply_feedback(chosen_dish_id=chosen_dish_id, context_scores=last_turn.context_scores)
        print(f"\nFeedback: chosen={choice} -> update success")
        print_feedback_report(report)
        append_feedback_report_log(report, last_turn.context_scores, get_feedback_log_file(pipeline))
        # RL feedback Layer1: chi ghi file de train offline sau do.
        append_layer1_rl_feedback(
            user_text=last_turn.user_text,
            turn_index=last_turn_idx,
            raw_scores=last_turn.raw_scores,
            context_scores=last_turn.context_scores,
            chosen_dish_id=report.chosen_dish_id,
            chosen_dish_name=report.chosen_dish_name,
            recommended_candidates=last_turn.recommendations,
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


DEFAULT_CHATS = [
    "Toi nay troi lanh, toi muon mon nuoc nong.",
    "Quay xe, gio toi muon mon nhanh va tien loi.",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Food Moo Duu CLI")
    parser.add_argument("--non-interactive", action="store_true", help="Chay khong can input tu terminal")
    parser.add_argument(
        "--chat",
        action="append",
        default=None,
        help="Cau chat (lap lai de them nhieu luot, toi da --max-turns). Dung cho --non-interactive.",
    )
    parser.add_argument("--choice", default="dish_001")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-turns", type=int, default=5, help="So luot chat toi da (mac dinh 5)")
    args = parser.parse_args()

    max_turns = max(1, args.max_turns)

    if args.non_interactive:
        chats = args.chat if args.chat else list(DEFAULT_CHATS)
        run_non_interactive(
            chats=chats,
            choice=args.choice,
            top_k=args.top_k,
            max_turns=max_turns,
        )
    else:
        run_interactive(top_k=args.top_k, max_turns=max_turns)


if __name__ == "__main__":
    main()
