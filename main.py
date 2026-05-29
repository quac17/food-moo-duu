from __future__ import annotations

import argparse

from src.core.pipeline import FoodSuggestionPipeline


def print_recommendations(items: list[dict]) -> None:
    print("\nTop goi y mon an:")
    for idx, item in enumerate(items, start=1):
        print(f"  {idx}. {item['name']} ({item['id']}) - score={item['score']}")


def run_interactive(top_k: int) -> None:
    pipeline = FoodSuggestionPipeline()
    print("=== FOOD MOO DUU - OFFLINE SMART RECOMMENDER ===")
    print("Kich ban demo: chat 2 luot -> chon mon -> he thong tu hoc.")

    user_turn_1 = input("\nUser chat #1: ").strip()
    turn1 = pipeline.process_turn(user_turn_1, top_k=top_k)
    print(f"Bot: {turn1.response}")
    print_recommendations(turn1.recommendations)

    user_turn_2 = input("\nUser chat #2 (co the quay xe): ").strip()
    turn2 = pipeline.process_turn(user_turn_2, top_k=top_k)
    print(f"Bot: {turn2.response}")
    print_recommendations(turn2.recommendations)

    choice = input(
        "\nNhap ID mon ban chon de xac nhan (de trong neu thoat app/khong chon): "
    ).strip()
    if choice:
        pipeline.apply_feedback(chosen_dish_id=choice, context_scores=turn2.context_scores)
        print("Da cap nhat Hebbian matrix va fitness response (success).")
    else:
        pipeline.apply_abandon_feedback()
        print("Da cap nhat fitness response (failure do khong chon mon).")

    print("Hoan tat workflow offline.")


def run_non_interactive(chat1: str, chat2: str, choice: str, top_k: int) -> None:
    pipeline = FoodSuggestionPipeline()
    print("=== FOOD MOO DUU - NON INTERACTIVE RUN ===")

    turn1 = pipeline.process_turn(chat1, top_k=top_k)
    print(f"Chat1: {chat1}")
    print(f"Bot1: {turn1.response}")
    print_recommendations(turn1.recommendations)

    turn2 = pipeline.process_turn(chat2, top_k=top_k)
    print(f"\nChat2: {chat2}")
    print(f"Bot2: {turn2.response}")
    print_recommendations(turn2.recommendations)

    if choice:
        pipeline.apply_feedback(chosen_dish_id=choice, context_scores=turn2.context_scores)
        print(f"\nFeedback: chosen={choice} -> update success")
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
