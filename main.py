from __future__ import annotations

from src.core.pipeline import FoodSuggestionPipeline


def print_recommendations(items: list[dict]) -> None:
    print("\nTop goi y mon an:")
    for idx, item in enumerate(items, start=1):
        print(f"  {idx}. {item['name']} ({item['id']}) - score={item['score']}")


def main() -> None:
    pipeline = FoodSuggestionPipeline()
    print("=== FOOD MOO DUU - OFFLINE SMART RECOMMENDER ===")
    print("Kich ban demo: chat 2 luot -> chon mon -> he thong tu hoc.")

    user_turn_1 = input("\nUser chat #1: ").strip()
    turn1 = pipeline.process_turn(user_turn_1, top_k=5)
    print(f"Bot: {turn1.response}")
    print_recommendations(turn1.recommendations)

    user_turn_2 = input("\nUser chat #2 (co the quay xe): ").strip()
    turn2 = pipeline.process_turn(user_turn_2, top_k=5)
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


if __name__ == "__main__":
    main()
