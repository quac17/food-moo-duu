from src.layer1_intent_context.dialog_state import DialogStateTracker
from src.layer1_intent_context.intent_tracker import IntentTracker


def main() -> None:
    tracker = IntentTracker()
    prediction = tracker.predict_tags("Sang nay troi mua, toi muon mon nong nhanh")
    dst = DialogStateTracker()
    context = dst.update_context(prediction.tag_scores)
    top = sorted(context.items(), key=lambda x: x[1], reverse=True)[:5]
    print("Layer1 ok. Top context tags:", top)


if __name__ == "__main__":
    main()
