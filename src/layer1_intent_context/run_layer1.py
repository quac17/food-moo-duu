from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from src.core.constants import LAYER1_DATA_DIR
from src.layer1_intent_context.dialog_state import DialogStateTracker
from src.layer1_intent_context.intent_tracker import IntentTracker


def _extract_top_tags(
    scores: Dict[str, float],
    top_k: int,
    threshold: float,
) -> List[Tuple[str, float]]:
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    filtered = [(tag, score) for tag, score in ranked if score >= threshold]
    return filtered[:top_k]


def _append_export_record(export_file: Path, record: Dict) -> None:
    export_file.parent.mkdir(parents=True, exist_ok=True)
    with export_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_one_turn(
    tracker: IntentTracker,
    dst: DialogStateTracker,
    text: str,
    top_k: int,
    threshold: float,
    export_file: Path | None,
    turn_index: int,
) -> None:
    # Flow Layer1-only:
    # 1) Predict intent tags tu text
    # 2) Update DST context
    # 3) Chi export va hien thi tags (khong goi Layer2/Layer3)
    prediction = tracker.predict_tags(text)
    context_scores = dst.update_context(prediction.tag_scores)
    top_tags = _extract_top_tags(context_scores, top_k=top_k, threshold=threshold)

    print(f"\n[Turn {turn_index}] User: {text}")
    print("Exported tags:")
    for tag, score in top_tags:
        print(f"- {tag}: {score:.4f}")

    if export_file is not None:
        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "turn": turn_index,
            "text": text,
            "tags": [{"tag": tag, "score": round(score, 4)} for tag, score in top_tags],
        }
        _append_export_record(export_file, record)
        print(f"Da export tags vao: {export_file}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Layer1 chat flow: chi trich xuat va export tags"
    )
    parser.add_argument("--chat", action="store_true", help="Bat che do chat nhieu luot")
    parser.add_argument(
        "--message",
        default="Sang nay troi mua, toi muon mon nong nhanh",
        help="Cau chat 1 luot khi khong dung --chat",
    )
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument(
        "--export-file",
        default=str(LAYER1_DATA_DIR / "tag_exports.jsonl"),
        help="File jsonl de luu ket qua export tag",
    )
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Khong ghi file export, chi in ra man hinh",
    )
    args = parser.parse_args()

    tracker = IntentTracker()
    dst = DialogStateTracker()
    export_file = None if args.no_export else Path(args.export_file)

    if args.chat:
        print("=== LAYER1 CHAT FLOW (EXPORT TAG ONLY) ===")
        print("Nhap 'exit' de ket thuc.")
        turn = 1
        while True:
            text = input("\nUser chat: ").strip()
            if not text or text.lower() in {"exit", "quit"}:
                print("Ket thuc chat flow Layer1.")
                break
            run_one_turn(
                tracker=tracker,
                dst=dst,
                text=text,
                top_k=args.top_k,
                threshold=args.threshold,
                export_file=export_file,
                turn_index=turn,
            )
            turn += 1
        return

    run_one_turn(
        tracker=tracker,
        dst=dst,
        text=args.message,
        top_k=args.top_k,
        threshold=args.threshold,
        export_file=export_file,
        turn_index=1,
    )


if __name__ == "__main__":
    main()
