from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from src.core.constants import HYPERPARAMS, LAYER1_DATA_DIR
from src.layer1_intent_context.dialog_state import DialogStateTracker
from src.layer1_intent_context.intent_tracker import IntentTracker


def _sanitize_text(value: str) -> str:
    # Loai bo surrogate chars de tranh UnicodeEncodeError khi dump json.
    return value.encode("utf-8", errors="replace").decode("utf-8", errors="replace")


def _sanitize_record(value):
    if isinstance(value, str):
        return _sanitize_text(value)
    if isinstance(value, list):
        return [_sanitize_record(item) for item in value]
    if isinstance(value, dict):
        return {key: _sanitize_record(item) for key, item in value.items()}
    return value


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
    safe_record = _sanitize_record(record)
    with export_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(safe_record, ensure_ascii=False) + "\n")


def run_one_turn(
    tracker: IntentTracker,
    dst: DialogStateTracker,
    text: str,
    top_k: int,
    raw_threshold: float,
    context_threshold: float,
    export_file: Path | None,
    turn_index: int,
    use_state: bool,
    export_mode: str,
) -> None:
    # Flow Layer1-only:
    # 1) Predict intent tags tu text
    # 2) Update DST context
    # 3) Chi export va hien thi tags (khong goi Layer2/Layer3)
    prediction = tracker.predict_tags(text)
    raw_top_tags = _extract_top_tags(
        prediction.tag_scores,
        top_k=top_k,
        threshold=raw_threshold,
    )

    # Neu khong dung state, context se la raw de test cau don cho minh bach.
    if use_state:
        context_scores = dst.update_context(prediction.tag_scores)
    else:
        context_scores = dict(prediction.tag_scores)
    context_top_tags = _extract_top_tags(
        context_scores,
        top_k=top_k,
        threshold=context_threshold,
    )

    selected_top_tags = context_top_tags if export_mode == "context" else raw_top_tags

    print(f"\n[Turn {turn_index}] User: {text}")
    print("Raw intent tags (tu cau chat hien tai):")
    for tag, score in raw_top_tags:
        print(f"- {tag}: {score:.4f}")
    print("Context tags (sau DST):")
    for tag, score in context_top_tags:
        print(f"- {tag}: {score:.4f}")

    if export_file is not None:
        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "turn": turn_index,
            "text": text,
            "export_mode": export_mode,
            "use_state": use_state,
            "raw_tags": [{"tag": tag, "score": round(score, 4)} for tag, score in raw_top_tags],
            "context_tags": [{"tag": tag, "score": round(score, 4)} for tag, score in context_top_tags],
            "tags": [{"tag": tag, "score": round(score, 4)} for tag, score in selected_top_tags],
        }
        _append_export_record(export_file, record)
        print(f"Da export tags ({export_mode}) vao: {export_file}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Layer1 chat flow (DL vi-SBERT): chi trich xuat va export tags"
    )
    parser.add_argument("--chat", action="store_true", help="Bat che do chat nhieu luot")
    parser.add_argument(
        "--all-datasets",
        action="store_true",
        help="Train/predict bang toan bo available_datasets thay vi active_dataset",
    )
    parser.add_argument(
        "--message",
        default="Sang nay troi mua, toi muon mon nong nhanh",
        help="Cau chat 1 luot khi khong dung --chat",
    )
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Nguong chung cho ca raw/context (giu tuong thich lenh cu)",
    )
    parser.add_argument("--raw-threshold", type=float, default=0.2)
    parser.add_argument("--context-threshold", type=float, default=0.1)
    parser.add_argument(
        "--no-state",
        action="store_true",
        help="Khong update DST state, chi predict raw intent cho tung cau",
    )
    parser.add_argument(
        "--reset-state",
        action="store_true",
        help="Reset session_state truoc khi chay de tranh anh huong context cu",
    )
    parser.add_argument(
        "--export-mode",
        choices=["context", "raw"],
        default="context",
        help="Chon nhom tag de dua vao truong tags trong file export",
    )
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
    parser.add_argument(
        "--warmup-train",
        action="store_true",
        help="Train/warmup model Layer1 truoc khi vao chat flow",
    )
    args = parser.parse_args()

    tracker = IntentTracker(use_all_datasets=args.all_datasets)
    if args.warmup_train:
        tracker.fit()
    dst = DialogStateTracker(
        decay_rate=HYPERPARAMS["context_decay"],
        time_decay_rate=HYPERPARAMS["context_decay_time"],
        accumulation_alpha=HYPERPARAMS["context_accumulation_alpha"],
        conflict_beta=HYPERPARAMS["context_conflict_beta"],
    )
    export_file = None if args.no_export else Path(args.export_file)

    # Neu co threshold chung thi uu tien de tranh vo lenh cu.
    if args.threshold is not None:
        raw_threshold = args.threshold
        context_threshold = args.threshold
    else:
        raw_threshold = args.raw_threshold
        context_threshold = args.context_threshold

    if args.reset_state:
        # Reset state de testcase moi khong bi dom boi context cu.
        dst.state.tag_scores = {tag: 0.0 for tag in dst.available_tags}
        dst.state.turn_index = 0
        dst.save_state()
        print(f"Da reset session state tai: {dst.state_file}")

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
                raw_threshold=raw_threshold,
                context_threshold=context_threshold,
                export_file=export_file,
                turn_index=turn,
                use_state=not args.no_state,
                export_mode=args.export_mode,
            )
            turn += 1
        return

    run_one_turn(
        tracker=tracker,
        dst=dst,
        text=args.message,
        top_k=args.top_k,
        raw_threshold=raw_threshold,
        context_threshold=context_threshold,
        export_file=export_file,
        turn_index=1,
        use_state=not args.no_state,
        export_mode=args.export_mode,
    )


if __name__ == "__main__":
    main()
