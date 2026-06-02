from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from main import append_feedback_report_log


class FeedbackLoggingTestCase(unittest.TestCase):
    def test_append_feedback_report_log_writes_jsonl_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_file = Path(tmp_dir) / "feedback_reports.jsonl"
            report = SimpleNamespace(
                chosen_dish_id="dish_001",
                chosen_dish_name="Pho bo",
                score_before=0.8,
                score_after=0.92,
                delta=0.12,
            )
            context_scores = {
                "pref_soup": 0.9,
                "pref_warm_drink": 0.7,
                "pref_instant": 0.1,
            }

            append_feedback_report_log(report, context_scores, log_file)

            lines = log_file.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 1)
            payload = json.loads(lines[0])
            self.assertEqual(payload["dish_id"], "dish_001")
            self.assertEqual(payload["dish_name"], "Pho bo")
            self.assertEqual(payload["score_before"], 0.8)
            self.assertEqual(payload["score_after"], 0.92)
            self.assertEqual(payload["delta"], 0.12)
            self.assertEqual(payload["top_context"][0]["tag"], "pref_soup")


if __name__ == "__main__":
    unittest.main()
