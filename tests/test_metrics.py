from __future__ import annotations

import unittest

from src.evaluation.metrics import hit_at_k, mrr, ndcg_at_k, precision_at_k, recall_at_k


class MetricsTestCase(unittest.TestCase):
    def test_hit_at_k(self) -> None:
        ranked = ["dish_001", "dish_002", "dish_003"]
        self.assertEqual(hit_at_k(ranked, ["dish_002"], 2), 1.0)
        self.assertEqual(hit_at_k(ranked, ["dish_003"], 2), 0.0)

    def test_mrr(self) -> None:
        ranked = ["dish_001", "dish_002", "dish_003"]
        self.assertAlmostEqual(mrr(ranked, ["dish_002"]), 0.5)
        self.assertAlmostEqual(mrr(ranked, ["dish_999"]), 0.0)

    def test_ndcg_at_k(self) -> None:
        ranked = ["dish_001", "dish_002", "dish_003"]
        self.assertGreater(ndcg_at_k(ranked, ["dish_002"], 3), 0.0)
        self.assertEqual(ndcg_at_k(ranked, ["dish_999"], 3), 0.0)

    def test_precision_recall_at_k(self) -> None:
        ranked = ["dish_001", "dish_002", "dish_003"]
        relevant = ["dish_001", "dish_003"]
        self.assertAlmostEqual(precision_at_k(ranked, relevant, 2), 0.5)
        self.assertAlmostEqual(recall_at_k(ranked, relevant, 2), 0.5)


if __name__ == "__main__":
    unittest.main()
