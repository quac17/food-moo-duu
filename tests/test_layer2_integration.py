from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.layer2_adaptive_recommendation.online_learning import HebbianLearner
from src.layer2_adaptive_recommendation.recommendation_engine import RecommendationEngine


class Layer2IntegrationTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.layer2 = self.root / "layer2"
        self.dataset_dir = self.layer2 / "dataset_v001"
        self.layer2.mkdir(parents=True, exist_ok=True)
        self.dataset_dir.mkdir(parents=True, exist_ok=True)

        (self.layer2 / "datasets.json").write_text(
            json.dumps(
                {
                    "active_dataset": "dataset_v001",
                    "available_datasets": ["dataset_v001"],
                }
            ),
            encoding="utf-8",
        )
        (self.dataset_dir / "dataset_manifest.json").write_text(
            json.dumps(
                {
                    "dataset_id": "dataset_v001",
                    "inputs": {
                        "canonical_matrix": "food_weight_matrix.json",
                        "legacy_matrix": "../dishes_100.json",
                    },
                }
            ),
            encoding="utf-8",
        )

        canonical_dishes = {
            "dishes": [
                {
                    "id": "dish_001",
                    "name": "Pho bo",
                    "popularity_score": 1.0,
                    "tag_weights": {
                        "pref_soup": 0.9,
                        "pref_warm_drink": 0.6,
                        "pref_instant": 0.1,
                    },
                },
                {
                    "id": "dish_002",
                    "name": "Banh mi",
                    "popularity_score": 1.0,
                    "tag_weights": {
                        "pref_soup": 0.2,
                        "pref_warm_drink": 0.1,
                        "pref_instant": 0.8,
                        "pref_convenient": 0.8,
                    },
                },
            ]
        }
        (self.dataset_dir / "food_weight_matrix.json").write_text(
            json.dumps(canonical_dishes, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_feedback_updates_runtime_not_canonical(self) -> None:
        engine = RecommendationEngine(data_dir=self.layer2)
        learner = HebbianLearner(engine=engine, lr_positive=0.1, lr_negative=0.02)

        context_turn1 = {
            "pref_soup": 0.9,
            "pref_warm_drink": 0.9,
            "pref_instant": 0.2,
            "pref_convenient": 0.2,
        }
        before = engine.recommend(context_turn1, top_k=2)
        self.assertEqual(before[0]["id"], "dish_001")

        # Turn 2: user quay xe va chon dish_002 theo context moi.
        context_turn2 = {
            "pref_soup": 0.1,
            "pref_warm_drink": 0.1,
            "pref_instant": 0.9,
            "pref_convenient": 0.9,
        }
        learner.update_after_choice("dish_002", context_turn2)

        runtime_file = self.layer2 / "runtime" / "dataset_v001_dishes_runtime.json"
        self.assertTrue(runtime_file.exists())

        canonical_payload = json.loads(
            (self.dataset_dir / "food_weight_matrix.json").read_text(encoding="utf-8")
        )
        runtime_payload = json.loads(runtime_file.read_text(encoding="utf-8"))

        canonical_dish2 = next(d for d in canonical_payload["dishes"] if d["id"] == "dish_002")
        runtime_dish2 = next(d for d in runtime_payload["dishes"] if d["id"] == "dish_002")

        # Canonical giu nguyen, runtime thay doi sau feedback.
        self.assertAlmostEqual(canonical_dish2["tag_weights"]["pref_instant"], 0.8)
        self.assertGreater(runtime_dish2["tag_weights"]["pref_instant"], 0.8)

        # Engine moi se uu tien runtime va ranking phan anh hoc online.
        engine_after = RecommendationEngine(data_dir=self.layer2)
        self.assertEqual(engine_after.source_kind, "runtime")
        after = engine_after.recommend(context_turn2, top_k=2)
        self.assertEqual(after[0]["id"], "dish_002")


if __name__ == "__main__":
    unittest.main()
