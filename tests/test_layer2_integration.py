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
                        "time_quick_meal": 0.9,
                        "pref_convenient": 0.7,
                        "time_busy": 0.6,
                    },
                },
                {
                    "id": "dish_002",
                    "name": "Banh mi",
                    "popularity_score": 1.0,
                    "tag_weights": {
                        "time_quick_meal": 0.2,
                        "pref_convenient": 0.2,
                        "time_busy": 0.1,
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

        context_turn1 = {"pref_instant": 1.0}
        before = engine.recommend(context_turn1, top_k=2)
        self.assertEqual(before[0]["id"], "dish_001")

        learner.update_after_choice("dish_001", context_turn1)

        runtime_file = self.layer2 / "runtime" / "dataset_v001_dishes_runtime.json"
        self.assertTrue(runtime_file.exists())

        canonical_payload = json.loads(
            (self.dataset_dir / "food_weight_matrix.json").read_text(encoding="utf-8")
        )
        runtime_payload = json.loads(runtime_file.read_text(encoding="utf-8"))

        canonical_dish1 = next(d for d in canonical_payload["dishes"] if d["id"] == "dish_001")
        runtime_dish1 = next(d for d in runtime_payload["dishes"] if d["id"] == "dish_001")

        # Canonical giu nguyen, runtime thay doi sau feedback.
        self.assertAlmostEqual(canonical_dish1["tag_weights"]["time_quick_meal"], 0.9)
        self.assertGreater(runtime_dish1["tag_weights"]["time_quick_meal"], 0.9)
        self.assertGreater(runtime_dish1["tag_weights"]["pref_convenient"], 0.7)
        self.assertGreater(runtime_dish1["tag_weights"]["time_busy"], 0.6)

        # Engine moi se uu tien runtime va ranking phan anh hoc online.
        engine_after = RecommendationEngine(data_dir=self.layer2)
        self.assertEqual(engine_after.source_kind, "runtime")
        after = engine_after.recommend(context_turn1, top_k=2)
        self.assertEqual(after[0]["id"], "dish_001")

    def test_non_selected_recommended_dishes_are_penalized(self) -> None:
        engine = RecommendationEngine(data_dir=self.layer2)
        learner = HebbianLearner(engine=engine, lr_positive=0.1, lr_negative=0.02)

        context_turn1 = {"pref_instant": 1.0}
        before = engine.recommend(context_turn1, top_k=2)
        self.assertEqual([item["id"] for item in before], ["dish_001", "dish_002"])

        pipeline_like_recommendations = before
        learner.update_after_choice("dish_001", context_turn1)

        for recommended in pipeline_like_recommendations:
            recommended_id = recommended["id"]
            if recommended_id == "dish_001":
                continue
            non_selected = engine.get_dish_by_id(recommended_id)
            engine.apply_negative_feedback_to_dish(
                non_selected,
                context_scores=context_turn1,
                penalty_rate=learner.lr_negative,
            )
        engine.save()

        runtime_payload = json.loads(
            (self.layer2 / "runtime" / "dataset_v001_dishes_runtime.json").read_text(encoding="utf-8")
        )
        runtime_dish2 = next(d for d in runtime_payload["dishes"] if d["id"] == "dish_002")

        self.assertLess(runtime_dish2["tag_weights"]["time_quick_meal"], 0.2)
        self.assertLess(runtime_dish2["tag_weights"]["pref_convenient"], 0.2)
        self.assertLess(runtime_dish2["tag_weights"]["time_busy"], 0.1)


if __name__ == "__main__":
    unittest.main()
