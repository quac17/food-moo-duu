from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.core.constants import ALL_TAGS
from src.layer2_adaptive_recommendation.online_learning import HebbianLearner
from src.layer2_adaptive_recommendation.recommendation_engine import RecommendationEngine


class Layer2TestCase(unittest.TestCase):
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

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _write_legacy(self, dishes: list[dict]) -> None:
        (self.layer2 / "dishes_100.json").write_text(
            json.dumps({"dishes": dishes}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _write_canonical(self, dishes: list[dict]) -> None:
        (self.dataset_dir / "food_weight_matrix.json").write_text(
            json.dumps({"dishes": dishes}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def test_expands_layer1_context_to_similar_layer2_tags(self) -> None:
        self._write_legacy(
            [
                {
                    "id": "dish_001",
                    "name": "Dish 1",
                    "tag_weights": {
                        "time_quick_meal": 0.9,
                        "pref_convenient": 0.7,
                        "time_busy": 0.4,
                        "time_noon": 0.2,
                    },
                }
            ]
        )

        engine = RecommendationEngine(data_dir=self.layer2)
        self.assertEqual(engine.source_kind, "legacy")
        dish = engine.get_dish_by_id("dish_001")

        score = engine.score_dish(dish, {"pref_instant": 1.0})

        self.assertEqual(dish["tag_weights"]["time_quick_meal"], 0.9)
        self.assertEqual(dish["tag_weights"]["pref_convenient"], 0.7)
        self.assertEqual(dish["tag_weights"]["time_busy"], 0.4)
        self.assertGreater(score, 1.0)

    def test_prefers_canonical_when_available(self) -> None:
        self._write_legacy(
            [
                {"id": "dish_001", "name": "Legacy 1", "tag_weights": {}},
                {"id": "dish_002", "name": "Legacy 2", "tag_weights": {}},
            ]
        )
        self._write_canonical(
            [
                {"id": "dish_001", "name": "Canonical 1", "tag_weights": {}},
            ]
        )

        engine = RecommendationEngine(data_dir=self.layer2)
        self.assertEqual(engine.source_kind, "canonical")
        self.assertEqual(len(engine.dishes), 1)

    def test_recommend_tie_break_is_stable(self) -> None:
        dishes = [
            {
                "id": "dish_001",
                "name": "A",
                "popularity_score": 1.0,
                "tag_weights": {"time_noon": 1.0},
            },
            {
                "id": "dish_002",
                "name": "B",
                "popularity_score": 2.0,
                "tag_weights": {"time_noon": 1.0},
            },
        ]
        self._write_canonical(dishes)
        self._write_legacy([])

        engine = RecommendationEngine(data_dir=self.layer2)
        out = engine.recommend({"time_noon": 0.8}, top_k=2)
        self.assertEqual([item["id"] for item in out], ["dish_002", "dish_001"])

    def test_hebbian_update_persists(self) -> None:
        self._write_canonical([])
        self._write_legacy(
            [
                {
                    "id": "dish_001",
                    "name": "Dish 1",
                    "tag_weights": {
                        "time_quick_meal": 0.95,
                        "pref_convenient": 0.7,
                        "time_busy": 0.4,
                        "time_noon": 0.2,
                    },
                }
            ]
        )

        engine = RecommendationEngine(data_dir=self.layer2)
        learner = HebbianLearner(engine=engine, lr_positive=0.1, lr_negative=0.02)

        learner.update_after_choice("dish_001", {"pref_instant": 1.0})

        runtime_file = self.layer2 / "runtime" / "dataset_v001_dishes_runtime.json"
        payload = json.loads(runtime_file.read_text(encoding="utf-8"))
        saved = payload["dishes"][0]["tag_weights"]
        self.assertGreater(saved["time_quick_meal"], 0.95)
        self.assertGreater(saved["pref_convenient"], 0.7)
        self.assertGreater(saved["time_busy"], 0.4)


if __name__ == "__main__":
    unittest.main()
