from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
LAYER1_DATA_DIR = DATA_DIR / "layer1"
LAYER2_DATA_DIR = DATA_DIR / "layer2"
LAYER3_DATA_DIR = DATA_DIR / "layer3"
COMMON_CONFIG_FILE = DATA_DIR / "common_config.json"
LAYER2_CONFIG_FILE = LAYER2_DATA_DIR / "layer2_config.json"

# Fallback tags de dam bao he thong van chay duoc neu data bi thieu.
DEFAULT_TAGS = [
    "time_morning",
    "time_noon",
    "time_afternoon",
    "time_evening",
    "time_night",
    "time_snacks",
    "weather_hot",
    "weather_cold",
    "weather_rain",
    "weather_storm",
    "weather_normal",
    "mood_exhausted",
    "mood_sluggish",
    "mood_normal",
    "mood_happy",
    "mood_excited",
    "mood_starving",
    "mood_stressed",
    "mood_lonely",
    "mood_sick",
    "mood_lazy",
    "mood_gossip",
    "mood_bored_taste",
    "mood_watching_movie",
    "mood_gathering",
    "pref_spicy",
    "pref_sweet",
    "pref_sour",
    "pref_flavorful",
    "pref_bland",
    "pref_soup",
    "pref_dry",
    "pref_cold",
    "pref_crunchy",
    "pref_soft",
    "pref_bbq_hotpot",
    "pref_instant",
    "pref_rice",
    "pref_raw",
    "pref_rich",
    "pref_low_fat",
    "pref_high_protein",
    "pref_vegetarian",
    "pref_convenient",
    "pref_finger_food",
    "pref_caffeine",
    "pref_milky",
    "pref_fruity",
    "pref_tea_base",
    "pref_fizzy",
    "pref_cold_drink",
    "pref_warm_drink",
    "pref_herbal",
]


def read_json_safely(file_path: Path, default: Any) -> Any:
    if not file_path.exists():
        return default
    try:
        return json.loads(file_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default


def load_layer1_tags() -> List[str]:
    payload = read_json_safely(LAYER1_DATA_DIR / "tags.json", {})
    if isinstance(payload, dict):
        if isinstance(payload.get("tag_ids"), list):
            return payload["tag_ids"]
        if isinstance(payload.get("tags"), list):
            return payload["tags"]
    return list(DEFAULT_TAGS)


def load_hyperparameters() -> Dict[str, float]:
    payload = read_json_safely(COMMON_CONFIG_FILE, {})
    hyper = payload.get("hyperparameters", {}) if isinstance(payload, dict) else {}
    defaults = {
        "learning_rate": 0.08,
        "punishment_rate": -0.02,
        "context_decay": 0.92,
        "epsilon": 0.2,
        "fitness_decay": 0.95,
    }
    return {
        "learning_rate": float(hyper.get("learning_rate", defaults["learning_rate"])),
        "punishment_rate": float(hyper.get("punishment_rate", defaults["punishment_rate"])),
        "context_decay": float(hyper.get("context_decay", defaults["context_decay"])),
        "epsilon": float(hyper.get("epsilon", defaults["epsilon"])),
        "fitness_decay": float(hyper.get("fitness_decay", defaults["fitness_decay"])),
    }


def load_layer2_config() -> Dict[str, Any]:
    payload = read_json_safely(LAYER2_CONFIG_FILE, {})
    if not isinstance(payload, dict):
        payload = {}

    defaults = {
        "learning": {
            "positive": 0.08,
            "negative": 0.02,
            "feedback_penalty": 0.02,
            "active_threshold": 0.25,
        },
        "similarity": {},
    }

    learning = payload.get("learning", {}) if isinstance(payload, dict) else {}
    similarity = payload.get("similarity", {}) if isinstance(payload, dict) else {}
    if not isinstance(learning, dict):
        learning = {}
    if not isinstance(similarity, dict):
        similarity = {}

    return {
        "learning": {
            "positive": float(learning.get("positive", defaults["learning"]["positive"])),
            "negative": float(learning.get("negative", defaults["learning"]["negative"])),
            "feedback_penalty": float(learning.get("feedback_penalty", defaults["learning"]["feedback_penalty"])),
            "active_threshold": float(learning.get("active_threshold", defaults["learning"]["active_threshold"])),
        },
        "similarity": similarity,
    }


ALL_TAGS = load_layer1_tags()
HYPERPARAMS = load_hyperparameters()
LAYER2_CONFIG = load_layer2_config()
