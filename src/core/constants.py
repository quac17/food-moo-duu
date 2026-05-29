from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
LAYER1_DATA_DIR = DATA_DIR / "layer1"
LAYER2_DATA_DIR = DATA_DIR / "layer2"
LAYER3_DATA_DIR = DATA_DIR / "layer3"

TIME_TAGS = [
    "time_morning",
    "time_noon",
    "time_afternoon",
    "time_evening",
    "time_late_night",
    "time_weekday",
    "time_weekend",
    "time_busy",
    "time_relaxed",
    "time_quick_meal",
]

WEATHER_TAGS = [
    "weather_hot",
    "weather_cold",
    "weather_rainy",
    "weather_sunny",
    "weather_humid",
    "weather_dry",
    "weather_windy",
    "weather_stormy",
    "weather_cloudy",
    "weather_mild",
]

MOOD_TAGS = [
    "mood_happy",
    "mood_sad",
    "mood_stressed",
    "mood_excited",
    "mood_tired",
    "mood_energetic",
    "mood_adventurous",
    "mood_lazy",
    "mood_social",
    "mood_comfort_seek",
]

ALL_TAGS = TIME_TAGS + WEATHER_TAGS + MOOD_TAGS
