from src.layer2_adaptive_recommendation.recommendation_engine import RecommendationEngine


def main() -> None:
    engine = RecommendationEngine()
    context = {
        "time_noon": 0.8,
        "weather_hot": 0.7,
        "pref_convenient": 0.6,
        "mood_happy": 0.5,
    }
    recommendations = engine.recommend(context, top_k=5)
    print("Layer2 ok. Top-5:", recommendations)


if __name__ == "__main__":
    main()
