"""Gan lai tag cho menu Layer 2 va bo sung mon moi.

Nguon: data/layer2/dishes_100.json -> migrate sang food_weight_matrix.json
Chi dung tag_ids trong data/layer1/tags.json (khong giu tag legacy).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
TAGS_FILE = ROOT / "data" / "layer1" / "tags.json"
OUTPUT_FILE = ROOT / "data" / "layer2" / "dishes_100.json"

# Mau tag theo nhom mon (chi tag co y nghia, khong nhieu 0.xxx)
CATEGORY_TAGS: Dict[str, Dict[str, float]] = {
    "pho": {
        "time_morning": 0.78,
        "time_evening": 0.55,
        "pref_soup": 0.92,
        "pref_warm_drink": 0.55,
        "weather_cold": 0.72,
        "mood_sluggish": 0.55,
        "pref_high_protein": 0.5,
        "pref_flavorful": 0.45,
    },
    "bun_soup": {
        "time_noon": 0.62,
        "time_evening": 0.45,
        "pref_soup": 0.82,
        "pref_flavorful": 0.55,
        "weather_rain": 0.35,
    },
    "bun_spicy": {
        "time_noon": 0.65,
        "pref_soup": 0.8,
        "pref_spicy": 0.82,
        "pref_flavorful": 0.72,
        "mood_excited": 0.45,
    },
    "com_rice": {
        "time_noon": 0.75,
        "time_evening": 0.5,
        "pref_rice": 0.92,
        "pref_convenient": 0.5,
        "pref_high_protein": 0.45,
    },
    "lau": {
        "pref_bbq_hotpot": 0.93,
        "pref_soup": 0.88,
        "mood_gathering": 0.85,
        "time_evening": 0.7,
        "weather_rain": 0.6,
        "weather_cold": 0.55,
        "pref_flavorful": 0.65,
        "pref_high_protein": 0.6,
        "pref_finger_food": 0.35,
    },
    "bbq_grill": {
        "pref_bbq_hotpot": 0.82,
        "mood_gathering": 0.72,
        "time_evening": 0.68,
        "pref_flavorful": 0.7,
        "pref_high_protein": 0.65,
        "pref_finger_food": 0.45,
    },
    "soup_chao": {
        "mood_sick": 0.75,
        "pref_soup": 0.95,
        "pref_soft": 0.85,
        "pref_bland": 0.7,
        "pref_warm_drink": 0.6,
        "weather_cold": 0.65,
        "time_evening": 0.4,
    },
    "goi_salad": {
        "pref_cold": 0.82,
        "pref_raw": 0.75,
        "pref_low_fat": 0.68,
        "pref_vegetarian": 0.4,
        "weather_hot": 0.6,
        "time_noon": 0.45,
    },
    "snack_finger": {
        "pref_finger_food": 0.88,
        "time_snacks": 0.85,
        "mood_gathering": 0.65,
        "mood_watching_movie": 0.6,
        "pref_crunchy": 0.55,
        "time_evening": 0.5,
    },
    "banh_mi_instant": {
        "time_morning": 0.72,
        "pref_instant": 0.88,
        "pref_convenient": 0.85,
        "pref_crunchy": 0.45,
    },
    "mi_noodle": {
        "pref_instant": 0.72,
        "pref_convenient": 0.68,
        "pref_flavorful": 0.55,
        "time_evening": 0.45,
    },
    "mi_quang_dry": {
        "time_morning": 0.68,
        "time_noon": 0.55,
        "pref_dry": 0.72,
        "pref_spicy": 0.58,
        "pref_flavorful": 0.68,
    },
    "seafood": {
        "pref_high_protein": 0.78,
        "pref_flavorful": 0.65,
        "pref_spicy": 0.4,
        "time_evening": 0.55,
    },
    "vegetarian": {
        "pref_vegetarian": 0.92,
        "pref_bland": 0.65,
        "pref_low_fat": 0.55,
        "pref_soup": 0.4,
    },
    "korean_japanese": {
        "mood_excited": 0.55,
        "mood_gathering": 0.5,
        "pref_instant": 0.55,
        "pref_soup": 0.55,
        "pref_spicy": 0.45,
    },
    "fried": {
        "pref_crunchy": 0.75,
        "pref_rich": 0.6,
        "time_evening": 0.55,
        "mood_happy": 0.45,
        "pref_finger_food": 0.4,
    },
    "drink_beer": {
        "mood_gathering": 0.9,
        "pref_fizzy": 0.9,
        "pref_cold_drink": 0.95,
        "time_evening": 0.82,
        "weather_hot": 0.85,
        "pref_finger_food": 0.7,
        "pref_soup": -0.35,
        "pref_bbq_hotpot": -0.2,
    },
    "drink_soft": {
        "pref_fizzy": 0.88,
        "pref_cold_drink": 0.92,
        "pref_sweet": 0.68,
        "weather_hot": 0.7,
        "mood_gathering": 0.45,
        "mood_watching_movie": 0.4,
    },
    "drink_coffee_tea": {
        "pref_caffeine": 0.88,
        "pref_tea_base": 0.75,
        "pref_cold_drink": 0.65,
        "time_morning": 0.55,
        "time_afternoon": 0.6,
        "pref_milky": 0.45,
    },
    "drink_herbal": {
        "pref_herbal": 0.88,
        "pref_warm_drink": 0.55,
        "pref_cold_drink": 0.7,
        "weather_hot": 0.65,
        "pref_sweet": 0.35,
    },
    "drink_dessert": {
        "pref_sweet": 0.92,
        "pref_milky": 0.72,
        "pref_cold": 0.75,
        "pref_fruity": 0.55,
        "time_snacks": 0.6,
        "mood_happy": 0.5,
    },
}

# (id, name, is_drink, category, popularity, extra_tags)
DISH_SPECS: List[Tuple[str, str, bool, str, float, Dict[str, float]]] = [
    ("dish_001", "Pho bo", False, "pho", 1.0, {"pref_high_protein": 0.65}),
    ("dish_002", "Bun bo Hue", False, "bun_spicy", 1.0, {}),
    ("dish_003", "Com tam", False, "com_rice", 1.0, {"pref_flavorful": 0.55}),
    ("dish_004", "Banh mi trung", False, "banh_mi_instant", 1.0, {"pref_high_protein": 0.4}),
    ("dish_005", "Mi xao bo", False, "mi_noodle", 0.95, {"pref_high_protein": 0.6}),
    ("dish_006", "Hu tieu nam vang", False, "bun_soup", 0.95, {"pref_soup": 0.88}),
    ("dish_007", "Bun rieu", False, "bun_spicy", 0.95, {"pref_sour": 0.55}),
    ("dish_008", "Goi cuon", False, "goi_salad", 0.95, {"pref_soft": 0.45}),
    ("dish_009", "Bun cha", False, "bbq_grill", 1.0, {"time_noon": 0.55, "pref_soup": 0.35}),
    ("dish_010", "Chao ga", False, "soup_chao", 0.95, {}),
    ("dish_011", "Canh chua ca", False, "seafood", 0.9, {"pref_soup": 0.75, "pref_sour": 0.7}),
    ("dish_012", "Com ga xoi mo", False, "com_rice", 0.95, {}),
    ("dish_013", "Sup cua", False, "soup_chao", 0.9, {"pref_high_protein": 0.55}),
    ("dish_014", "Banh xeo", False, "fried", 0.95, {"mood_gathering": 0.55, "pref_crunchy": 0.65}),
    ("dish_015", "Lau thai", False, "lau", 1.0, {"pref_spicy": 0.75, "pref_sour": 0.5}),
    ("dish_016", "Banh canh cua", False, "bun_soup", 0.9, {"pref_soup": 0.9}),
    ("dish_017", "Bun mam", False, "bun_spicy", 0.9, {"pref_flavorful": 0.8}),
    ("dish_018", "Mi quang", False, "mi_quang_dry", 1.0, {}),
    ("dish_019", "Bun dau mam tom", False, "bun_spicy", 0.9, {"mood_gathering": 0.5}),
    ("dish_020", "Xoi ga", False, "com_rice", 0.9, {"time_morning": 0.65}),
    ("dish_021", "Com chien duong chau", False, "com_rice", 0.9, {"pref_instant": 0.45}),
    ("dish_022", "Mien ga", False, "soup_chao", 0.9, {"mood_sick": 0.6}),
    ("dish_023", "Com suon nuong", False, "com_rice", 1.0, {"pref_bbq_hotpot": 0.45}),
    ("dish_024", "Bun thit nuong", False, "bun_soup", 0.95, {"pref_bbq_hotpot": 0.5}),
    ("dish_025", "Salad ca ngu", False, "goi_salad", 0.85, {"pref_high_protein": 0.6}),
    ("dish_026", "Bun oc", False, "bun_spicy", 0.9, {"pref_sour": 0.6}),
    ("dish_027", "Com chay kho quet", False, "vegetarian", 0.85, {"pref_rice": 0.8}),
    ("dish_028", "Banh cuon", False, "banh_mi_instant", 0.95, {"time_morning": 0.8, "pref_soft": 0.55}),
    ("dish_029", "Cha ca", False, "seafood", 0.95, {"pref_flavorful": 0.75}),
    ("dish_030", "Lau nam", False, "lau", 0.95, {"pref_vegetarian": 0.55}),
    ("dish_031", "Mi tron", False, "mi_noodle", 0.95, {"pref_dry": 0.55}),
    ("dish_032", "Com ga hoi an", False, "com_rice", 0.9, {}),
    ("dish_033", "Bun bo xao", False, "bun_spicy", 0.9, {}),
    ("dish_034", "Ca kho to", False, "com_rice", 0.9, {"pref_rich": 0.65, "pref_flavorful": 0.7}),
    ("dish_035", "Canh rau cu", False, "vegetarian", 0.85, {"pref_soup": 0.8}),
    ("dish_036", "Banh da cua", False, "seafood", 0.9, {"pref_spicy": 0.55}),
    ("dish_037", "Bun suon", False, "bun_soup", 0.9, {}),
    ("dish_038", "Bap xao bo", False, "mi_noodle", 0.85, {}),
    ("dish_039", "Nem nuong", False, "bbq_grill", 0.95, {"pref_finger_food": 0.6}),
    ("dish_040", "Banh can", False, "banh_mi_instant", 0.85, {"time_evening": 0.55}),
    ("dish_041", "Pho ga", False, "pho", 0.95, {"pref_bland": 0.45}),
    ("dish_042", "Bun ca", False, "bun_soup", 0.9, {}),
    ("dish_043", "Com hap la sen", False, "com_rice", 0.85, {"pref_bland": 0.55}),
    ("dish_044", "Lau ga la e", False, "lau", 0.9, {"pref_herbal": 0.45}),
    ("dish_045", "Muc xao chua ngot", False, "seafood", 0.9, {"pref_sour": 0.6}),
    ("dish_046", "Banh trang tron", False, "snack_finger", 0.95, {"pref_spicy": 0.55, "pref_sour": 0.5}),
    ("dish_047", "Cha gio", False, "fried", 0.95, {"mood_gathering": 0.6}),
    ("dish_048", "Banh bao", False, "snack_finger", 0.9, {"pref_soft": 0.55, "pref_milky": 0.35}),
    ("dish_049", "Suon ram man", False, "com_rice", 0.9, {"pref_sweet": 0.45, "pref_rich": 0.55}),
    ("dish_050", "Com cuon rong bien", False, "korean_japanese", 0.85, {"pref_rice": 0.7}),
    ("dish_051", "Ca nuong muoi ot", False, "bbq_grill", 0.9, {"pref_spicy": 0.55}),
    ("dish_052", "Lau kim chi", False, "lau", 0.95, {"pref_spicy": 0.7, "pref_sour": 0.55}),
    ("dish_053", "Bun ga nuong", False, "bun_soup", 0.9, {"pref_bbq_hotpot": 0.45}),
    ("dish_054", "Mi ga tiem", False, "soup_chao", 0.9, {}),
    ("dish_055", "Com tron ngu sac", False, "korean_japanese", 0.9, {"pref_rice": 0.75}),
    ("dish_056", "Bo luc lac", False, "bbq_grill", 1.0, {}),
    ("dish_057", "Canh ga chien nuoc mam", False, "fried", 0.85, {"pref_soup": 0.5}),
    ("dish_058", "Mien tron", False, "mi_quang_dry", 0.85, {}),
    ("dish_059", "Bap bo ham", False, "soup_chao", 0.9, {"pref_high_protein": 0.7}),
    ("dish_060", "Oc len xao dua", False, "seafood", 0.85, {"pref_spicy": 0.5}),
    ("dish_061", "Bun thang", False, "bun_soup", 0.9, {"time_morning": 0.55, "pref_bland": 0.5}),
    ("dish_062", "Com ca kho", False, "com_rice", 0.9, {"pref_rich": 0.55}),
    ("dish_063", "Banh canh ghe", False, "bun_soup", 0.9, {"pref_high_protein": 0.6}),
    ("dish_064", "Moc suon", False, "soup_chao", 0.85, {}),
    ("dish_065", "Ca vien chien", False, "snack_finger", 0.9, {}),
    ("dish_066", "Banh trang cuon thit heo", False, "goi_salad", 0.9, {"mood_gathering": 0.7}),
    ("dish_067", "Com chien hai san", False, "com_rice", 0.9, {"pref_instant": 0.5}),
    ("dish_068", "Ga nuong sa", False, "bbq_grill", 0.9, {"pref_herbal": 0.45}),
    ("dish_069", "Tom rang muoi", False, "seafood", 0.95, {"pref_crunchy": 0.5}),
    ("dish_070", "Sup bi do", False, "soup_chao", 0.85, {"pref_vegetarian": 0.45}),
    ("dish_071", "Bun kim chi", False, "bun_spicy", 0.85, {"pref_sour": 0.65}),
    ("dish_072", "Pho xao", False, "mi_noodle", 0.9, {"pref_instant": 0.55}),
    ("dish_073", "Banh canh gio heo", False, "bun_soup", 0.9, {}),
    ("dish_074", "Canh kim chi dau hu", False, "vegetarian", 0.85, {"pref_spicy": 0.5}),
    ("dish_075", "Com bo luc lac", False, "com_rice", 0.95, {"pref_bbq_hotpot": 0.4}),
    ("dish_076", "Mi ramen", False, "korean_japanese", 0.9, {"pref_soup": 0.7}),
    ("dish_077", "Bun tom", False, "bun_soup", 0.9, {}),
    ("dish_078", "Com ga nuong mat ong", False, "com_rice", 0.9, {"pref_sweet": 0.45}),
    ("dish_079", "Canh rong bien", False, "vegetarian", 0.8, {"pref_soup": 0.85}),
    ("dish_080", "Ga hap hanh", False, "soup_chao", 0.85, {"pref_bland": 0.6}),
    ("dish_081", "Com chien kim chi", False, "korean_japanese", 0.9, {"pref_spicy": 0.55}),
    ("dish_082", "Mi udon", False, "korean_japanese", 0.85, {}),
    ("dish_083", "Bun moc", False, "bun_soup", 0.85, {"pref_bland": 0.45}),
    ("dish_084", "Ca hoi ap chao", False, "seafood", 0.85, {"pref_rich": 0.55}),
    ("dish_085", "Com tron han quoc", False, "korean_japanese", 0.95, {"pref_spicy": 0.6}),
    ("dish_086", "Sup tom", False, "soup_chao", 0.85, {}),
    ("dish_087", "Banh ep", False, "snack_finger", 0.88, {}),
    ("dish_088", "Com cha ca", False, "com_rice", 0.9, {}),
    ("dish_089", "Lau bo", False, "lau", 1.0, {"pref_high_protein": 0.75}),
    ("dish_090", "Chao hai san", False, "soup_chao", 0.9, {"pref_high_protein": 0.55}),
    ("dish_091", "Bun cha ca", False, "bun_soup", 0.9, {}),
    ("dish_092", "Com suon trung", False, "com_rice", 0.9, {}),
    ("dish_093", "Mi quang ech", False, "mi_quang_dry", 0.85, {}),
    ("dish_094", "Goi ga bap cai", False, "goi_salad", 0.85, {}),
    ("dish_095", "Ga chien gion", False, "fried", 0.9, {}),
    ("dish_096", "Com cari ga", False, "com_rice", 0.85, {"pref_spicy": 0.55, "pref_rich": 0.5}),
    ("dish_097", "Bun suon chua", False, "bun_spicy", 0.85, {"pref_sour": 0.55}),
    ("dish_098", "Canh khoai tay ham", False, "soup_chao", 0.8, {}),
    ("dish_099", "Com tam bi cha", False, "com_rice", 0.95, {}),
    ("dish_100", "Pho cuon", False, "goi_salad", 0.9, {"mood_gathering": 0.55, "pref_raw": 0.5}),
    ("dish_101", "Bia tuoi uop lanh", True, "drink_beer", 0.95, {}),
    ("dish_102", "Coca Cola da", True, "drink_soft", 0.93, {}),
    ("dish_103", "Soda chanh", True, "drink_soft", 0.9, {"pref_sour": 0.55, "pref_fruity": 0.45}),
    # Mon moi bo sung da dang menu
    ("dish_104", "Tra da chanh", True, "drink_herbal", 0.88, {"pref_sour": 0.5}),
    ("dish_105", "Nuoc mia", True, "drink_herbal", 0.85, {"pref_sweet": 0.6}),
    ("dish_106", "Che ba mau", True, "drink_dessert", 0.9, {}),
    ("dish_107", "Bo kho banh mi", False, "bun_soup", 0.92, {"pref_high_protein": 0.75, "time_morning": 0.5}),
    ("dish_108", "Banh trang nuong", False, "snack_finger", 0.9, {"pref_spicy": 0.5}),
    ("dish_109", "Nem chua ran", False, "fried", 0.88, {"pref_sour": 0.55, "mood_gathering": 0.55}),
    ("dish_110", "Bun bo nam bo", False, "goi_salad", 0.92, {"pref_sour": 0.6, "pref_herbal": 0.4}),
    ("dish_111", "Mi cay", False, "mi_noodle", 0.9, {"pref_spicy": 0.88, "mood_excited": 0.5}),
    ("dish_112", "Com ga roti", False, "com_rice", 0.88, {"pref_rich": 0.5}),
    ("dish_113", "Lau hai san", False, "lau", 0.98, {"pref_raw": 0.45}),
    ("dish_114", "Thit nuong than", False, "bbq_grill", 0.93, {}),
    ("dish_115", "Bun rieu cua", False, "bun_spicy", 0.9, {"pref_sour": 0.65}),
    ("dish_116", "Kem oc que", True, "drink_dessert", 0.87, {"pref_cold_drink": 0.6}),
    ("dish_117", "Cafe sua da", True, "drink_coffee_tea", 0.94, {}),
    ("dish_118", "Banh flan", True, "drink_dessert", 0.86, {"pref_soft": 0.55}),
    ("dish_119", "Thit nuong cuon banh trang", False, "bbq_grill", 0.9, {"pref_finger_food": 0.75}),
    ("dish_120", "Bun bo kho", False, "bun_spicy", 0.91, {"pref_high_protein": 0.7}),
]


def load_tag_ids() -> List[str]:
    payload = json.loads(TAGS_FILE.read_text(encoding="utf-8"))
    return list(payload.get("tag_ids", []))


def merge_tags(category: str, extra: Dict[str, float]) -> Dict[str, float]:
    base = dict(CATEGORY_TAGS.get(category, {}))
    for tag, value in extra.items():
        if tag in base:
            base[tag] = max(base[tag], value) if value > 0 else min(base[tag], value)
        else:
            base[tag] = value
    return {tag: round(weight, 4) for tag, weight in base.items() if weight != 0.0}


def build_catalog() -> Dict[str, object]:
    tag_ids = load_tag_ids()
    dishes = []
    for dish_id, name, is_drink, category, popularity, extra in DISH_SPECS:
        tag_weights = merge_tags(category, extra)
        # Chi giu tag thuoc schema chinh
        tag_weights = {tag: weight for tag, weight in tag_weights.items() if tag in tag_ids}
        dishes.append(
            {
                "id": dish_id,
                "name": name,
                "is_drink": is_drink,
                "popularity_score": popularity,
                "tag_weights": tag_weights,
            }
        )
    return {"dishes": dishes}


def main() -> None:
    catalog = build_catalog()
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")

    lau_count = sum(1 for d in catalog["dishes"] if d["tag_weights"].get("pref_bbq_hotpot", 0) >= 0.8)
    drink_count = sum(1 for d in catalog["dishes"] if d.get("is_drink"))
    print(f"Da ghi {len(catalog['dishes'])} mon -> {OUTPUT_FILE}")
    print(f"  - do uong: {drink_count}")
    print(f"  - mon lau/nuong (pref_bbq_hotpot>=0.8): {lau_count}")


if __name__ == "__main__":
    main()
