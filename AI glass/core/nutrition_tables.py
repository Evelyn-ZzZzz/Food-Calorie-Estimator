"""食物密度与每 100g 卡路里查表（演示用；可换 CSV）。"""

# 单位：密度 g/cm³，kcal_per_100g 千卡/100g
# cook: raw | cooked | fried 等，简化演示
FOOD_DB: dict[str, dict] = {
    "apple": {"density_g_cm3": 0.85, "kcal_per_100g": 52, "cook": "raw"},
    "banana": {"density_g_cm3": 0.94, "kcal_per_100g": 89, "cook": "raw"},
    "bread": {"density_g_cm3": 0.35, "kcal_per_100g": 265, "cook": "baked"},
    "chips": {"density_g_cm3": 0.25, "kcal_per_100g": 536, "cook": "fried"},
    "pastry": {"density_g_cm3": 0.45, "kcal_per_100g": 350, "cook": "baked"},
    "snacks": {"density_g_cm3": 0.40, "kcal_per_100g": 450, "cook": "processed"},
    "broccoli": {"density_g_cm3": 0.64, "kcal_per_100g": 34, "cook": "cooked"},
    "cheese": {"density_g_cm3": 1.05, "kcal_per_100g": 402, "cook": "processed"},
    "chicken": {"density_g_cm3": 1.05, "kcal_per_100g": 165, "cook": "cooked"},
    "boiled chicken": {"density_g_cm3": 1.02, "kcal_per_100g": 165, "cook": "boiled"},
    "fried chicken": {"density_g_cm3": 0.95, "kcal_per_100g": 246, "cook": "fried"},
    "egg product": {"density_g_cm3": 1.03, "kcal_per_100g": 155, "cook": "cooked"},
    "fish": {"density_g_cm3": 1.05, "kcal_per_100g": 206, "cook": "cooked"},
    "fried fish": {"density_g_cm3": 0.95, "kcal_per_100g": 232, "cook": "fried"},
    "pizza": {"density_g_cm3": 0.75, "kcal_per_100g": 266, "cook": "baked"},
    "potatoes": {"density_g_cm3": 0.77, "kcal_per_100g": 77, "cook": "cooked"},
    "rice": {"density_g_cm3": 0.85, "kcal_per_100g": 130, "cook": "cooked"},
    "salad fresh": {"density_g_cm3": 0.45, "kcal_per_100g": 20, "cook": "raw"},
    "tomato": {"density_g_cm3": 0.96, "kcal_per_100g": 18, "cook": "raw"},
    "pasta": {"density_g_cm3": 0.75, "kcal_per_100g": 131, "cook": "cooked"},
    "melon": {"density_g_cm3": 0.92, "kcal_per_100g": 34, "cook": "raw"},
    "default": {"density_g_cm3": 0.90, "kcal_per_100g": 150, "cook": "unknown"},
}


def lookup_food(name: str) -> dict:
    key = name.strip().lower()
    if key in FOOD_DB:
        return FOOD_DB[key]
    for k, v in FOOD_DB.items():
        if k in key or key in k:
            return v
    return FOOD_DB["default"]


def mass_g_from_volume_cm3(volume_cm3: float, density_g_cm3: float) -> float:
    return max(0.0, volume_cm3 * density_g_cm3)


def kcal_from_mass_g(mass_g: float, kcal_per_100g: float) -> float:
    return max(0.0, mass_g * kcal_per_100g / 100.0)
