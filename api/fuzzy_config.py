SYSTEM_CONFIG = {
    "thresholds": {
        "pattern_score": 0.18,
        "how_long_used_signs_years": 11.50,
        "fisherman_age": 25.50,
        "how_often_used_signs": 3.25,
    },
    "fuzzy_params": {
        "pattern": { 'slope_start': 0.0495, 'slope_end': 0.3105 },
        "how_long": { 'slope_start': 8.1961, 'slope_end': 14.8039 },
        "age": { 'slope_start': 21.7364, 'slope_end': 29.2636 },
        "how_often": { 'slope_start': 1.0, 'slope_end': 6.5743 },
    },
    "scaling": {
        "min_val": 1/3,
        "max_val": 2/3
    }
}