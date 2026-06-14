MODEL_PATH = "models/model_diabetes_terbaik.pkl"
METRICS_PATH = "models/metrics.json"

# Model utama yang dipilih (3 model dilatih -> 1 dipakai sebagai skor utama)
BEST_MODEL_KEY = "gb"

# key -> file pkl masing-masing model nyata
MODEL_FILES = {
    "gb": "models/model_gb.pkl",
    "rf": "models/model_rf.pkl",
    "lr": "models/model_lr.pkl",
}

DEFAULT_FEATURE_ORDER = [
    "HighBP",
    "HighChol",
    "CholCheck",
    "BMI",
    "Smoker",
    "Stroke",
    "HeartDiseaseorAttack",
    "PhysActivity",
    "Fruits",
    "Veggies",
    "HvyAlcoholConsump",
    "AnyHealthcare",
    "NoDocbcCost",
    "GenHlth",
    "MentHlth",
    "PhysHlth",
    "DiffWalk",
    "Sex",
    "Age",
    "Education",
    "Income",
]
