import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

# =========================
# PATH SETUP
# =========================
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "final_analysis.csv"
OUT_DIR = ROOT / "data" / "processed"
OUT_DIR.mkdir(exist_ok=True)

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(DATA_PATH)

# Sort by time (VERY IMPORTANT — not random split)
df = df.sort_values("year").reset_index(drop=True)

# =========================
# FEATURE ENGINEERING
# =========================
df["cases_lag_1"] = df["cases"].shift(1)
df["cases_rolling_3"] = df["cases"].rolling(window=3).mean()

df = df.dropna().reset_index(drop=True)

FEATURES = [
    "female_lfp",
    "year",
    "cases_lag_1",
    "cases_rolling_3"
]
TARGET = "cases"

X = df[FEATURES]
y = df[TARGET]

# =========================
# TIME-AWARE TRAIN / TEST
# =========================
split_idx = int(len(df) * 0.8)

X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# =========================
# MODELS
# =========================
models = {
    "RandomForest": RandomForestRegressor(
        n_estimators=300,
        max_depth=6,
        random_state=42
    ),
    "GradientBoosting": GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        random_state=42
    )
}

results = []
feature_importances = []

# =========================
# TRAIN + EVALUATE
# =========================
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    results.append({
        "model": name,
        "MAE": mae,
        "RMSE": rmse
    })

    if hasattr(model, "feature_importances_"):
        for f, imp in zip(FEATURES, model.feature_importances_):
            feature_importances.append({
                "model": name,
                "feature": f,
                "importance": imp
            })

# =========================
# SAVE OUTPUTS
# =========================
results_df = pd.DataFrame(results)
fi_df = pd.DataFrame(feature_importances)

results_df.to_csv(OUT_DIR / "ml_model_performance.csv", index=False)
fi_df.to_csv(OUT_DIR / "ml_feature_importance.csv", index=False)

# =========================
# PRINT SUMMARY
# =========================
print("\nML MODEL PERFORMANCE")
print(results_df)

print("\nFEATURE IMPORTANCE")
print(fi_df)
