import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("data/processed/final_analysis.csv")
X = df[["female_lfp", "year"]]
y = df["cases"]

model = RandomForestRegressor(random_state=42)
model.fit(X, y)

explainer = shap.Explainer(model, X)
shap_values = explainer(X)

shap.summary_plot(shap_values, X)
