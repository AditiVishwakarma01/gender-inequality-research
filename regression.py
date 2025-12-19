import pandas as pd
import statsmodels.api as sm

# =========================
# LOAD FINAL MERGED DATA
# =========================
DATA_PATH = "data/processed/final_analysis.csv"
df = pd.read_csv(DATA_PATH)

print("Loaded data shape:", df.shape)
print(df.head())

# =========================
# PREPARE VARIABLES
# =========================
X = df["female_lfp"]
y = df["cases"]

# Add constant for intercept
X = sm.add_constant(X)

# =========================
# FIT REGRESSION MODEL
# =========================
model = sm.OLS(y, X).fit()

# =========================
# OUTPUT RESULTS
# =========================
print("\n=== LINEAR REGRESSION RESULTS ===")
print(model.summary())

# =========================
# SAVE RESULTS
# =========================
results_df = pd.DataFrame({
    "coef": model.params,
    "p_value": model.pvalues
})

results_df.to_csv("data/processed/regression_results.csv")
print("\nSaved regression_results.csv")
