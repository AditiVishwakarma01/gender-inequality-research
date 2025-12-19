import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Load final merged data
df = pd.read_csv("data/processed/final_analysis.csv")

X = sm.add_constant(df["female_lfp"])
y = df["cases"]

model = sm.OLS(y, X).fit()

df["predicted"] = model.predict(X)
df["residuals"] = y - df["predicted"]

plt.figure(figsize=(8, 5))
plt.scatter(df["predicted"], df["residuals"])
plt.axhline(0, linestyle="--")
plt.title("Residual Plot: Crime vs Female LFP Model")
plt.xlabel("Predicted Crime Cases")
plt.ylabel("Residuals")
plt.grid(True)
plt.tight_layout()
plt.savefig("final_relationship_plot.png", dpi=300, bbox_inches="tight")
plt.close()
