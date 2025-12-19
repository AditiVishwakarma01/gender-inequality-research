import pandas as pd
import matplotlib.pyplot as plt

# Load merged data
df = pd.read_csv("data/processed/final_analysis.csv")

plt.figure(figsize=(8,5))
plt.scatter(df["female_lfp"], df["cases"])
plt.plot(
    df["female_lfp"],
    df["cases"].rolling(window=3).mean(),
    linestyle="--"
)

plt.xlabel("Female Labor Force Participation (%)")
plt.ylabel("Total Crimes Against Women")
plt.title("Female Labor Force Participation vs Crimes Against Women (India, 2001–2022)")
plt.grid(True)

plt.tight_layout()
plt.savefig("final_relationship_plot.png", dpi=300)
plt.savefig("final_relationship_plot.png", dpi=300, bbox_inches="tight")
plt.close()
