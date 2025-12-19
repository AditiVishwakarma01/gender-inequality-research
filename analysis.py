import pandas as pd

# =========================
# LOAD DATA
# =========================
CRIME_PATH = "data/processed/crimes_long.csv"
LFP_PATH = "data/raw/female_lfp_india_2001_2022.csv"

# -------------------------
# Crimes data (already clean)
# -------------------------
crimes = pd.read_csv(CRIME_PATH)

print("Loaded crimes:", crimes.shape)

# -------------------------
# Female LFP data (World Bank fix)
# -------------------------
lfp = pd.read_csv(
    LFP_PATH,
    sep=",",                  # World Bank files are comma-separated
    skiprows=4,               # ⬅️ CRITICAL: skip metadata rows
    encoding="utf-8"
)

print("Loaded raw LFP:", lfp.shape)

# Keep only India
lfp_india = lfp[lfp["Country Name"] == "India"]

# Identify year columns safely
year_cols = [c for c in lfp_india.columns if c.isdigit()]

# Convert to long format
lfp_long = lfp_india.melt(
    id_vars=["Country Name"],
    value_vars=year_cols,
    var_name="year",
    value_name="female_lfp"
)

lfp_long["year"] = lfp_long["year"].astype(int)
lfp_long["female_lfp"] = pd.to_numeric(lfp_long["female_lfp"], errors="coerce")

print("\nLFP (India) sample:")
print(lfp_long.head())

# -------------------------
# Aggregate crimes per year
# -------------------------
total_crimes = (
    crimes.groupby("year", as_index=False)["cases"]
    .sum()
)

print("\nTotal crimes per year:")
print(total_crimes.head())

# -------------------------
# Merge datasets
# -------------------------
merged = pd.merge(
    total_crimes,
    lfp_long[["year", "female_lfp"]],
    on="year",
    how="inner"
)

print("\nMerged data:")
print(merged.head())
print("Merged shape:", merged.shape)

# -------------------------
# Correlation analysis
# -------------------------
print("\nCorrelation Analysis")
print("Pearson:", merged["cases"].corr(merged["female_lfp"], method="pearson"))
print("Spearman:", merged["cases"].corr(
    merged["female_lfp"], method="spearman"))

# -------------------------
# Save final dataset
# -------------------------
merged.to_csv("data/processed/final_analysis.csv", index=False)
print("\n✅ Saved data/processed/final_analysis.csv")
