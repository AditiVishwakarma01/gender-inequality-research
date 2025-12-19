# Gender Inequality & Crime in India (2001–2022)

## Overview
This project investigates the relationship between female labor force participation (FLFP) and reported crimes against women in India between 2001 and 2022. Using official crime statistics and World Bank labor force data, the study applies statistical analysis to explore long-term structural patterns.

## Data Sources
- National Crime Records Bureau (India): Crimes against women (year-wise)
- World Bank: Female labor force participation rate (modeled ILO estimate)

## Methodology
1. Data cleaning and reshaping into long format
2. Aggregation of total crimes per year
3. Correlation analysis (Pearson & Spearman)
4. Ordinary Least Squares (OLS) regression using statsmodels

## Machine Learning Extension
In addition to statistical analysis, machine learning models were implemented
to evaluate the predictive relationship between labor force participation and
reported crime trends.

Models used:
- Random Forest Regressor
- Gradient Boosting Regressor

A time-aware train–test split was applied to preserve temporal structure.
Model performance was evaluated using MAE and RMSE, and feature importance
was analyzed to interpret model behavior.

## Project Structure
- analytics/: data processing, statistical analysis, ML models
- data/: processed datasets
- ethics_and_limitations.md: ethical considerations and constraints
- final_relationship_plot.png: key visualization

## Key Findings
- A strong negative correlation between FLFP and reported crimes
- OLS regression indicates a statistically significant association (p < 0.001)
- Results highlight structural socio-economic patterns rather than causal claims

## Ethical Considerations
- Crime data reflects reporting behavior and institutional capacity
- The analysis does not imply causation
- Findings are contextualized within broader social and economic dynamics

## Tools Used
- Python
- Pandas, NumPy
- Matplotlib
- Statsmodels (OLS regression)
- Scikit-learn (Random Forest, Gradient Boosting)


## Author
Aditi
