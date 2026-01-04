# Telco Customer Churn Analysis

## Project Overview
An end-to-end customer churn analysis using a real-world telecom dataset.
The project covers data cleaning, exploratory data analysis, feature engineering,
churn prediction modeling, and actionable business recommendations.

## Objective
- Identify customers at high risk of churn
- Understand key churn drivers
- Provide data-driven retention strategies

## Key Steps
- Data cleaning and preparation
- Exploratory Data Analysis (EDA)
- Feature engineering
- Classification modeling (Logistic Regression, Random Forest)
- Model evaluation (ROC-AUC, Precision, Recall)
- Post-prediction error analysis
- Business recommendations

## Tools & Technologies
- Python (pandas, numpy, scikit-learn, matplotlib)
- Jupyter Notebook
- GitHub

## Key Insights
- Month-to-month contracts have the highest churn
- Early tenure customers (first 12 months) are

## What’s inside
- `Telco_Churn_Case_Study.ipynb` — narrative notebook (problem - cleaning - EDA - feature engineering - modelling - recommendations)
- `churn_analysis.py` — runnable python script version (baseline Logistic Regression pipeline)

## Setup
```bash
pip install pandas numpy scikit-learn matplotlib openpyxl
```

## Run
Notebook:
- Open `Telco_Churn_Case_Study.ipynb` and run all cells

Script:
```bash
python churn_analysis.py
```

## Notes
- Model uses `class_weight="balanced"` because churn is imbalanced (~26.5%).
- Replace threshold (0.5) based on business trade-offs (cost of outreach vs cost of churn).
