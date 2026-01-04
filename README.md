# Telco Customer Churn — End-to-End Case Study

## What’s inside
- `Telco_Churn_Case_Study.ipynb` — narrative notebook (problem → cleaning → EDA → feature engineering → modelling → recommendations)
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
