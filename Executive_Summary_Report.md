# Executive Summary — Telco Churn (Case Study)

## Business problem
Churn = customers who cancel service (`Churn = Yes`). Reducing churn protects recurring revenue and lowers acquisition replacement costs.

## Data quality
- `TotalCharges` contains **11 missing** values (blank strings in raw data) → converted to numeric and imputed (median).

## Key insights (EDA)
- **Month-to-month** contracts churn far more than longer contracts.
- **Fiber optic** customers churn more than DSL and “No internet” groups.
- **Electronic check** has the highest churn among payment methods.
- Churn is highest in the **first 12 months** (early tenure risk).
- Customers without **OnlineSecurity** / **TechSupport** churn more.

## Modelling results (Logistic Regression baseline)
- ROC-AUC ≈ **0.84** (good ranking power)
- Confusion matrix highlights a trade-off:
  - More churners caught (higher recall)
  - Some false alarms (FP) — can be managed with threshold tuning

## Postdictive findings
- Most misses (FN) happen at probabilities near the decision boundary → use threshold tuning and add richer features (e.g., support interactions if available).

## Recommendations
1. Convert month-to-month to annual via targeted offers at months 2–3 and 10–12.
2. Improve first-90-day onboarding + proactive outreach (setup, billing, support).
3. Bundle security/support for at-risk fiber customers; test price/value messaging.
4. Encourage auto-pay; investigate electronic-check friction and failure points.
5. Operate a weekly “Top-risk customer list” from churn scores and track retention lift.
