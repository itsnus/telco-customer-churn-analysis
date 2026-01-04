"""Telco churn end-to-end pipeline (clean → EDA → feature engineering → modeling).

Run:
    python churn_analysis.py

Outputs:
    - prints key metrics
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score, accuracy_score


DATA_PATH = "WA_Fn-UseC_-Telco-Customer-Churn 2.xlsx"

def yes_no_to_binary(s: pd.Series) -> pd.Series:
    return s.replace({"Yes":1,"No":0,"No phone service":0,"No internet service":0})

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[-1, 0, 12, 24, 48, 72],
        labels=["0","1-12","13-24","25-48","49-72"]
    )

    df["avg_monthly_spend"] = df["TotalCharges"] / (df["tenure"].replace(0, np.nan))
    df["avg_monthly_spend"] = df["avg_monthly_spend"].fillna(df["MonthlyCharges"])

    service_cols = [
        "PhoneService","MultipleLines","OnlineSecurity","OnlineBackup","DeviceProtection",
        "TechSupport","StreamingTV","StreamingMovies"
    ]
    for c in service_cols:
        df[c+"_bin"] = yes_no_to_binary(df[c])

    df["services_count"] = df[[c+"_bin" for c in service_cols]].sum(axis=1)

    return df

def summarize(y_true, y_pred, y_proba):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

def main():
    df = pd.read_excel(DATA_PATH)
    df = build_features(df)

    y = df["Churn"].map({"Yes":1,"No":0})
    X = df.drop(columns=["Churn"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in X_train.columns if c not in numeric_features]

    preprocess = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_features),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                              ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_features),
        ]
    )

    model = Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", LogisticRegression(max_iter=2000, class_weight="balanced"))
    ])

    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:,1]
    pred = (proba>=0.5).astype(int)

    print("Churn rate:", float(y.mean()))
    print("Metrics:", summarize(y_test, pred, proba))

if __name__ == "__main__":
    main()
