"""
Customer Credit Risk Predictor - Training Script
---------------------------------------------------
Ye script customer/udhaar (credit) data se ek Logistic Regression model train
karta hai jo predict karega ke customer time pe payment karega ya nahi (risk score).

USAGE:
1. Apna customer data CSV mein daalein: customer_data.csv
   Required columns:
     - customer_id        (unique identifier)
     - total_purchases     (total kitni dafa khareeda - count)
     - total_credit_amount (total kitna udhaar diya gaya - Rs)
     - total_paid_amount   (total kitna wapas mila - Rs)
     - avg_days_to_pay     (average kitne din mein payment karta hai)
     - max_days_overdue    (sab se zyada kitne din late hua)
     - defaulted           (0 = time pe paid / 1 = default kiya - ye TARGET column hai)

   Agar aapke data mein columns ke naam different hain, mujhe bata dein,
   main script adjust kar dunga.

2. Run karein: python train_model.py

3. Output: credit_risk_model.pkl (backend folder mein)

Agar sales_data.csv nahi milta to ye dummy data generate kar dega taake
pipeline test ho sake.
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

DATA_PATH = "customer_data.csv"
MODEL_PATH = "credit_risk_model.pkl"

FEATURES = [
    "total_purchases",
    "total_credit_amount",
    "total_paid_amount",
    "avg_days_to_pay",
    "max_days_overdue",
    "credit_utilization"  # engineered feature
]
TARGET = "defaulted"


def generate_dummy_data(n=500):
    """Agar customer_data.csv nahi mila to sample data generate karta hai."""
    print("⚠️  customer_data.csv nahi mila — dummy data generate kar raha hoon...")
    np.random.seed(42)

    rows = []
    for i in range(n):
        total_purchases = np.random.randint(5, 200)
        total_credit_amount = round(np.random.uniform(2000, 100000), 2)
        payment_ratio = np.random.beta(5, 2)  # zyada log achi payment history rakhtay hain
        total_paid_amount = round(total_credit_amount * payment_ratio, 2)
        avg_days_to_pay = np.random.randint(1, 60)
        max_days_overdue = np.random.randint(0, 90)

        # default risk higher agar payment ratio kam ho aur overdue zyada ho
        risk_score = (1 - payment_ratio) * 0.6 + (max_days_overdue / 90) * 0.4
        defaulted = 1 if risk_score > 0.55 or np.random.rand() < 0.05 else 0

        rows.append({
            "customer_id": f"CUST{1000+i}",
            "total_purchases": total_purchases,
            "total_credit_amount": total_credit_amount,
            "total_paid_amount": total_paid_amount,
            "avg_days_to_pay": avg_days_to_pay,
            "max_days_overdue": max_days_overdue,
            "defaulted": defaulted
        })

    df = pd.DataFrame(rows)
    df.to_csv(DATA_PATH, index=False)
    print(f"✅ Dummy data saved to {DATA_PATH} ({len(df)} rows)")
    return df


def load_data():
    if os.path.exists(DATA_PATH):
        print(f"📂 Loading data from {DATA_PATH}")
        df = pd.read_csv(DATA_PATH)
    else:
        df = generate_dummy_data()

    missing = [c for c in ["total_purchases", "total_credit_amount",
                            "total_paid_amount", "avg_days_to_pay",
                            "max_days_overdue", TARGET] if c not in df.columns]
    if missing:
        raise ValueError(
            f"❌ Ye columns CSV mein missing hain: {missing}\n"
            f"   Available columns: {list(df.columns)}\n"
            f"   Please column names match karein ya mujhe batayein taake main script adjust kar doon."
        )

    # Engineered feature: kitna % credit wapas aaya
    df["credit_utilization"] = df["total_paid_amount"] / df["total_credit_amount"].replace(0, 1)

    return df


def train():
    df = load_data()

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("🚀 Training Logistic Regression...")

    # Logistic Regression is sensitive to feature scale, isliye scaling zaroori hai
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(
        class_weight="balanced",  # imbalanced data (kam log default karte hain)
        max_iter=1000,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)

    preds = model.predict(X_test_scaled)
    probs = model.predict_proba(X_test_scaled)[:, 1]

    print("\n📊 Classification Report:")
    print(classification_report(y_test, preds, target_names=["On-time", "Default Risk"]))
    print(f"📊 ROC-AUC Score: {roc_auc_score(y_test, probs):.4f}")

    print("\n📈 Feature Coefficients (higher = zyada risk badhata hai):")
    for feat, coef in sorted(zip(FEATURES, model.coef_[0]), key=lambda x: -abs(x[1])):
        direction = "increases risk" if coef > 0 else "decreases risk"
        print(f"   {feat}: {coef:+.3f}  ({direction})")

    joblib.dump({"model": model, "scaler": scaler, "features": FEATURES}, MODEL_PATH)
    print(f"\n✅ Model saved to {MODEL_PATH}")
    print("Ab backend/app.py run karein: uvicorn app:app --reload")


if __name__ == "__main__":
    train()
