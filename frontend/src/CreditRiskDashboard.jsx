import React, { useState } from "react";
import modelData from "./modelData.json";
import "./CreditRiskDashboard.css";

const { coef, intercept, scaler_mean, scaler_scale, features } = modelData;

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

// Logistic Regression prediction, replicating scikit-learn's StandardScaler + predict_proba
function predictRisk(rawFeatures) {
  // Standardize: (x - mean) / scale
  const scaled = rawFeatures.map((v, i) => (v - scaler_mean[i]) / scaler_scale[i]);
  let z = intercept;
  for (let i = 0; i < coef.length; i++) {
    z += coef[i] * scaled[i];
  }
  return sigmoid(z);
}

function riskLabel(prob) {
  if (prob >= 0.6) return "High Risk";
  if (prob >= 0.3) return "Medium Risk";
  return "Low Risk";
}

function riskColor(label) {
  if (label === "High Risk") return "#ff5470";
  if (label === "Medium Risk") return "#ff9d00";
  return "#00e5ff";
}

export default function CreditRiskDashboard() {
  const [form, setForm] = useState({
    customer_id: "NEW-CUSTOMER",
    total_purchases: 30,
    total_credit_amount: 15000,
    total_paid_amount: 12000,
    avg_days_to_pay: 15,
    max_days_overdue: 10
  });

  const creditUtilization = form.total_paid_amount / (form.total_credit_amount || 1);
  const featureValues = [
    form.total_purchases,
    form.total_credit_amount,
    form.total_paid_amount,
    form.avg_days_to_pay,
    form.max_days_overdue,
    creditUtilization
  ];
  const prob = predictRisk(featureValues);
  const label = riskLabel(prob);
  const color = riskColor(label);

  const handleChange = (field, value) => {
    setForm(prev => ({ ...prev, [field]: value }));
  };

  return (
    <div className="cr-dashboard">
      <div className="cr-bg-orb cr-orb-1"></div>
      <div className="cr-bg-orb cr-orb-2"></div>

      <div className="cr-header">
        <h1>💳 Customer Credit Risk Dashboard</h1>
        <p className="cr-subtitle">AI-powered udhaar/default risk prediction — runs entirely in your browser</p>
      </div>

      <div className="cr-layout">
        <div className="cr-glass-card">
          <h2>Customer Details</h2>
          <div className="cr-form">
            <label>Total Purchases (count)</label>
            <input type="number" value={form.total_purchases}
              onChange={e => handleChange("total_purchases", Number(e.target.value))} />

            <label>Total Credit Amount (Rs)</label>
            <input type="number" value={form.total_credit_amount}
              onChange={e => handleChange("total_credit_amount", Number(e.target.value))} />

            <label>Total Paid Amount (Rs)</label>
            <input type="number" value={form.total_paid_amount}
              onChange={e => handleChange("total_paid_amount", Number(e.target.value))} />

            <label>Average Days to Pay</label>
            <input type="number" value={form.avg_days_to_pay}
              onChange={e => handleChange("avg_days_to_pay", Number(e.target.value))} />

            <label>Max Days Overdue</label>
            <input type="number" value={form.max_days_overdue}
              onChange={e => handleChange("max_days_overdue", Number(e.target.value))} />
          </div>
        </div>

        <div className="cr-glass-card cr-result-card">
          <h2>Risk Prediction</h2>
          <div className="cr-result-display">
            <div className="cr-gauge-wrap">
              <div className="cr-gauge-bg">
                <div
                  className="cr-gauge-fill"
                  style={{ width: `${prob * 100}%`, background: color }}
                ></div>
              </div>
              <span className="cr-gauge-text">{(prob * 100).toFixed(1)}%</span>
            </div>
            <span
              className="cr-badge-large"
              style={{
                background: `${color}22`,
                color: color,
                border: `1px solid ${color}55`
              }}
            >
              {label}
            </span>
            <p className="cr-explain">
              Credit Utilization: {(creditUtilization * 100).toFixed(1)}%
              (total paid / total credit given)
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
