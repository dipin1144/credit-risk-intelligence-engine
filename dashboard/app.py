# dashboard/app.py — Standalone version for Streamlit Cloud deployment

import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import os
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title = "Credit Risk Intelligence Engine",
    page_icon  = "🏦",
    layout     = "wide"
)

# ── Load model directly (no API needed) ───────────────────────
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 
                              '..', 'data', 'processed', 'lgb_model.pkl')
    with open(model_path, 'rb') as f:
        return pickle.load(f)

model = load_model()

# ── Risk pricing ───────────────────────────────────────────────
def get_risk_tier_and_rate(pd_score):
    if pd_score < 0.05:   return 'A1', 7.9
    elif pd_score < 0.10: return 'A2', 10.9
    elif pd_score < 0.15: return 'B1', 13.9
    elif pd_score < 0.20: return 'B2', 16.9
    elif pd_score < 0.30: return 'C1', 21.9
    else:                  return 'C2', None

# ── Score function ─────────────────────────────────────────────
def score_application(amt_credit, amt_income, amt_annuity, amt_goods,
                       age_years, emp_years, ext1, ext2, ext3,
                       bureau_util, late_rate, refusal_rate):

    ext_vals = [ext1, ext2, ext3]
    ext_mean = np.mean(ext_vals)

    features = {
        'AMT_CREDIT':            amt_credit,
        'AMT_INCOME_TOTAL':      amt_income,
        'AMT_ANNUITY':           amt_annuity,
        'AMT_GOODS_PRICE':       amt_goods,
        'DAYS_BIRTH':            int(-age_years * 365),
        'DAYS_EMPLOYED':         int(-emp_years * 365),
        'EXT_SOURCE_1':          ext1,
        'EXT_SOURCE_2':          ext2,
        'EXT_SOURCE_3':          ext3,
        'EXT_SOURCE_MEAN':       ext_mean,
        'EXT_SOURCE_MIN':        min(ext_vals),
        'EXT_SOURCE_STD':        np.std(ext_vals),
        'EXT_SOURCE_PRODUCT':    np.prod(ext_vals),
        'CREDIT_INCOME_RATIO':   amt_credit / (amt_income + 1),
        'ANNUITY_INCOME_RATIO':  amt_annuity / (amt_income + 1),
        'CREDIT_TERM_MONTHS':    amt_credit / (amt_annuity + 1),
        'GOODS_CREDIT_RATIO':    amt_goods / (amt_credit + 1),
        'AGE_YEARS':             age_years,
        'EMPLOYMENT_YEARS':      emp_years,
        'IS_PENSIONER':          0,
        'BUREAU_UTILIZATION':    bureau_util,
        'INST_LATE_RATE':        late_rate,
        'PREV_REFUSAL_RATE':     refusal_rate,
    }

    features_df = pd.DataFrame([features])
    model_features = model.feature_name_
    for col in model_features:
        if col not in features_df.columns:
            features_df[col] = 0
    features_df = features_df[model_features]

    pd_score   = float(model.predict_proba(features_df)[0][1])
    risk_tier, rate = get_risk_tier_and_rate(pd_score)
    decision   = "DECLINED" if rate is None else "APPROVED"
    exp_loss   = pd_score * 0.45 * amt_credit

    # Adverse action codes
    adverse = []
    if decision == "DECLINED":
        if ext_mean < 0.4:      adverse.append(("[01]", "Insufficient credit bureau score"))
        if late_rate > 0.1:     adverse.append(("[04]", "History of late installment payments"))
        if bureau_util > 0.7:   adverse.append(("[07]", "High credit utilization ratio"))
        if amt_credit/amt_income > 5: adverse.append(("[09]", "Debt-to-income ratio too high"))
        if refusal_rate > 0.3:  adverse.append(("[11]", "Previous credit applications declined"))

    return pd_score, decision, risk_tier, rate, exp_loss, adverse

# ── Header ─────────────────────────────────────────────────────
st.title("🏦 Credit Risk Intelligence Engine")
st.markdown("*Production-grade underwriting system — Powered by LightGBM + SHAP*")
st.divider()

# ── Sidebar ────────────────────────────────────────────────────
st.sidebar.header("Applicant Details")
application_id = st.sidebar.text_input("Application ID", value="APP-001")

st.sidebar.subheader("Financial Profile")
amt_income  = st.sidebar.number_input("Annual Income (€)",    10000,  10000000, 150000, 5000)
amt_credit  = st.sidebar.number_input("Loan Amount (€)",      10000,  5000000,  500000, 10000)
amt_annuity = st.sidebar.number_input("Monthly Annuity (€)",  1000,   500000,   25000,  1000)
amt_goods   = st.sidebar.number_input("Goods Price (€)",      10000,  5000000,  450000, 10000)

st.sidebar.subheader("Personal Details")
age_years = st.sidebar.slider("Age (years)",         18, 70, 35)
emp_years = st.sidebar.slider("Employment (years)",   0, 40,  5)

st.sidebar.subheader("Bureau Scores (0-1)")
ext1 = st.sidebar.slider("Bureau Score 1", 0.0, 1.0, 0.5, 0.01)
ext2 = st.sidebar.slider("Bureau Score 2", 0.0, 1.0, 0.5, 0.01)
ext3 = st.sidebar.slider("Bureau Score 3", 0.0, 1.0, 0.5, 0.01)

st.sidebar.subheader("Risk Indicators")
bureau_util  = st.sidebar.slider("Credit Utilization",     0.0, 1.0, 0.3, 0.01)
late_rate    = st.sidebar.slider("Late Payment Rate",       0.0, 1.0, 0.0, 0.01)
refusal_rate = st.sidebar.slider("Previous Refusal Rate",  0.0, 1.0, 0.0, 0.01)

score_btn = st.sidebar.button("SCORE APPLICATION", 
                               type="primary", 
                               use_container_width=True)

# ── Main Panel ─────────────────────────────────────────────────
if score_btn:
    pd_score, decision, risk_tier, rate, exp_loss, adverse = score_application(
        amt_credit, amt_income, amt_annuity, amt_goods,
        age_years, emp_years, ext1, ext2, ext3,
        bureau_util, late_rate, refusal_rate
    )

    # Row 1: Key metrics
    col1, col2, col3, col4 = st.columns(4)
    decision_icon = "🟢" if decision == "APPROVED" else "🔴"
    col1.metric("Decision",       f"{decision_icon} {decision}")
    col2.metric("PD Score",       f"{pd_score:.4f}")
    col3.metric("Risk Tier",      risk_tier)
    col4.metric("Expected Loss",  f"€{exp_loss:,.0f}")

    st.divider()

    # Row 2: Gauge + Summary
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Risk Score Gauge")
        fig = go.Figure(go.Indicator(
            mode  = "gauge+number",
            value = pd_score * 100,
            title = {'text': "Probability of Default (%)"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar':  {'color': "darkblue"},
                'steps': [
                    {'range': [0,  10], 'color': "#16A34A"},
                    {'range': [10, 20], 'color': "#65A30D"},
                    {'range': [20, 30], 'color': "#D97706"},
                    {'range': [30, 50], 'color': "#DC2626"},
                    {'range': [50,100], 'color': "#7F1D1D"},
                ],
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Credit Decision Summary")
        if decision == "APPROVED":
            st.success(f"APPROVED — Risk Tier {risk_tier}")
            st.info(f"Recommended APR: **{rate}%**")
            monthly = (amt_credit * (rate/100/12)) / \
                      (1 - (1 + rate/100/12)**-60)
            st.metric("Est. Monthly Payment", f"€{monthly:,.0f}")
        else:
            st.error("APPLICATION DECLINED")
            st.warning("Risk score exceeds acceptable threshold")

        st.metric("Expected Loss (LGD=45%)", f"€{exp_loss:,.2f}")
        st.caption(f"PD Score {pd_score:.3f} → Risk Tier {risk_tier} → "
                   f"{'Approved at ' + str(rate) + '% APR' if decision == 'APPROVED' else 'Declined'}")

    st.divider()

    # Row 3: Adverse + Summary
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Adverse Action Codes")
        if adverse:
            for code, desc in adverse:
                st.error(f"{code} {desc}")
            st.caption("Rights: GDPR Article 22 review available")
        else:
            st.success("No adverse action factors identified")

    with col_b:
        st.subheader("Application Summary")
        data = {
            "Annual Income":     f"€{amt_income:,}",
            "Loan Amount":       f"€{amt_credit:,}",
            "Credit/Income":     f"{amt_credit/amt_income:.1f}x",
            "Bureau Score Avg":  f"{(ext1+ext2+ext3)/3:.3f}",
            "Utilization":       f"{bureau_util*100:.0f}%",
            "Late Payment Rate": f"{late_rate*100:.0f}%",
        }
        for k, v in data.items():
            c1, c2 = st.columns(2)
            c1.write(f"**{k}**")
            c2.write(v)

else:
    st.info("Configure applicant details in the sidebar and click SCORE APPLICATION")
    c1, c2, c3 = st.columns(3)
    c1.metric("Model",         "LightGBM")
    c2.metric("Gini Score",    "0.5717")
    c3.metric("Training Size", "307,511")

    st.subheader("System Capabilities")
    caps = {
        "Probability of Default":  "LightGBM — 5-fold CV, Optuna tuned",
        "Explainability":          "SHAP values + Adverse Action Codes",
        "Fairness Audit":          "Gender + Age disparate impact analysis",
        "Risk-Based Pricing":      "6 tiers from 7.9% to 21.9% APR",
        "Expected Loss":           "PD × LGD × EAD framework",
        "Regulatory Compliance":   "GDPR Art.22 + EU AI Act Art.13",
    }
    for k, v in caps.items():
        st.write(f"**{k}:** {v}")