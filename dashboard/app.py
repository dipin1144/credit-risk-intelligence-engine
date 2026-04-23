# dashboard/app.py — Credit Risk Intelligence Engine
# Underwriter Desk UI

import streamlit as st
import requests
import plotly.graph_objects as go
import pandas as pd

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title = "Credit Risk Intelligence Engine",
    page_icon  = "🏦",
    layout     = "wide"
)

# ── Header ─────────────────────────────────────────────────────
st.title("🏦 Credit Risk Intelligence Engine")
st.markdown("*Production-grade underwriting system — Powered by LightGBM + SHAP*")
st.divider()

# ── Sidebar — Applicant Input ──────────────────────────────────
st.sidebar.header("Applicant Details")

application_id = st.sidebar.text_input("Application ID", value="APP-001")

st.sidebar.subheader("Financial Profile")
amt_income   = st.sidebar.number_input("Annual Income (€)", 
                min_value=10000, max_value=10000000, value=150000, step=5000)
amt_credit   = st.sidebar.number_input("Loan Amount (€)",  
                min_value=10000, max_value=5000000,  value=500000, step=10000)
amt_annuity  = st.sidebar.number_input("Monthly Annuity (€)", 
                min_value=1000,  max_value=500000,   value=25000,  step=1000)
amt_goods    = st.sidebar.number_input("Goods Price (€)",  
                min_value=10000, max_value=5000000,  value=450000, step=10000)

st.sidebar.subheader("Personal Details")
age_years    = st.sidebar.slider("Age (years)", 18, 70, 35)
emp_years    = st.sidebar.slider("Employment (years)", 0, 40, 5)

st.sidebar.subheader("Credit Bureau Scores (0-1)")
ext1 = st.sidebar.slider("Bureau Score 1", 0.0, 1.0, 0.5, 0.01)
ext2 = st.sidebar.slider("Bureau Score 2", 0.0, 1.0, 0.5, 0.01)
ext3 = st.sidebar.slider("Bureau Score 3", 0.0, 1.0, 0.5, 0.01)

st.sidebar.subheader("Risk Indicators")
bureau_util  = st.sidebar.slider("Credit Utilization", 0.0, 1.0, 0.3, 0.01)
late_rate    = st.sidebar.slider("Late Payment Rate",  0.0, 1.0, 0.0, 0.01)
refusal_rate = st.sidebar.slider("Previous Refusal Rate", 0.0, 1.0, 0.0, 0.01)

score_btn = st.sidebar.button("SCORE APPLICATION", type="primary", 
                               use_container_width=True)

# ── Main Panel ─────────────────────────────────────────────────
if score_btn:
    payload = {
        "application_id":    application_id,
        "amt_credit":        amt_credit,
        "amt_income_total":  amt_income,
        "amt_annuity":       amt_annuity,
        "amt_goods_price":   amt_goods,
        "days_birth":        int(-age_years * 365),
        "days_employed":     int(-emp_years * 365),
        "ext_source_1":      ext1,
        "ext_source_2":      ext2,
        "ext_source_3":      ext3,
        "bureau_utilization": bureau_util,
        "inst_late_rate":    late_rate,
        "prev_refusal_rate": refusal_rate
    }

    with st.spinner("Scoring application..."):
        try:
            resp = requests.post(
                "http://127.0.0.1:8000/score",
                json=payload, timeout=10
            )
            result = resp.json()

            # ── Row 1: Key Metrics ─────────────────────────────
            col1, col2, col3, col4 = st.columns(4)

            decision_color = "🟢" if result['decision'] == 'APPROVED' else "🔴"
            col1.metric("Decision", 
                        f"{decision_color} {result['decision']}")
            col2.metric("PD Score",  
                        f"{result['pd_score']:.4f}",
                        delta=f"{'Low Risk' if result['pd_score'] < 0.15 else 'High Risk'}",
                        delta_color="inverse")
            col3.metric("Risk Tier", result['risk_tier'])
            col4.metric("Expected Loss",
                        f"€{result['expected_loss_eur']:,.0f}")

            st.divider()

            # ── Row 2: Gauge + Details ─────────────────────────
            col_left, col_right = st.columns([1, 1])

            with col_left:
                st.subheader("Risk Score Gauge")
                fig = go.Figure(go.Indicator(
                    mode  = "gauge+number",
                    value = result['pd_score'] * 100,
                    title = {'text': "Probability of Default (%)"},
                    gauge = {
                        'axis': {'range': [0, 100]},
                        'bar':  {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 10],  'color': "#16A34A"},
                            {'range': [10, 20], 'color': "#65A30D"},
                            {'range': [20, 30], 'color': "#D97706"},
                            {'range': [30, 50], 'color': "#DC2626"},
                            {'range': [50, 100],'color': "#7F1D1D"},
                        ],
                        'threshold': {
                            'line':  {'color': "black", 'width': 4},
                            'value': result['pd_score'] * 100
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

            with col_right:
                st.subheader("Credit Decision Summary")
                rate = result['recommended_rate_apr']

                if result['decision'] == 'APPROVED':
                    st.success(f"APPROVED — Risk Tier {result['risk_tier']}")
                    st.info(f"Recommended APR: **{rate}%**")
                    monthly = (amt_credit * (rate/100/12)) / \
                              (1 - (1 + rate/100/12)**-60)
                    st.metric("Est. Monthly Payment", f"€{monthly:,.0f}")
                else:
                    st.error("APPLICATION DECLINED")
                    st.warning("Risk score exceeds acceptable threshold")

                st.metric("Expected Loss (LGD=45%)", 
                          f"€{result['expected_loss_eur']:,.2f}")
                st.caption(result['explanation'])

            st.divider()

            # ── Row 3: Adverse Actions + Risk Factors ──────────
            col_a, col_b = st.columns(2)

            with col_a:
                st.subheader("Adverse Action Codes")
                if result['adverse_action_codes']:
                    for item in result['adverse_action_codes']:
                        st.error(f"[{item['code']}] {item['description']}")
                    st.caption("Rights: GDPR Article 22 review available")
                else:
                    st.success("No adverse action factors identified")

            with col_b:
                st.subheader("Application Summary")
                summary = {
                    "Annual Income":    f"€{amt_income:,}",
                    "Loan Amount":      f"€{amt_credit:,}",
                    "Credit/Income":    f"{amt_credit/amt_income:.1f}x",
                    "Bureau Score Avg": f"{(ext1+ext2+ext3)/3:.3f}",
                    "Utilization":      f"{bureau_util*100:.0f}%",
                    "Late Payment Rate":f"{late_rate*100:.0f}%",
                }
                for k, v in summary.items():
                    col_x, col_y = st.columns(2)
                    col_x.write(f"**{k}**")
                    col_y.write(v)

        except Exception as e:
            st.error(f"API Error: {e}")
            st.info("Make sure the FastAPI server is running on port 8000")

else:
    # Default state
    st.info("Configure applicant details in the sidebar and click SCORE APPLICATION")
    col1, col2, col3 = st.columns(3)
    col1.metric("Model", "LightGBM")
    col2.metric("Gini Score", "0.5717")
    col3.metric("Training Size", "307,511")

    st.subheader("System Capabilities")
    caps = {
        "Probability of Default (PD)": "LightGBM — 5-fold CV, Optuna tuned",
        "Explainability":              "SHAP values + Adverse Action Codes",
        "Fairness Audit":              "Gender + Age disparate impact analysis",
        "Risk-Based Pricing":          "6 tiers from 7.9% to 21.9% APR",
        "Expected Loss":               "PD × LGD × EAD framework",
        "Regulatory Compliance":       "GDPR Art.22 + EU AI Act Art.13",
    }
    for k, v in caps.items():
        st.write(f"**{k}:** {v}")