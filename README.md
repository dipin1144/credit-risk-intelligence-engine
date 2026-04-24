# Credit Risk Intelligence Engine
> Production-grade credit underwriting system with Explainable AI, Fairness Auditing, and EU AI Act compliance

[![Python](https://img.shields.io/badge/Python-3.13-blue)](https://python.org)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.1-green)](https://lightgbm.readthedocs.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28-FF4B4B)](https://streamlit.io)

---

## Live Demo
- **Streamlit Dashboard:** [Credit Risk Underwriter Desk](https://your-streamlit-url.streamlit.app)
- **FastAPI Docs:** `http://localhost:8000/docs`

---

## Overview

A production-style credit risk scoring system modeled on how fintechs like **Revolut, N26, Adyen, and Wise** approach credit underwriting. Goes far beyond standard Kaggle notebooks by implementing the full ML lifecycle:### What Makes This Unique

| Feature | Detail |
|---|---|
| **Multi-table Feature Engineering** | 181 features engineered from 7 related tables (56M+ rows) |
| **Expected Loss Framework** | Outputs PD × LGD × EAD in euros — not just a probability |
| **Adverse Action Codes** | Legally-formatted decline reasons (GDPR Article 22) |
| **Fairness Audit** | Disparate impact analysis across gender and age groups |
| **Risk-Based Pricing** | 6 APR tiers from 7.9% to 21.9% based on PD score |
| **EU AI Act Compliance** | Model card + explainability + fairness documentation |

---

## Architecture---

## Dataset

**Home Credit Default Risk** (Kaggle)
- 307,511 loan applications
- 7 related tables (bureau, installments, credit card balances)
- 56M+ rows processed during feature engineering
- 8.07% default rate (realistic consumer credit)

---

## Feature Engineering

Starting from 122 raw features, engineered **181 features** across 7 tables:

| Table | Features Created | Key Signals |
|---|---|---|
| `application_train` | 17 | Credit/income ratio, EXT_SOURCE combined, pensioner flag |
| `bureau` + `bureau_balance` | 13 | DPD ratio, utilization, active loan count |
| `previous_application` | 12 | Refusal rate, approval history |
| `installments_payments` | 9 | Late payment rate, days past due |
| `credit_card_balance` | 10 | CC utilization, payment ratio |

**Key insight:** 7 of the top 10 predictive features were engineered from supplementary tables — not available in the main application file.

---

## Model Performance

| Metric | Value |
|---|---|
| **Gini Coefficient** | 0.5717 |
| **AUC-ROC** | 0.7858 |
| **CV Strategy** | 5-fold Stratified |
| **Tuning** | Optuna (50 trials) |
| **Algorithm** | LightGBM |

### Top 10 Features
1. `CREDIT_TERM_MONTHS` — Loan duration
2. `PREV_ANNUITY_MEAN` — Past repayment burden
3. `EXT_SOURCE_3` — Bureau score proxy
4. `BUREAU_CREDIT_MAX` — Maximum credit exposure
5. `GOODS_CREDIT_RATIO` — LTV equivalent
6. `PREV_CREDIT_SUM` — Total borrowing history
7. `EXT_SOURCE_MEAN` — Combined bureau score
8. `INST_COUNT` — Number of installment payments
9. `INST_LATE_RATE` — Late payment frequency
10. `EXT_SOURCE_1` — Third bureau score signal

---

## Explainability (SHAP)

Every credit decision comes with:
- **Global SHAP summary** — which features drive the model
- **Local waterfall plot** — why THIS applicant was scored
- **Adverse action codes** — legally formatted decline reasons

### Sample Credit Decision Notice---

## Fairness Audit

| Group | Default Rate | Disparate Impact |
|---|---|---|
| Female | 6.90% | — |
| Male | 9.90% | 0.69 (below 0.80 threshold) |
| Age 18-25 | 12.30% | Highest risk segment |
| Age 55+ | 5.22% | Lowest risk segment |

**Finding:** Gender disparate impact ratio of 0.69 falls below the 0.80 legal threshold → mitigation recommended (remove gender from feature set).

---

## API Endpoint

```bash
# Score an application
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{
    "application_id": "APP-001",
    "amt_credit": 500000,
    "amt_income_total": 150000,
    "amt_annuity": 25000,
    "days_birth": -12775,
    "days_employed": -2000,
    "ext_source_2": 0.6
  }'
```

**Response:**
```json
{
  "application_id": "APP-001",
  "pd_score": 0.0661,
  "decision": "APPROVED",
  "risk_tier": "A2",
  "recommended_rate_apr": 10.9,
  "adverse_action_codes": [],
  "expected_loss_eur": 14874.75,
  "explanation": "PD Score 0.066 → Risk Tier A2 → Approved at 10.9% APR"
}
```

---

## Risk-Based Pricing

| Risk Tier | PD Range | APR |
|---|---|---|
| A1 | < 5% | 7.9% |
| A2 | 5–10% | 10.9% |
| B1 | 10–15% | 13.9% |
| B2 | 15–20% | 16.9% |
| C1 | 20–30% | 21.9% |
| C2 | > 30% | Declined |

---

## Setup & Installation

```bash
# Clone repo
git clone https://github.com/dipin1144/credit-risk-intelligence-engine.git
cd credit-risk-intelligence-engine

# Create virtual environment
python -m venv venv
source venv/Scripts/activate  # Windows
source venv/bin/activate       # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Download data
# Place Home Credit CSV files in data/raw/
# kaggle competitions download -c home-credit-default-risk

# Run FastAPI
python -m uvicorn api.main:app --reload --port 8000

# Run Streamlit
streamlit run dashboard/app.py
```

---

## Regulatory Compliance

| Regulation | Status |
|---|---|
| GDPR Article 22 | Compliant — adverse action explanations provided |
| EU AI Act Article 13 | Compliant — model card + transparency documentation |
| Equal Credit Opportunity | Monitored — gender mitigation recommended |
| SR 11-7 Model Risk | Addressed — CV validation + feature importance |

---

## Tech Stack

`Python 3.13` · `LightGBM` · `SHAP` · `Optuna` · `FastAPI` · `Streamlit` · `Plotly` · `Pandas` · `Scikit-learn` · `Fairlearn`

---

## Author

**Dipin** — ML Engineer targeting fintech credit risk roles
- GitHub: [@dipin1144](https://github.com/dipin1144)
- Project: [Credit Risk Intelligence Engine](https://credit-risk-intelligence-engine.streamlit.app)