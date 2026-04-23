# api/main.py — Credit Risk Intelligence Engine
# Production-grade FastAPI scoring endpoint

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import pickle
import numpy as np
import pandas as pd
import os

app = FastAPI(
    title       = "Credit Risk Intelligence Engine",
    description = "Production credit scoring API with explainable AI",
    version     = "1.0.0"
)

# ── Load model on startup ──────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), 
                          '..', 'data', 'processed', 'lgb_model.pkl')
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

# ── Adverse action codes ───────────────────────────────────────
ADVERSE_CODES = {
    'EXT_SOURCE_MEAN':      ('01', 'Insufficient credit bureau score'),
    'EXT_SOURCE_2':         ('02', 'Low external risk assessment'),
    'INST_LATE_RATE':       ('04', 'History of late payments'),
    'INST_DPD_MEAN':        ('05', 'Excessive days past due'),
    'BUREAU_UTILIZATION':   ('07', 'High credit utilization'),
    'CREDIT_INCOME_RATIO':  ('09', 'Debt-to-income ratio too high'),
    'PREV_REFUSAL_RATE':    ('11', 'Previous applications declined'),
    'ANNUITY_INCOME_RATIO': ('13', 'Monthly obligations exceed threshold'),
}

# ── Risk-based pricing ─────────────────────────────────────────
def get_risk_tier_and_rate(pd_score: float):
    if pd_score < 0.05:
        return 'A1', 7.9
    elif pd_score < 0.10:
        return 'A2', 10.9
    elif pd_score < 0.15:
        return 'B1', 13.9
    elif pd_score < 0.20:
        return 'B2', 16.9
    elif pd_score < 0.30:
        return 'C1', 21.9
    else:
        return 'C2', None  # Decline

# ── Request schema ─────────────────────────────────────────────
class ApplicationRequest(BaseModel):
    application_id:       str
    amt_credit:           float
    amt_income_total:     float
    amt_annuity:          float
    amt_goods_price:      Optional[float] = None
    days_birth:           int
    days_employed:        int
    ext_source_1:         Optional[float] = None
    ext_source_2:         Optional[float] = None
    ext_source_3:         Optional[float] = None
    name_education_type:  Optional[str]   = "Secondary / secondary special"
    name_income_type:     Optional[str]   = "Working"
    bureau_utilization:   Optional[float] = 0.3
    inst_late_rate:       Optional[float] = 0.0
    prev_refusal_rate:    Optional[float] = 0.0

# ── Response schema ────────────────────────────────────────────
class ScoringResponse(BaseModel):
    application_id:       str
    pd_score:             float
    decision:             str
    risk_tier:            str
    recommended_rate_apr: Optional[float]
    adverse_action_codes: List[dict]
    expected_loss_eur:    float
    explanation:          str

# ── Feature engineering ────────────────────────────────────────
def build_features(req: ApplicationRequest) -> pd.DataFrame:
    ext_vals = [v for v in [req.ext_source_1, 
                             req.ext_source_2, 
                             req.ext_source_3] if v is not None]
    ext_mean = np.mean(ext_vals) if ext_vals else 0.5

    features = {
        'AMT_CREDIT':           req.amt_credit,
        'AMT_INCOME_TOTAL':     req.amt_income_total,
        'AMT_ANNUITY':          req.amt_annuity,
        'AMT_GOODS_PRICE':      req.amt_goods_price or req.amt_credit * 0.9,
        'DAYS_BIRTH':           req.days_birth,
        'DAYS_EMPLOYED':        req.days_employed,
        'EXT_SOURCE_1':         req.ext_source_1 or ext_mean,
        'EXT_SOURCE_2':         req.ext_source_2 or ext_mean,
        'EXT_SOURCE_3':         req.ext_source_3 or ext_mean,
        'EXT_SOURCE_MEAN':      ext_mean,
        'EXT_SOURCE_MIN':       min(ext_vals) if ext_vals else 0.5,
        'EXT_SOURCE_STD':       np.std(ext_vals) if len(ext_vals)>1 else 0,
        'EXT_SOURCE_PRODUCT':   np.prod(ext_vals) if ext_vals else 0.125,
        'CREDIT_INCOME_RATIO':  req.amt_credit / (req.amt_income_total + 1),
        'ANNUITY_INCOME_RATIO': req.amt_annuity / (req.amt_income_total + 1),
        'CREDIT_TERM_MONTHS':   req.amt_credit / (req.amt_annuity + 1),
        'GOODS_CREDIT_RATIO':   (req.amt_goods_price or req.amt_credit*0.9) / (req.amt_credit + 1),
        'AGE_YEARS':            -req.days_birth / 365,
        'EMPLOYMENT_YEARS':     -req.days_employed / 365 if req.days_employed != 365243 else 0,
        'IS_PENSIONER':         1 if req.days_employed == 365243 else 0,
        'BUREAU_UTILIZATION':   req.bureau_utilization or 0.3,
        'INST_LATE_RATE':       req.inst_late_rate or 0.0,
        'PREV_REFUSAL_RATE':    req.prev_refusal_rate or 0.0,
    }
    return pd.DataFrame([features])

# ── Endpoints ──────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "service": "Credit Risk Intelligence Engine",
        "version": "1.0.0",
        "endpoints": ["/score", "/health", "/docs"]
    }

@app.get("/health")
def health():
    return {"status": "healthy", "model": "LGBMClassifier"}

@app.post("/score", response_model=ScoringResponse)
def score_application(req: ApplicationRequest):
    try:
        features_df = build_features(req)

        # Get model feature names and align
        model_features = model.feature_name_
        for col in model_features:
            if col not in features_df.columns:
                features_df[col] = 0
        features_df = features_df[model_features]

        pd_score = float(model.predict_proba(features_df)[0][1])
        risk_tier, rate = get_risk_tier_and_rate(pd_score)
        decision = "DECLINED" if rate is None else "APPROVED"

        # Expected loss
        avg_lgd    = 0.45
        exp_loss   = pd_score * avg_lgd * req.amt_credit

        # Adverse action codes
        adverse = []
        if decision == "DECLINED":
            feat_vals = features_df.iloc[0].to_dict()
            risky_feats = {
                'EXT_SOURCE_MEAN':     feat_vals.get('EXT_SOURCE_MEAN', 0.5) < 0.4,
                'INST_LATE_RATE':      feat_vals.get('INST_LATE_RATE', 0) > 0.1,
                'BUREAU_UTILIZATION':  feat_vals.get('BUREAU_UTILIZATION', 0) > 0.7,
                'CREDIT_INCOME_RATIO': feat_vals.get('CREDIT_INCOME_RATIO', 0) > 5,
                'PREV_REFUSAL_RATE':   feat_vals.get('PREV_REFUSAL_RATE', 0) > 0.3,
            }
            for feat, triggered in risky_feats.items():
                if triggered and feat in ADVERSE_CODES:
                    code, desc = ADVERSE_CODES[feat]
                    adverse.append({"code": code, "description": desc})

        explanation = (
            f"PD Score {pd_score:.3f} → Risk Tier {risk_tier} → "
            f"{'Approved at ' + str(rate) + '% APR' if decision == 'APPROVED' else 'Declined'}"
        )

        return ScoringResponse(
            application_id       = req.application_id,
            pd_score             = round(pd_score, 4),
            decision             = decision,
            risk_tier            = risk_tier,
            recommended_rate_apr = rate,
            adverse_action_codes = adverse,
            expected_loss_eur    = round(exp_loss, 2),
            explanation          = explanation
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))