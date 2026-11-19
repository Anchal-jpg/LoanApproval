from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_DIR / "models" / "loan_pipeline.joblib"

st.set_page_config(page_title="Loan Approval Predictor", page_icon="🏦", layout="wide")
st.title("🏦 Loan Approval Prediction")
st.caption("ML-powered loan approval decision support")

if not MODEL_PATH.exists():
    st.error("Trained model not found. Run `python src/train.py` first.")
    st.stop()

model = joblib.load(MODEL_PATH)

with st.sidebar:
    st.header("Applicant Details")
    dependents = st.number_input("Number of dependents", 0, 20, 2)
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    employment = st.selectbox("Employment", ["No", "Yes"])

col1, col2, col3 = st.columns(3)
with col1:
    income = st.number_input("Annual income", 100000, 100000000, 5000000, step=100000)
    loan_amount = st.number_input("Loan amount", 100000, 100000000, 15000000, step=100000)
    loan_term = st.slider("Loan term (years)", 2, 30, 10)
with col2:
    cibil = st.slider("CIBIL score", 300, 900, 600)
    residential_assets = st.number_input("Residential assets value", 0, 100000000, 5000000, step=100000)
    commercial_assets = st.number_input("Commercial assets value", 0, 100000000, 3000000, step=100000)
with col3:
    luxury_assets = st.number_input("Luxury assets value", 0, 100000000, 7000000, step=100000)
    bank_assets = st.number_input("Bank assets value", 0, 100000000, 4000000, step=100000)

if st.button("Predict Loan Status", type="primary", use_container_width=True):
    application = pd.DataFrame(
        [{
            "no_of_dependents": dependents,
            "education": education,
            "self_employed": employment,
            "income_annum": income,
            "loan_amount": loan_amount,
            "loan_term": loan_term,
            "cibil_score": cibil,
            "residential_assets_value": residential_assets,
            "commercial_assets_value": commercial_assets,
            "luxury_assets_value": luxury_assets,
            "bank_asset_value": bank_assets,
        }]
    )
    prediction = int(model.predict(application)[0])
    probability = float(model.predict_proba(application)[0][prediction])
    if prediction == 1:
        st.success(f"✅ Loan Approved  |  Confidence: {probability:.1%}")
    else:
        st.error(f"❌ Loan Rejected  |  Confidence: {probability:.1%}")
