import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Risk Scorer",
    page_icon="🏦",
    layout="centered"
)

# ── Load model assets ─────────────────────────────────────────────
@st.cache_resource
def load_assets():
    model    = joblib.load(r'D:\.Eslam\py projects\xgb_model.pkl')
    medians  = pd.read_csv(
                   r'D:\.Eslam\py projects\feature_medians.csv',
                   index_col=0).squeeze()
    features = pd.read_csv(
                   r'D:\.Eslam\py projects\feature_cols.csv'
               ).iloc[:, 0].tolist()
    explainer = shap.TreeExplainer(model)
    return model, medians, features, explainer

model, medians, features, explainer = load_assets()

# ── UI ────────────────────────────────────────────────────────────
st.title("🏦 Credit Risk Scorer")
st.caption("Enter applicant details to get a default probability and risk drivers.")

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Applicant Profile")
    age            = st.slider("Age", 20, 70, 35)
    gender         = st.selectbox("Gender", ["Male", "Female"])
    education      = st.selectbox("Education", [
                        "Lower secondary",
                        "Secondary / secondary special",
                        "Incomplete higher",
                        "Higher education",
                        "Academic degree"
                     ])
    income_type    = st.selectbox("Income Type", [
                        "Working",
                        "Commercial associate",
                        "Pensioner",
                        "State servant",
                        "Unemployed"
                     ])
    family_members = st.slider("Family Members", 1, 10, 2)
    children       = st.slider("Children", 0, 5, 0)

with col2:
    st.subheader("Loan Details")
    annual_income  = st.number_input(
                        "Annual Income (USD)", 
                        min_value=10000, 
                        max_value=1000000, 
                        value=150000, 
                        step=5000
                     )
    loan_amount    = st.number_input(
                        "Loan Amount (USD)", 
                        min_value=10000, 
                        max_value=2000000, 
                        value=300000, 
                        step=10000
                     )
    annuity        = st.number_input(
                        "Monthly Payment (USD)", 
                        min_value=1000, 
                        max_value=100000, 
                        value=15000, 
                        step=500
                     )
    goods_price    = st.number_input(
                        "Goods Price (USD)", 
                        min_value=10000, 
                        max_value=2000000, 
                        value=250000, 
                        step=10000
                     )
    days_employed  = st.slider("Years at Current Job", 0, 40, 5)
    ext_source_2   = st.slider(
                        "External Credit Score (0-1)", 
                        0.0, 1.0, 0.5, 0.01
                     )

st.divider()

# ── Predict button ────────────────────────────────────────────────
if st.button("Calculate Risk Score", type="primary", use_container_width=True):

    # Build input row from medians baseline
    input_data = medians.copy()

    # Override with user inputs
    input_data['DAYS_BIRTH']        = age * -365
    input_data['CODE_GENDER']       = 1 if gender == "Male" else 0
    input_data['AMT_INCOME_TOTAL']  = annual_income
    input_data['AMT_CREDIT']        = loan_amount
    input_data['AMT_ANNUITY']       = annuity
    input_data['AMT_GOODS_PRICE']   = goods_price
    input_data['DAYS_EMPLOYED']     = days_employed * -365
    input_data['EXT_SOURCE_2']      = ext_source_2
    input_data['CNT_FAM_MEMBERS']   = family_members
    input_data['CNT_CHILDREN']      = children

    # Engineered features
    input_data['LOAN_TO_GOODS']       = loan_amount / goods_price
    input_data['DEBT_TO_INCOME']      = loan_amount / annual_income
    input_data['PAYMENT_BURDEN']      = annuity / annual_income
    input_data['INCOME_PER_PERSON']   = annual_income / family_members
    input_data['EMPLOYMENT_RATIO']    = (days_employed * 365) / (age * 365)

    # Align to model feature order
    input_df = pd.DataFrame([input_data])[features]

    # Predict
    prob       = model.predict_proba(input_df)[0][1]
    risk_pct   = prob * 100

    # ── Risk display ──────────────────────────────────────────────
    if risk_pct < 10:
        color, verdict, emoji = "green",  "LOW RISK",    "✅"
    elif risk_pct < 25:
        color, verdict, emoji = "orange", "MEDIUM RISK", "⚠️"
    else:
        color, verdict, emoji = "red",    "HIGH RISK",   "🚨"

    st.markdown(f"""
    <div style='text-align:center; padding:30px; 
                border-radius:12px; background:#f8f9fa;
                border: 2px solid {color}'>
        <h1 style='color:{color}; margin:0'>{emoji} {verdict}</h1>
        <h2 style='margin:10px 0'>Default Probability: 
            <span style='color:{color}'>{risk_pct:.1f}%</span>
        </h2>
        <p style='color:gray'>Industry threshold: 10% = low risk | 
            25% = high risk</p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── Key ratios ────────────────────────────────────────────────
    st.subheader("📊 Financial Ratios")
    r1, r2, r3 = st.columns(3)
    r1.metric("Debt to Income",    f"{loan_amount/annual_income:.1f}x",
              help="Loan amount vs annual income")
    r2.metric("Payment Burden",    f"{annuity/annual_income*100:.1f}%",
              help="Monthly payment as % of annual income")
    r3.metric("Loan to Goods",     f"{loan_amount/goods_price:.2f}x",
              help="Over-borrowing signal — above 1.0 is risky")

    st.divider()

    # ── SHAP explanation ──────────────────────────────────────────
    st.subheader("🔍 Why this score? — Top Risk Drivers")

    shap_vals  = explainer.shap_values(input_df)
    shap_series = pd.Series(
        shap_vals[0], index=features
    ).abs().sort_values(ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(8, 4))
    colors  = ['#e74c3c' if v > 0 else '#2ecc71'
               for v in shap_vals[0][
                   [features.index(f) for f in shap_series.index]
               ]]
    ax.barh(shap_series.index[::-1], shap_series.values[::-1],
            color=colors[::-1])
    ax.set_xlabel("Impact on default probability")
    ax.set_title("Top 10 factors driving this applicant's risk score")
    ax.axvline(x=0, color='black', linewidth=0.8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.caption("🔴 Red = increases default risk  |  "
               "🟢 Green = decreases default risk")