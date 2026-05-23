# Credit Risk Scorer

End-to-end credit default prediction system built on 307,000 
real loan applications from Home Credit Group.

## What it does
- Predicts default probability for any loan applicant
- Explains the top 10 risk drivers using SHAP values
- Deployed as an interactive Streamlit application

## Results
- XGBoost AUC: 0.761 | Gini: 0.52
- Logistic Regression baseline AUC: 0.748
- 182 features after engineering (original: 122)

## Key engineered features
- LOAN_TO_GOODS — over-borrowing signal (rank 4/182)
- EMPLOYMENT_RATIO — stability signal
- DAYS_EMPLOYED_ANOMALY — pensioner/unemployed flag

## Stack
Python · XGBoost · Scikit-learn · SHAP · Streamlit · Pandas



Data From : https://www.kaggle.com/competitions/home-credit-default-risk/overview
