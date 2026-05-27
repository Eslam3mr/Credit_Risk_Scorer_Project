# ── Imports ───────────────────────────────────────────────────────
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import joblib

# ── 1. Load Data ──────────────────────────────────────────────────
def load_data(path):
    df = pd.read_csv(path)
    print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

# ── 2. Clean Data ─────────────────────────────────────────────────
def clean_data(df):
    # Build all missing flag columns at once — no fragmentation
    missing_flags = pd.concat([
        df[col].isnull().astype(int).rename(f'{col}_WAS_MISSING')
        for col in high_missing
    ], axis=1)
    df = pd.concat([df, missing_flags], axis=1)
    
    # Fill numeric with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # Fill categorical with mode
    categorical_cols = df.select_dtypes(include=['object', 'str']).columns
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    # Fix DAYS_EMPLOYED sentinel
    df['DAYS_EMPLOYED_ANOMALY'] = (df['DAYS_EMPLOYED'] == 365243).astype(int)
    df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace(365243, np.nan)
    df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].fillna(df['DAYS_EMPLOYED'].median())
    
    df = df.copy()
    print(f"Cleaned: {df.isnull().sum().sum()} missing values remaining")
    return df

# ── 3. Engineer Features ──────────────────────────────────────────
def engineer_features(df):
    df['PAYMENT_BURDEN']    = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['DEBT_TO_INCOME']    = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['LOAN_TO_GOODS']     = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']
    df['EMPLOYMENT_RATIO']  = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['AGE_YEARS']         = (df['DAYS_BIRTH'] / -365).astype(int)
    
    print(f"Features engineered: {df.shape[1]} total columns")
    return df

# ── 4. Prepare Model Input ────────────────────────────────────────
def prepare_model_input(df):
    # Encode categoricals
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    categorical_cols = df.select_dtypes(include=['object', 'str']).columns
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col].astype(str))
    
    X = df.drop(['TARGET', 'SK_ID_CURR'], axis=1)
    y = df['TARGET']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test

# ── 5. Train Models ───────────────────────────────────────────────
def train_models(X_train, X_test, y_train, y_test):
    # Scale for Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    
    # Logistic Regression
    print("Training Logistic Regression...")
    lr = LogisticRegression(
        max_iter=2000,
        class_weight='balanced',
        solver='saga',
        random_state=42
    )
    lr.fit(X_train_scaled, y_train)
    lr_auc = roc_auc_score(y_test, lr.predict_proba(X_test_scaled)[:, 1])
    print(f"LR AUC: {lr_auc:.4f}")
    
    # XGBoost
    print("Training XGBoost...")
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=9,
        random_state=42,
        eval_metric='auc',
        verbosity=0
    )
    xgb.fit(X_train, y_train)
    xgb_auc = roc_auc_score(y_test, xgb.predict_proba(X_test)[:, 1])
    print(f"XGB AUC: {xgb_auc:.4f}")
    print(f"Improvement: +{(xgb_auc - lr_auc):.4f}")
    
    return lr, xgb, scaler, lr_auc, xgb_auc

# ── 6. Save Artifacts ─────────────────────────────────────────────
def save_artifacts(xgb, scaler, X_train, path):
    joblib.dump(xgb, f'{path}/xgb_model.pkl')
    joblib.dump(scaler, f'{path}/scaler.pkl')
    X_train.median().to_csv(f'{path}/feature_medians.csv')
    pd.Series(X_train.columns.tolist()).to_csv(
        f'{path}/feature_cols.csv', index=False
    )
    print("Artifacts saved.")

# ── MAIN ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    PATH = r'D:\.Eslam\py projects'
    
    df = load_data(f'{PATH}/application_train.csv')
    df = clean_data(df)
    df = engineer_features(df)
    
    X_train, X_test, y_train, y_test = prepare_model_input(df)
    lr, xgb, scaler, lr_auc, xgb_auc = train_models(
        X_train, X_test, y_train, y_test
    )
    
    save_artifacts(xgb, scaler, X_train, PATH)
