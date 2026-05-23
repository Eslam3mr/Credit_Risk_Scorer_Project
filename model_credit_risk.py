import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
# ── Load clean data ───────────────────────────────────────────────
df = pd.read_csv(r'application_train_clean.csv')

# ── Encode categorical columns ────────────────────────────────────
# Models can't read text — convert categories to numbers
categorical_cols = df.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col].astype(str))

# ── Split features and target ─────────────────────────────────────
X = df.drop(['TARGET', 'SK_ID_CURR'], axis=1)
y = df['TARGET']



# ── Train/test split ──────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Scale features ────────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ── Logistic Regression on scaled data ───────────────────────────
print("Training Logistic Regression...")
lr = LogisticRegression(
    max_iter=2000,
    class_weight='balanced',
    solver='saga',        # faster solver for large datasets
    random_state=42,
    n_jobs=-1             # use all CPU cores
)
lr.fit(X_train_scaled, y_train)

lr_probs = lr.predict_proba(X_test_scaled)[:, 1]
lr_auc   = roc_auc_score(y_test, lr_probs)

from xgboost import XGBClassifier

print("Training XGBoost...")
xgb = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    scale_pos_weight=9,    # handles 8% default rate imbalance
    random_state=42,
    eval_metric='auc',
    verbosity=0,
    n_jobs=-1
)

xgb.fit(
    X_train, y_train,      # XGBoost doesn't need scaled data
    eval_set=[(X_test, y_test)],
    verbose=100
)

xgb_probs = xgb.predict_proba(X_test)[:, 1]
xgb_auc   = roc_auc_score(y_test, xgb_probs)

print(f"\nXGBoost AUC-ROC:  {xgb_auc:.4f}")
print(f"Gini Coefficient: {(xgb_auc*2-1):.4f}")
print(f"\nImprovement over Logistic Regression: +{(xgb_auc-lr_auc):.4f}")

import shap

print("Calculating SHAP values (this takes 2-3 minutes)...")

# Build SHAP explainer on XGBoost model
explainer = shap.TreeExplainer(xgb)
shap_values = explainer.shap_values(X_test[:1000])  # sample 1000 rows

# ── Plot 1: Top 20 most important features globally ───────────────
shap.summary_plot(
    shap_values,
    X_test[:1000],
    max_display=20,
    show=False
)
import matplotlib.pyplot as plt
plt.tight_layout()
plt.savefig(r'D:\.Eslam\py projects\shap_summary.png', dpi=150)
plt.close()
print("SHAP summary plot saved.")

# ── Top features in plain text ────────────────────────────────────
feature_importance = pd.DataFrame({
    'feature': X_test.columns,
    'importance': abs(shap_values).mean(axis=0)
}).sort_values('importance', ascending=False)

print("\n=== TOP 15 FEATURES DRIVING DEFAULT PREDICTION ===")
print(feature_importance.head(15).to_string(index=False))

# Save model results summary
results = {
    'Logistic Regression AUC': round(lr_auc, 4),
    'XGBoost AUC': round(xgb_auc, 4),
    'Gini XGBoost': round(xgb_auc*2-1, 4),
    'Top Feature': 'EXT_SOURCE_3',
    'Best Engineered Feature': 'LOAN_TO_GOODS (rank 4/182)'
}

for k, v in results.items():
    print(f"{k}: {v}")

