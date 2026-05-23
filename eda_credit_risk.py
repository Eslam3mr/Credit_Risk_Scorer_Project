
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r'application_train.csv')

# ── Build missing report first ────────────────────────────────────
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_report = pd.DataFrame({
    'Missing Count': missing,
    'Missing %': missing_pct
}).sort_values('Missing %', ascending=False)

# ── Step 1: Flag high-missing columns BEFORE filling ──────────────
high_missing = missing_report[missing_report['Missing %'] > 40].index.tolist()

for col in high_missing:
    df[f'{col}_WAS_MISSING'] = df[col].isnull().astype(int)

print(f"Created {len(high_missing)} binary missing flags")

# ── Step 2: Fill numeric columns with median ──────────────────────
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# ── Step 3: Fill categorical columns with mode ────────────────────
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])  # reassign instead of inplace

# ── Step 4: Verify ────────────────────────────────────────────────
remaining_missing = df.isnull().sum().sum()
print(f"Remaining missing values: {remaining_missing}")
print(f"Dataset shape after cleaning: {df.shape}")

# ── Step 5: Show what's still missing if anything ─────────────────
still_missing = df.isnull().sum()
still_missing = still_missing[still_missing > 0]
if len(still_missing) > 0:
    print("\nStill missing:")
    print(still_missing)
else:
    print("\nDataset is fully clean. Zero missing values.")

    df = df.copy()  # defragment the dataframe

# ── 1. Default rate overview ──────────────────────────────────────
print("=== DEFAULT RATE ===")
print(df['TARGET'].value_counts())
print(f"\nDefault rate: {df['TARGET'].mean()*100:.2f}%")

# ── 2. Default rate by income type ───────────────────────────────
print("\n=== DEFAULT RATE BY INCOME TYPE ===")
print(df.groupby('NAME_INCOME_TYPE')['TARGET'].mean().sort_values(ascending=False).round(3))

# ── 3. Default rate by education ─────────────────────────────────
print("\n=== DEFAULT RATE BY EDUCATION ===")
print(df.groupby('NAME_EDUCATION_TYPE')['TARGET'].mean().sort_values(ascending=False).round(3))

# ── 4. Default rate by gender ────────────────────────────────────
print("\n=== DEFAULT RATE BY GENDER ===")
print(df.groupby('CODE_GENDER')['TARGET'].mean().round(3))

# ── 5. Age vs default ─────────────────────────────────────────────
# DAYS_BIRTH is negative (days before application) - convert to years
df['AGE_YEARS'] = (df['DAYS_BIRTH'] / -365).astype(int)
print("\n=== DEFAULT RATE BY AGE GROUP ===")
df['AGE_GROUP'] = pd.cut(df['AGE_YEARS'], bins=[20,30,40,50,60,70])
print(df.groupby('AGE_GROUP', observed=True)['TARGET'].mean().round(3))
# ── Numeric correlations with default ─────────────────────────────
print("=== NUMERIC CORRELATIONS WITH DEFAULT ===")
numeric_features = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 
                    'AMT_ANNUITY', 'DAYS_EMPLOYED', 'EXT_SOURCE_2']

correlations = df[numeric_features + ['TARGET']].corr()['TARGET'].drop('TARGET')
print(correlations.sort_values(key=abs, ascending=False).round(4))

# ── Credit to Income Ratio ─────────────────────────────────────────
# This is the ratio every banker actually looks at
df['CREDIT_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
df['ANNUITY_INCOME_RATIO'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']

print("\n=== DEFAULT RATE BY CREDIT/INCOME RATIO ===")
df['CREDIT_INCOME_BUCKET'] = pd.cut(df['CREDIT_INCOME_RATIO'], 
                                     bins=[0,1,2,4,8,100])
print(df.groupby('CREDIT_INCOME_BUCKET', observed=True)['TARGET']
        .mean().round(3))

print("\n=== ANNUITY/INCOME RATIO VS DEFAULT ===")
print(f"Defaulters avg ratio:     {df[df['TARGET']==1]['ANNUITY_INCOME_RATIO'].mean():.3f}")
print(f"Non-defaulters avg ratio: {df[df['TARGET']==0]['ANNUITY_INCOME_RATIO'].mean():.3f}")

# Who has the suspicious DAYS_EMPLOYED value?
mask = df['DAYS_EMPLOYED'] == 365243
print("=== WHO HAS 365,243 DAYS EMPLOYED ===")
print(df[mask]['NAME_INCOME_TYPE'].value_counts())
print(f"\nTotal affected: {mask.sum()} rows ({mask.mean()*100:.1f}% of dataset)")
# ── Fix DAYS_EMPLOYED sentinel value ──────────────────────────────

# First: create a flag capturing who these people are
df['DAYS_EMPLOYED_ANOMALY'] = (df['DAYS_EMPLOYED'] == 365243).astype(int)

# Then: replace the fake value with NaN, then fill with median
df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace(365243, np.nan)
df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].fillna(df['DAYS_EMPLOYED'].median())

# Verify
print(f"Max DAYS_EMPLOYED now: {df['DAYS_EMPLOYED'].max()}")
print(f"Anomaly flags created: {df['DAYS_EMPLOYED_ANOMALY'].sum()}")

# Check if the flag itself predicts default
print("\n=== DEFAULT RATE: ANOMALY VS NORMAL ===")
print(df.groupby('DAYS_EMPLOYED_ANOMALY')['TARGET'].mean().round(3))
df.to_csv(r'D:\.Eslam\py projects\application_train_clean.csv', index=False)
print("Clean dataset saved.")