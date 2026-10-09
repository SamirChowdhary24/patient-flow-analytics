import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import shap
import os

print("Starting Predictive Analytics & SHAP Engine...")

# 1. Load the processed dataset
df = pd.read_csv('data/processed/powerbi_ready_advanced.csv')

# 2. Feature Selection & Preprocessing
features = ['Length_Of_Stay_Hours', 'Claim_Cost', 'Insurance_Coverage', 'Out_Of_Pocket_Cost', 'Coverage_Ratio']

# Handle missing values
for col in features:
    df[col] = df[col].fillna(0)

# Fill missing Days_Since_Last_Visit (first-time visits get a default high number)
df['Days_Since_Last_Visit_Cleaned'] = df['Days_Since_Last_Visit'].fillna(999)
features.append('Days_Since_Last_Visit_Cleaned')

# Encode categorical Department
dept_dummies = pd.get_dummies(df['Department'], prefix='Dept', drop_first=True, dtype=int)
features.extend(dept_dummies.columns.tolist())

X = pd.concat([df[features[:6]], dept_dummies], axis=1)
y = df['Is_30_Day_Readmission']

# 3. Train-Test Split & Random Forest Classifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Model Performance Metric (AUC-ROC)
y_pred_proba = model.predict_proba(X_test)[:, 1]
auc_score = roc_auc_score(y_test, y_pred_proba)
print(f"Model AUC-ROC Score: {auc_score:.4f}")

# 4. Generate Readmission Risk Probabilities across all encounters
df['Predicted_Readmission_Risk'] = np.round(model.predict_proba(X)[:, 1] * 100, 1)

# Assign Risk Tiers
conditions = [
    df['Predicted_Readmission_Risk'] >= 65,
    (df['Predicted_Readmission_Risk'] >= 30) & (df['Predicted_Readmission_Risk'] < 65),
    df['Predicted_Readmission_Risk'] < 30
]
tiers = ['High Risk', 'Medium Risk', 'Low Risk']
df['Risk_Tier'] = np.select(conditions, tiers, default='Low Risk')

# 5. Explainable AI: SHAP Feature Importance
print("Computing SHAP explanations...")
sample_X = X.sample(n=min(1000, len(X)), random_state=42)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(sample_X)

# Handle binary classification output shape across SHAP versions
if isinstance(shap_values, list):
    mean_shap = np.abs(shap_values[1]).mean(axis=0)
elif shap_values.ndim == 3:
    mean_shap = np.abs(shap_values[:, :, 1]).mean(axis=0)
else:
    mean_shap = np.abs(shap_values).mean(axis=0)

shap_df = pd.DataFrame({
    'Feature': X.columns,
    'Impact_Weight': mean_shap
}).sort_values(by='Impact_Weight', ascending=False)

# Human-readable feature names for dashboard presentation
feature_map = {
    'Days_Since_Last_Visit_Cleaned': 'Days Since Previous Visit',
    'Length_Of_Stay_Hours': 'Length of Stay (Hours)',
    'Out_Of_Pocket_Cost': 'Out-of-Pocket Expense',
    'Claim_Cost': 'Total Claim Cost',
    'Coverage_Ratio': 'Insurance Coverage Ratio',
    'Insurance_Coverage': 'Payer Coverage Amount',
    'Dept_emergency': 'Department: Emergency',
    'Dept_inpatient': 'Department: Inpatient',
    'Dept_urgentcare': 'Department: Urgent Care'
}
shap_df['Feature_Label'] = shap_df['Feature'].map(lambda x: feature_map.get(x, x))

# 6. Save final artifacts for Power BI
output_ml_path = 'data/processed/powerbi_ready_ml.csv'
output_shap_path = 'data/processed/shap_feature_importance.csv'

df.to_csv(output_ml_path, index=False)
shap_df.to_csv(output_shap_path, index=False)

print(f"ML Pipeline Complete!")
print(f"  Enriched patient data: {output_ml_path}")
print(f"  SHAP importance table: {output_shap_path}")