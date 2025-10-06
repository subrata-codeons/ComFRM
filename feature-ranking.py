import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.feature_selection import RFE
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
import shap
import warnings

warnings.filterwarnings("ignore")

# === Load and preprocess data ===
df = pd.read_csv("BRCA-TCGA.csv")
df.replace(".", np.nan, inplace=True)

# Identify columns
identifier_col = df.columns[0]
target_col = df.columns[-1]
feature_cols = df.columns[1:-1]

# Encode target if necessary
if df[target_col].dtype == "object":
    le = LabelEncoder()
    df[target_col] = le.fit_transform(df[target_col])

# Convert features to numeric
df[feature_cols] = df[feature_cols].apply(pd.to_numeric)

# === KNN Imputation ===
imputer = KNNImputer(n_neighbors=5)
df[feature_cols] = imputer.fit_transform(df[feature_cols])

# Optional: Clip values to [0, 1] if expected to be rankscores
df[feature_cols] = np.clip(df[feature_cols], 0, 1)

# Save imputed data
df.to_csv("BRCA-TCGA_validation_26Nov2024_imputed.csv", index=False)
print("Imputed data saved as 'BRCA-TCGA_validation_26Nov2024_imputed.csv'.")


# Split data
X = df[feature_cols].values
y = df[target_col].values
feature_names = feature_cols.tolist()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
}

# === RFE Feature Ranking ===
for name, model in models.items():
    try:
        rfe = RFE(estimator=model, n_features_to_select=1, step=1)
        rfe.fit(X_train, y_train)
        rfe_ranking = pd.DataFrame({
            "Feature": feature_names,
            "RFE_Ranking": rfe.ranking_
        }).sort_values(by="RFE_Ranking")
        rfe_ranking.to_csv(f"V1-rfe_ranking_{name}.csv", index=False)
        print(f"RFE ranking saved for {name}.")
    except Exception as e:
        print(f"Skipping RFE for {name}: {e}")

# === SHAP Feature Ranking ===
for name, model in models.items():
    try:
        model.fit(X_train, y_train)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train)

        if isinstance(shap_values, list):  # binary case
            shap_vals = np.abs(shap_values[1]).mean(axis=0)
        else:
            shap_vals = np.abs(shap_values).mean(axis=0)

        shap_ranking = pd.DataFrame({
            "Feature": feature_names,
            "SHAP_Importance": shap_vals
        }).sort_values(by="SHAP_Importance", ascending=False)
        shap_ranking["Ranking"] = range(1, len(shap_ranking) + 1)
        shap_ranking.to_csv(f"V1-shap_ranking_{name}.csv", index=False)
        print(f"SHAP ranking saved for {name}.")
    except Exception as e:
        print(f"Skipping SHAP for {name}: {e}")

# === Feature Importance Ranking (Gini for RF) ===
for name, model in models.items():
    try:
        model.fit(X_train, y_train)
        
        importances = model.feature_importances_

        imp_df = pd.DataFrame({
            "Feature": feature_names,
            "Feature_Importance": importances
        }).sort_values(by="Feature_Importance", ascending=False)
        imp_df["Ranking"] = range(1, len(imp_df) + 1)
        imp_df.to_csv(f"V1-feature_importance_{name}.csv", index=False)
        print(f"Feature importance ranking saved for {name}.")
    except Exception as e:
        print(f"Skipping feature importance for {name}: {e}")
