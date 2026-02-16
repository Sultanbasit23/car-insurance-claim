import pandas as pd
import joblib
import shap

from sklearn.preprocessing import LabelEncoder


# ============================
# LOAD DATA
# ============================

df = pd.read_csv("../data/car_insurance.csv")


# ============================
# REMOVE ID
# ============================

if "ID" in df.columns:
    df = df.drop("ID", axis=1)


# ============================
# HANDLE MISSING VALUES
# ============================

for col in df.columns:

    if df[col].dtype == "object":
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].median())


# ============================
# PREPARE FEATURES
# ============================

X = df.drop("CLAIM_FLAG", axis=1)


# Encode categoricals
encoder = LabelEncoder()

for col in X.select_dtypes(include="object"):
    X[col] = encoder.fit_transform(X[col])


# ============================
# LOAD TRAINED MODEL
# ============================

model = joblib.load("../models/model.pkl")


# ============================
# SHAP EXPLAINER
# ============================

explainer = shap.TreeExplainer(model)


# Use small sample for speed
X_sample = X.sample(500, random_state=42)


shap_values = explainer.shap_values(X_sample)


# ============================
# VISUALIZATION
# ============================

print("Showing SHAP summary plot...")

shap.summary_plot(shap_values, X_sample)
