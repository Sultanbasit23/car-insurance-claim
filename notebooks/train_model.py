import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# ==========================
# LOAD DATA
# ==========================
df = pd.read_csv("data/car_insurance.csv")

# ==========================
# CLEAN CURRENCY COLUMNS
# ==========================
currency_cols = ["INCOME", "HOME_VAL", "BLUEBOOK"]

for col in currency_cols:
    df[col] = (
        df[col]
        .astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
    )
    df[col] = pd.to_numeric(df[col], errors="coerce")

# ==========================
# SELECT ONLY APP FEATURES
# ==========================
features = [
    "AGE",
    "INCOME",
    "HOME_VAL",
    "BLUEBOOK",
    "CLM_FREQ",
    "CAR_AGE",
    "URBANICITY",
    "CAR_USE",
    "EDUCATION"
]

X = df[features]
y = df["CLAIM_FLAG"]

# ==========================
# SPLIT
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==========================
# PREPROCESSING
# ==========================
numeric_features = [
    "AGE", "INCOME", "HOME_VAL",
    "BLUEBOOK", "CLM_FREQ", "CAR_AGE"
]

categorical_features = [
    "URBANICITY", "CAR_USE", "EDUCATION"
]

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# ==========================
# LOGISTIC MODEL
# ==========================
logistic_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=3000))
])

logistic_pipeline.fit(X_train, y_train)
log_auc = roc_auc_score(y_test, logistic_pipeline.predict_proba(X_test)[:,1])
print("Logistic ROC-AUC:", log_auc)

# ==========================
# RANDOM FOREST
# ==========================
rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42))
])

rf_pipeline.fit(X_train, y_train)
rf_auc = roc_auc_score(y_test, rf_pipeline.predict_proba(X_test)[:,1])
print("Random Forest ROC-AUC:", rf_auc)

# ==========================
# SAVE MODELS
# ==========================
os.makedirs("models", exist_ok=True)

joblib.dump(logistic_pipeline, "models/logistic_pipeline.pkl")
joblib.dump(rf_pipeline, "models/random_forest_pipeline.pkl")

print("✅ Models saved successfully.")
