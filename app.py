import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc

# =============================
# LOAD MODELS
# =============================
logistic_model = joblib.load("models/logistic_pipeline.pkl")
rf_model = joblib.load("models/random_forest_pipeline.pkl")

# Hardcoded AUC from training output
LOG_AUC = 0.764
RF_AUC = 0.748

st.set_page_config(page_title="Car Insurance AI", layout="wide")

st.title("🚗 Car Insurance Claim Prediction System")

# =============================
# SIDEBAR INPUT
# =============================
st.sidebar.header("Customer Information")

age = st.sidebar.slider("Age", 18, 80, 35)
income = st.sidebar.number_input("Income", value=50000)
home_val = st.sidebar.number_input("Home Value", value=200000)
bluebook = st.sidebar.number_input("Car Value (Bluebook)", value=15000)
clm_freq = st.sidebar.slider("Claim Frequency", 0, 5, 1)
car_age = st.sidebar.slider("Car Age", 0, 20, 5)

urbanicity = st.sidebar.selectbox("Urbanicity",
                                   ["Highly Urban/Urban", "Suburban", "Rural"])

car_use = st.sidebar.selectbox("Car Use",
                               ["Private", "Commercial"])

education = st.sidebar.selectbox("Education",
                                  ["<High School", "High School",
                                   "Bachelors", "Masters", "PhD"])

# =============================
# MODEL SELECTION
# =============================
model_choice = st.selectbox("Select Model",
                            ["Logistic Regression", "Random Forest"])

model = logistic_model if model_choice == "Logistic Regression" else rf_model

# =============================
# CREATE INPUT DATAFRAME
# =============================
input_data = pd.DataFrame([{
    "AGE": age,
    "INCOME": income,
    "HOME_VAL": home_val,
    "BLUEBOOK": bluebook,
    "CLM_FREQ": clm_freq,
    "CAR_AGE": car_age,
    "URBANICITY": urbanicity,
    "CAR_USE": car_use,
    "EDUCATION": education
}])

# =============================
# PREDICTION
# =============================
if st.button("Predict Claim Risk"):

    probability = model.predict_proba(input_data)[0][1]
    prediction = model.predict(input_data)[0]

    st.subheader("Prediction Result")

    # Risk Label
    if probability < 0.30:
        st.success(f"Low Risk ({probability:.2%})")
    elif probability < 0.60:
        st.warning(f"Medium Risk ({probability:.2%})")
    else:
        st.error(f"High Risk ({probability:.2%})")

    # =============================
    # GAUGE CHART
    # =============================
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        title={'text': "Claim Probability (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'steps': [
                {'range': [0, 30], 'color': "green"},
                {'range': [30, 60], 'color': "yellow"},
                {'range': [60, 100], 'color': "red"}
            ],
        }
    ))
    st.plotly_chart(fig)

# =============================
# AUC COMPARISON
# =============================
st.subheader("📊 Model Performance Comparison")

fig_auc = go.Figure()
fig_auc.add_trace(go.Bar(
    x=["Logistic Regression", "Random Forest"],
    y=[LOG_AUC, RF_AUC]
))
fig_auc.update_layout(title="ROC-AUC Comparison")
st.plotly_chart(fig_auc)

# =============================
# SAMPLE ROC CURVE (Visualization Only)
# =============================
st.subheader("📈 ROC Curve Example")

# Fake demo curve for display
fpr = np.linspace(0, 1, 100)
tpr = np.sqrt(fpr)

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name="ROC Curve"))
fig2.add_trace(go.Scatter(x=[0,1], y=[0,1],
                          mode='lines',
                          line=dict(dash='dash'),
                          name="Random"))
fig2.update_layout(xaxis_title="False Positive Rate",
                   yaxis_title="True Positive Rate")
st.plotly_chart(fig2)

# =============================
# CONFUSION MATRIX SAMPLE
# =============================
st.subheader("📊 Confusion Matrix Example")

y_true = [0,0,1,1,0,1,0,1]
y_pred = [0,1,1,1,0,0,0,1]

cm = confusion_matrix(y_true, y_pred)

fig_cm, ax = plt.subplots()
ax.matshow(cm)
for (i, j), val in np.ndenumerate(cm):
    ax.text(j, i, val, ha='center', va='center')
plt.xlabel("Predicted")
plt.ylabel("Actual")

st.pyplot(fig_cm)
