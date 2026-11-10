from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="NaijaGuard AI - Fraud Detection",
    page_icon="🛡️",
    layout="centered",
)


# Load the model directly (cached for performance)
@st.cache_resource
def load_model():
  # Adjust this path if your model folder is in a different location relative to this file
  base_dir = Path(__file__).resolve().parent.parent
  model_path = base_dir / "models" / "fraud_detection_model.pkl"
  return joblib.load(model_path)


model = load_model()

st.title("🛡️ NaijaGuard AI: Fraud Detection System")
st.markdown("Enter transaction details below to evaluate fraud risk in real-time.")

# Input Form
with st.form("fraud_form"):
  st.subheader("Transaction Details")

  col1, col2 = st.columns(2)

  with col1:
    step = st.number_input(
        "Step (Time Hour)", min_value=1, max_value=744, value=1
    )
    trans_type = st.selectbox(
        "Transaction Type",
        ["TRANSFER", "CASH_OUT", "CASH_IN", "PAYMENT", "DEBIT"],
    )
    amount = st.number_input(
        "Transaction Amount", min_value=0.0, value=50000.0, step=1000.0
    )

  with col2:
    oldbalanceOrg = st.number_input(
        "Sender Initial Balance", min_value=0.0, value=50000.0, step=1000.0
    )
    newbalanceOrig = st.number_input(
        "Sender New Balance", min_value=0.0, value=0.0, step=1000.0
    )
    oldbalanceDest = st.number_input(
        "Receiver Initial Balance", min_value=0.0, value=0.0, step=1000.0
    )
    newbalanceDest = st.number_input(
        "Receiver New Balance", min_value=0.0, value=50000.0, step=1000.0
    )

  submit_button = st.form_submit_button(label="Evaluate Transaction")

# When the button is clicked, run the model prediction directly
if submit_button:
  # Create a DataFrame matching the model's expected input columns
  input_data = pd.DataFrame({
      "step": [step],
      "type": [trans_type],
      "amount": [amount],
      "oldbalanceOrg": [oldbalanceOrg],
      "newbalanceOrig": [newbalanceOrig],
      "oldbalanceDest": [oldbalanceDest],
      "newbalanceDest": [newbalanceDest],
  })

  try:
    # Run predictions directly using the loaded model object
    prediction = model.predict(input_data)[0]

    # Get probabilities if the model supports it
    if hasattr(model, "predict_proba"):
      probabilities = model.predict_proba(input_data)[0]
      legit_probability = round(probabilities[0] * 100, 2)
      fraud_probability = round(probabilities[1] * 100, 2)
    else:
      fraud_probability = 100.0 if prediction == 1 else 0.0
      legit_probability = 100.0 - fraud_probability

    # Determine risk level
    if fraud_probability > 75:
      risk_level = "High"
    elif fraud_probability > 40:
      risk_level = "Moderate"
    else:
      risk_level = "Low"

    st.markdown("---")
    st.subheader("Evaluation Results")

    # Display Alert Banner
    if prediction == 1 or prediction == "Fraudulent":
      st.error("🚨 **Alert: Fraudulent Transaction Detected!**")
    else:
      st.success("✅ **Transaction is Legitimate.**")

    # Side-by-side metrics display
    col_m1, col_m2 = st.columns(2)
    with col_m1:
      st.metric(label="Legitimate Probability", value=f"{legit_probability}%")
    with col_m2:
      st.metric(label="Fraud Probability Score", value=f"{fraud_probability}%")

    st.info(f"Risk Level: **{risk_level}**")

  except Exception as e:
    st.error(f"An error occurred during prediction: {e}")