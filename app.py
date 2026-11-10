from flask import Flask, request, jsonify
import pandas as pd
import joblib
from pathlib import Path

# Initialize Flask App
app = Flask(__name__)

# Define paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "fraud_detection_model.pkl"

# Load the trained model pipeline
print("Loading NaijaGuard AI model into memory...")
model = joblib.load(MODEL_PATH)
print("Model loaded successfully!")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_df = pd.DataFrame([data])
        
        prediction = int(model.predict(input_df)[0])
        probabilities = model.predict_proba(input_df)[0]
        
        legit_proba = probabilities[0]
        fraud_proba = probabilities[1]
        
        if fraud_proba > 0.7:
            risk_level = "High Risk"
        elif fraud_proba > 0.3:
            risk_level = "Medium Risk"
        else:
            risk_level = "Low Risk"
            
        # ==========================================
        # PASTE IT HERE (replacing your old return)
        # ==========================================
        return jsonify({
            "prediction": "Fraudulent" if prediction == 1 else "Legitimate",
            "fraud_probability": round(float(fraud_proba) * 100, 2),
            "legitimate_probability": round(float(legit_proba) * 100, 2),
            "risk_level": risk_level,
            "status": "success"
        })
        
    except Exception as e:
        return jsonify({"error": str(e), "status": "failed"}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)