# Import libraries
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc

warnings.filterwarnings("ignore")

print("========== NaijaGuard AI: Model Training & Evaluation ==========")

# Get project directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "PS_20174392717_1491204439457.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load Dataset
print("\n[1/6] Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"Dataset loaded successfully with shape: {df.shape}")

# 2. Feature Engineering & Selection
print("\n[2/6] Preparing features and target...")
drop_cols = ["nameOrig", "nameDest", "isFlaggedFraud"]
df_model = df.drop(columns=[col for col in drop_cols if col in df.columns])

# Define features (X) and target (y)
X = df_model.drop(columns=["isFraud"])
y = df_model["isFraud"]

# Identify numerical and categorical columns
numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

print(f"Numerical features: {numerical_cols}")
print(f"Categorical features: {categorical_cols}")

# 3. Train/Test Split (Stratified to maintain class proportion due to rare fraud cases)
print("\n[3/6] Splitting data into training and test sets (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# 4. Preprocessing Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ]
)

# 5. Model Training
print("\n[4/6] Training models...")

# Baseline Model: Logistic Regression
print("-> Training Logistic Regression (Baseline)...")
lr_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000))
])
lr_pipeline.fit(X_train, y_train)

# Advanced Model: Random Forest (handling class imbalance with class_weight='balanced')
print("-> Training Random Forest Classifier...")
rf_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100, 
        max_depth=15, 
        class_weight="balanced", 
        random_state=42, 
        n_jobs=-1
    ))
])
rf_pipeline.fit(X_train, y_train)

# 6. Model Evaluation
print("\n[5/6] Evaluating models on test set...")

models = {
    "Logistic Regression (Baseline)": lr_pipeline,
    "Random Forest": rf_pipeline
}

best_model = None
best_score = 0.0
best_model_name = ""

for name, model in models.items():
    print(f"\n--- Evaluation for {name} ---")
    y_proba = model.predict_proba(X_test)[:, 1]
    
    roc_auc = roc_auc_score(y_test, y_proba)
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall, precision)
    
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    print(f"PR-AUC Score:  {pr_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_proba >= 0.5, zero_division=0))
    
    if pr_auc > best_score:
        best_score = pr_auc
        best_model = model
        best_model_name = name

# 7. Save the Best Model
print(f"\n[6/6] Saving best model ({best_model_name}) to disk...")
model_path = MODEL_DIR / "fraud_detection_model.pkl"
joblib.dump(best_model, model_path)
print(f"Model saved successfully at: {model_path}")

print("\n========== Model Training Completed Successfully! ==========")