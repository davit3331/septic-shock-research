import os
import re
import time
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from google import genai
from sklearn.metrics import roc_auc_score

API_KEY = os.getenv("GEMINI_API_KEY")

MODEL_NAME = "gemini-2.5-flash"

CSV_PATH = "/Users/davitpiruzyan/Desktop/septic-shock-research-github/data/processed/physionet_balanced.csv"

RULES_PATH = "/Users/davitpiruzyan/Desktop/septic-shock-research-github/outputs/association_rules/PhysioNet_sepsis_rules.csv"

MAX_PATIENTS = 100
MAX_TIMESTEPS_PER_PATIENT = 48
SLEEP_SECONDS = 2.0

run_timestamp_file = time.strftime("%Y%m%d_%H%M%S")

os.makedirs("agentA_ML_output", exist_ok=True)
OUTPUT_CSV = f"agentA_ML_output/gemini_agentA_ML_results_{run_timestamp_file}.csv"
REPORT_FILE = f"agentA_ML_output/gemini_agentA_ML_report_{run_timestamp_file}.txt"

if not API_KEY:
    raise ValueError("Missing GEMINI_API_KEY.")

client = genai.Client(api_key=API_KEY)


# =========================
# LOAD DATA
# =========================
df = pd.read_csv(CSV_PATH)
df = df.set_index("Patient_ID")
df = df.groupby("Patient_ID").ffill()
df = df.groupby("Patient_ID").bfill()
df = df.reset_index()

#aggregate each patients columns to max, except 02Sat to min
agg_funcs = {col: "max" for col in df.columns if col not in ["Patient_ID", "SepsisLabel", "O2Sat", "Hour", "HospAdmTime", "Age", "Gender"]}
agg_funcs["O2Sat"] = "min"
agg_funcs["HospAdmTime"] = "first"
agg_funcs["Age"] = "first"
agg_funcs["Gender"] = "first"
agg_funcs["SepsisLabel"] = "max"

df = df.groupby("Patient_ID").agg(agg_funcs).reset_index()


# fill remaining NaNs with column median
feature_cols = [c for c in df.columns if c not in ["Patient_ID", "SepsisLabel"]]

medians_of_feature_cols = df[feature_cols].median()
df[feature_cols] = df[feature_cols].fillna(medians_of_feature_cols)


##Train/Test Split

X = df.drop(columns=["Patient_ID", "SepsisLabel"], errors="ignore")
y = df["SepsisLabel"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# =========================
# TRAIN MODEL

# Create XGBoost model and Random Forest model with specified hyperparameters
model1_XGB = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model2_RF = RandomForestClassifier(
    n_estimators=300,
    max_depth=5,
    random_state=42
)

# Train both models on the training data

model1_XGB.fit(X_train, y_train)

model2_RF.fit(X_train, y_train)

# =========================
# EVALUATE MODELS
y_pred_XGB = model1_XGB.predict(X_test)
y_pred_RF = model2_RF.predict(X_test)
report_XGB = classification_report(y_test, y_pred_XGB)
report_RF = classification_report(y_test, y_pred_RF)

print("XGBoost Classification Report:\n", report_XGB)
print("Random Forest Classification Report:\n", report_RF)

auroc_XGB = roc_auc_score(y_test, model1_XGB.predict_proba(X_test)[:, 1])
auroc_RF = roc_auc_score(y_test, model2_RF.predict_proba(X_test)[:, 1])
print(f"XGBoost AUROC: {auroc_XGB:.3f}")
print(f"Random Forest AUROC: {auroc_RF:.3f}")
