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


# =========================
# SECTION 5 — FP-GROWTH RULES + SHAP
# =========================
def load_rules(path):
    rules_df = pd.read_csv(path)
    parsed = []
    for _, row in rules_df.iterrows():
        raw = row["antecedents"]
        inner = re.search(r"frozenset\(\{(.+?)\}\)", raw)
        if inner:
            items = re.findall(r"'([^']+)'", inner.group(1))
            parsed.append({
                "antecedents": set(items),
                "confidence": float(row["confidence"]),
                "support": float(row["support"]),
            })
    return parsed
 
RULES = load_rules(RULES_PATH)
print(f"Loaded {len(RULES)} FP-Growth rules.")
 
THRESHOLDS = {"Temp": 38.45, "Resp": 20.14, "HR": 126.97, "O2Sat": 92.60}
 
def discretize_patient_row(row):
    categories = set()
    if pd.notna(row.get("HR")):
        categories.add("HR_high" if row["HR"] > THRESHOLDS["HR"] else "HR_normal")
    if pd.notna(row.get("Temp")):
        categories.add("Temp_high" if row["Temp"] > THRESHOLDS["Temp"] else "Temp_normal")
    if pd.notna(row.get("Resp")):
        categories.add("Resp_high" if row["Resp"] > THRESHOLDS["Resp"] else "Resp_normal")
    if pd.notna(row.get("O2Sat")):
        categories.add("O2Sat_low" if row["O2Sat"] < THRESHOLDS["O2Sat"] else "O2Sat_normal")
    return categories
 
def get_matching_rules(patient_categories, top_n=5):
    matches = [r for r in RULES if r["antecedents"].issubset(patient_categories)]
    return sorted(matches, key=lambda r: r["confidence"], reverse=True)[:top_n]
 
def format_rules_block(matching_rules, patient_categories):
    if not matching_rules:
        return (
            f"Variable profile: {', '.join(sorted(patient_categories))}\n"
            "No sepsis-associated patterns found in this patient's vitals."
        )
    lines = [f"Variable profile: {', '.join(sorted(patient_categories))}", ""]
    lines.append("Matched FP-Growth rules (mined from 5,864 ICU patients):")
    for i, rule in enumerate(matching_rules, 1):
        antecedents_str = " + ".join(sorted(rule["antecedents"]))
        lines.append(f"  Rule {i}: {antecedents_str} → Sepsis (confidence: {rule['confidence']*100:.1f}%, support: {rule['support']*100:.1f}%)")
    return "\n".join(lines)
 
# SHAP explainer
explainer = shap.TreeExplainer(model1_XGB)
shap_values = explainer.shap_values(X_test)
 
def get_top_shap_features(shap_vals, feature_names, patient_row, top_n=3):
    pairs = list(zip(feature_names, shap_vals, patient_row))
    pairs_sorted = sorted(pairs, key=lambda x: abs(x[1]), reverse=True)
    lines = []
    for feat, shap_val, actual_val in pairs_sorted[:top_n]:
        direction = "toward sepsis" if shap_val > 0 else "away from sepsis"
        lines.append(f"  {feat}={actual_val:.2f} (SHAP={shap_val:+.3f}, {direction})")
    return "\n".join(lines)
 
 
# =========================
# SECTION 6 — BUILD LLM PROMPT
# =========================
def build_prompt(patient_id, xgb_proba, shap_explanation, rules_block, patient_row):
    xgb_pct = xgb_proba * 100
    prediction_label = "SEPSIS" if xgb_proba >= 0.5 else "NO SEPSIS"
 
    vitals_lines = []
    for col in feature_cols:
        val = patient_row[col] if col in patient_row.index else "N/A"
        vitals_lines.append(f"  {col}: {val}")
    vitals_str = "\n".join(vitals_lines)
 
    prompt = f"""You are a clinical AI assistant explaining a sepsis prediction.
 
Patient ID: {patient_id}
 
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ML PREDICTION (XGBoost):
Sepsis probability: {xgb_pct:.1f}%
Prediction: {prediction_label}
 
Top factors driving this prediction (SHAP):
{shap_explanation}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 
DATA-DRIVEN RULES (FP-Growth):
{rules_block}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 
Patient vitals summary:
{vitals_str}
 
Task:
The ML model has predicted {prediction_label} with {xgb_pct:.1f}% confidence.
Based on the SHAP factors and FP-Growth rules above, confirm this prediction.
 
Important:
- Respond with ONLY one label: 0 or 1
- 1 means Sepsis
- 0 means No Sepsis
- Do not explain your answer""".strip()
 
    return prompt
 
 
# =========================
# SECTION 7 — CALL GEMINI + RUN EXPERIMENT
# =========================
def parse_binary_prediction(text):
    if text is None: return None
    text = text.strip()
    if text == "0": return 0
    if text == "1": return 1
    match = re.search(r"\b([01])\b", text)
    if match: return int(match.group(1))
    return None
 
def call_gemini(prompt):
    response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
    raw_text = response.text if hasattr(response, "text") else str(response)
    pred = parse_binary_prediction(raw_text)
    return pred, raw_text
 
results = []
feature_names = X_test.columns.tolist()
xgb_probas = model1_XGB.predict_proba(X_test)[:, 1]

# Limit LLM to first 100 test patients
X_test_llm = X_test.iloc[:100]
y_test_llm = y_test.iloc[:100]
shap_values_llm = shap_values[:100]
xgb_probas_llm = xgb_probas[:100]
 
print(f"\nRunning LLM on 100 test patients...")

for i, (idx, row) in enumerate(X_test_llm.iterrows()):
    patient_id = df.loc[idx, "Patient_ID"]
    true_label = int(y_test_llm.loc[idx])
    xgb_proba = float(xgb_probas_llm[i])
    xgb_pred = int(xgb_proba >= 0.5)

    shap_explanation = get_top_shap_features(shap_values_llm[i], feature_names, row.values)
 
    patient_categories = discretize_patient_row(row)
    matching_rules = get_matching_rules(patient_categories)
    rules_block = format_rules_block(matching_rules, patient_categories)
 
    prompt = build_prompt(patient_id, xgb_proba, shap_explanation, rules_block, row)
 
    try:
        llm_pred, llm_raw = call_gemini(prompt)
    except Exception as e:
        llm_pred, llm_raw = None, f"ERROR: {e}"
 
    results.append({
        "Patient_ID": patient_id,
        "TrueLabel": true_label,
        "XGB_Prediction": xgb_pred,
        "XGB_Probability": round(xgb_proba, 4),
        "LLM_Prediction": llm_pred,
        "LLM_Raw": llm_raw,
    })
 
    print(f"[{i+1}/{len(X_test)}] Patient {patient_id} | Truth={true_label} | XGB={xgb_pred} ({xgb_proba:.0%}) | LLM={llm_pred}")
    time.sleep(SLEEP_SECONDS)
 
 
# =========================
# SECTION 8 — METRICS AND SAVE
# =========================
results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_CSV, index=False)
print(f"\nResults saved to {OUTPUT_CSV}")
 
valid = results_df.dropna(subset=["LLM_Prediction"])
 
with open(REPORT_FILE, "w") as f:
    f.write("=" * 60 + "\n")
    f.write("AGENT A ML REPORT\n")
    f.write("=" * 60 + "\n")
    f.write(f"LLM Model:              {MODEL_NAME}\n")
    f.write(f"XGBoost AUROC:          {auroc_XGB:.3f}\n")
    f.write(f"Random Forest AUROC:    {auroc_RF:.3f}\n")
    f.write(f"Test patients:          {len(X_test)}\n")
    f.write(f"Valid LLM predictions:  {len(valid)}\n")
    f.write("=" * 60 + "\n\n")
 
    f.write("XGBoost Standalone:\n")
    f.write(report_XGB + "\n")
 
    f.write("Random Forest Standalone:\n")
    f.write(report_RF + "\n")
 
    if len(valid) > 0:
        llm_report = classification_report(
            valid["TrueLabel"], valid["LLM_Prediction"], zero_division=0
        )
        print("\nLLM (guided by XGBoost) CLASSIFICATION REPORT")
        print(llm_report)
        f.write("LLM guided by XGBoost:\n")
        f.write(llm_report + "\n")
 
print(f"Report saved to {REPORT_FILE}")