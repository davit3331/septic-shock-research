import os
import re
import ast
import time
import pandas as pd
from google import genai
from sklearn.metrics import classification_report

#########################################################
#### AGENT A — RULE-AUGMENTED GEMINI SEPSIS PREDICTION
#### Upgrade from zero-shot contextualized:
#### Adds FP-Growth rule matching to the prompt context.
#########################################################

from google.colab import userdata
API_KEY = userdata.get("GEMINI_API_KEY")

MODEL_NAME = "gemini-2.5-flash"

CSV_PATH = "Dataset_balanced.csv"
RULES_PATH = "PhysioNet_sepsis_rules.csv"  # your FP-Growth rules file

MAX_PATIENTS = 100
MAX_TIMESTEPS_PER_PATIENT = 48
SLEEP_SECONDS = 2.0

run_timestamp_file = time.strftime("%Y%m%d_%H%M%S")
OUTPUT_CSV = f"gemini_agentA_results_{run_timestamp_file}.csv"
REPORT_FILE = f"gemini_agentA_report_{run_timestamp_file}.txt"

if not API_KEY:
    raise ValueError("Missing GEMINI_API_KEY.")

client = genai.Client(api_key=API_KEY)


# =========================
# PHYSIONET THRESHOLDS
# From your decision_tree_thresholds.py output
# =========================
THRESHOLDS = {
    "Temp":   38.45,
    "Resp":   20.14,
    "HR":    126.97,
    "O2Sat":  92.60,
}


# =========================
# LOAD DATA
# =========================
df = pd.read_csv(CSV_PATH)
df = df.set_index("Patient_ID")
df = df.groupby("Patient_ID").ffill()
df = df.groupby("Patient_ID").bfill()
df = df.reset_index()


# =========================
# LOAD & PARSE FP-GROWTH RULES
# =========================
def load_rules(path):
    """
    Load FP-Growth rules CSV and parse frozenset strings into Python sets.
    Returns a list of dicts: {antecedents: set, confidence: float, support: float}
    """
    rules_df = pd.read_csv(path)
    parsed = []
    for _, row in rules_df.iterrows():
        # Parse frozenset string like "frozenset({'Temp_high', 'Resp_high'})" into a Python set
        raw = row["antecedents"]
        # Extract the inner content between frozenset({ and })
        inner = re.search(r"frozenset\(\{(.+?)\}\)", raw)
        if inner:
            items_str = inner.group(1)
            # Parse each quoted item
            items = re.findall(r"'([^']+)'", items_str)
            parsed.append({
                "antecedents": set(items),
                "confidence": float(row["confidence"]),
                "support": float(row["support"]),
            })
    return parsed

RULES = load_rules(RULES_PATH)
print(f"Loaded {len(RULES)} FP-Growth rules.")


# =========================
# DISCRETIZE ONE ROW
# Uses PhysioNet decision tree thresholds
# =========================
def discretize_row(row):
    """
    Convert a row's raw vitals into category labels using PhysioNet thresholds.
    Returns a set of active categories e.g. {'Temp_high', 'Resp_high', 'O2Sat_low'}
    """
    categories = set()

    def get(col):
        val = row.get(col, None)
        if val is None or pd.isna(val):
            return None
        return float(val)

    # Temp
    temp = get("Temp")
    if temp is not None:
        categories.add("Temp_high" if temp > THRESHOLDS["Temp"] else "Temp_normal")

    # Resp
    resp = get("Resp")
    if resp is not None:
        categories.add("Resp_high" if resp > THRESHOLDS["Resp"] else "Resp_normal")

    # HR
    hr = get("HR")
    if hr is not None:
        categories.add("HR_high" if hr > THRESHOLDS["HR"] else "HR_normal")

    # O2Sat — note: low is BELOW threshold
    o2sat = get("O2Sat")
    if o2sat is not None:
        categories.add("O2Sat_low" if o2sat < THRESHOLDS["O2Sat"] else "O2Sat_normal")

    return categories


# =========================
# MATCH RULES FOR A PATIENT
# =========================
def get_matching_rules(patient_categories, top_n=5):
    """
    Given a set of category labels for a patient (e.g. {'Temp_high', 'O2Sat_low'}),
    find all FP-Growth rules whose antecedents are a subset of the patient's categories.
    Returns top N matches sorted by confidence descending.
    """
    matches = []
    for rule in RULES:
        if rule["antecedents"].issubset(patient_categories):
            matches.append(rule)

    # Sort by confidence descending, take top N
    matches = sorted(matches, key=lambda r: r["confidence"], reverse=True)
    return matches[:top_n]


def format_rules_for_prompt(matching_rules, patient_categories):
    """
    Format matching rules into readable text for the prompt.
    """
    if not matching_rules:
        return "No association rules matched this patient's variable profile."

    lines = ["Agent A — Data Analysis Rules (from FP-Growth mining on 5,864 ICU patients):"]
    lines.append(f"Patient variable profile: {', '.join(sorted(patient_categories))}")
    lines.append("")

    for i, rule in enumerate(matching_rules, 1):
        antecedents_str = " + ".join(sorted(rule["antecedents"]))
        confidence_pct = rule["confidence"] * 100
        support_pct = rule["support"] * 100
        lines.append(
            f"  Rule {i}: {antecedents_str} → Sepsis "
            f"(confidence: {confidence_pct:.1f}%, support: {support_pct:.1f}%)"
        )

    lines.append("")
    lines.append(
        "These rules were mined from real ICU data. "
        "Higher confidence means this combination of variables appeared with sepsis more frequently. "
        "Use these rules as supporting evidence — they do not override clinical judgment."
    )

    return "\n".join(lines)


# =========================
# HELPER FUNCTIONS
# (same as your existing script)
# =========================
def clean_value(value):
    if pd.isna(value):
        return None
    return value

def interpret_hr(val):
    if val is None: return "Heart Rate: Not recorded."
    val = float(val)
    if val < 60: return f"Heart Rate: {val:.1f} bpm — BRADYCARDIC. Low HR in sepsis can indicate late-stage cardiovascular decompensation."
    elif 60 <= val <= 100: return f"Heart Rate: {val:.1f} bpm — NORMAL. Normal HR does not rule out early sepsis."
    elif 100 < val <= 120: return f"Heart Rate: {val:.1f} bpm — ELEVATED (mild tachycardia). Sensitive early warning sign of systemic infection."
    else: return f"Heart Rate: {val:.1f} bpm — CRITICALLY ELEVATED (severe tachycardia). Strongly associated with sepsis and hemodynamic instability."

def interpret_o2sat(val):
    if val is None: return "Oxygen Saturation: Not recorded."
    val = float(val)
    if val >= 95: return f"Oxygen Saturation: {val:.1f}% — NORMAL. Note SpO2 may overestimate true saturation in sepsis."
    elif 90 <= val < 95: return f"Oxygen Saturation: {val:.1f}% — LOW (mild hypoxemia). Associated with increased risk of organ dysfunction."
    else: return f"Oxygen Saturation: {val:.1f}% — CRITICALLY LOW (severe hypoxemia). Strongly associated with organ failure and mortality."

def interpret_temp(val):
    if val is None: return "Temperature: Not recorded."
    val = float(val)
    if val < 36.0: return f"Temperature: {val:.1f}°C — HYPOTHERMIC. Associated with ~47% mortality in sepsis vs ~22% with fever. Strong indicator of septic shock."
    elif 36.0 <= val <= 38.2: return f"Temperature: {val:.1f}°C — NORMAL. Does not rule out sepsis."
    elif 38.2 < val <= 39.5: return f"Temperature: {val:.1f}°C — FEVER (mild). Supports suspicion of systemic infection."
    else: return f"Temperature: {val:.1f}°C — FEVER (high). Strongly associated with systemic infection and sepsis physiology."

def interpret_sbp(val):
    if val is None: return "Systolic BP: Not recorded."
    val = float(val)
    if val <= 90: return f"Systolic BP: {val:.1f} mmHg — CRITICALLY LOW. Strong indicator of septic shock."
    elif 90 < val <= 100: return f"Systolic BP: {val:.1f} mmHg — LOW (qSOFA threshold met ≤100 mmHg). Associated with poor outcomes."
    elif 100 < val <= 120: return f"Systolic BP: {val:.1f} mmHg — BORDERLINE LOW. Warrants monitoring."
    elif 120 < val <= 140: return f"Systolic BP: {val:.1f} mmHg — NORMAL."
    else: return f"Systolic BP: {val:.1f} mmHg — ELEVATED. Less concerning for septic shock."

def interpret_map(val):
    if val is None: return "Mean Arterial Pressure: Not recorded."
    val = float(val)
    if val < 65: return f"Mean Arterial Pressure: {val:.1f} mmHg — CRITICALLY LOW (septic shock threshold). MAP <65 meets clinical definition of septic shock."
    elif 65 <= val <= 70: return f"Mean Arterial Pressure: {val:.1f} mmHg — LOW (borderline shock threshold). High risk for organ hypoperfusion."
    elif 70 < val <= 82: return f"Mean Arterial Pressure: {val:.1f} mmHg — ACCEPTABLE (lowest mortality range 70-82 mmHg)."
    elif 82 < val <= 100: return f"Mean Arterial Pressure: {val:.1f} mmHg — NORMAL."
    else: return f"Mean Arterial Pressure: {val:.1f} mmHg — ELEVATED."

def interpret_dbp(val):
    if val is None: return "Diastolic BP: Not recorded."
    val = float(val)
    if val < 40: return f"Diastolic BP: {val:.1f} mmHg — CRITICALLY LOW. Severely impaired vascular tone."
    elif 40 <= val < 60: return f"Diastolic BP: {val:.1f} mmHg — LOW (below ~60 mmHg ICU mortality change-point)."
    elif 60 <= val <= 80: return f"Diastolic BP: {val:.1f} mmHg — NORMAL."
    else: return f"Diastolic BP: {val:.1f} mmHg — ELEVATED."

def interpret_resp(val):
    if val is None: return "Respiratory Rate: Not recorded."
    val = float(val)
    if val < 12: return f"Respiratory Rate: {val:.1f} breaths/min — LOW (bradypnea). May indicate CNS depression."
    elif 12 <= val <= 20: return f"Respiratory Rate: {val:.1f} breaths/min — NORMAL."
    elif 20 < val < 22: return f"Respiratory Rate: {val:.1f} breaths/min — BORDERLINE ELEVATED. Approaching qSOFA threshold of 22/min."
    elif 22 <= val <= 30: return f"Respiratory Rate: {val:.1f} breaths/min — ELEVATED (qSOFA threshold met ≥22/min). Tachypnea is an early warning sign."
    else: return f"Respiratory Rate: {val:.1f} breaths/min — CRITICALLY ELEVATED. Strongly associated with sepsis severity."

def interpret_age(val):
    if val is None: return "Age: Not recorded."
    val = float(val)
    if val < 45: return f"Age: {val:.0f} years. Young adult — lower baseline sepsis risk."
    elif 45 <= val < 65: return f"Age: {val:.0f} years. Middle-aged — moderate sepsis risk."
    elif 65 <= val < 85: return f"Age: {val:.0f} years. Elderly — significantly higher sepsis risk due to immunosenescence."
    else: return f"Age: {val:.0f} years. Very elderly (≥85) — highest risk group, sepsis mortality ~38%."

def interpret_gender(val):
    if val is None: return "Gender: Not recorded."
    val = float(val)
    if val == 1: return "Gender: Male. ~1.3x higher sepsis incidence than females, longer ICU stay and higher mortality."
    else: return "Gender: Female. Lower sepsis incidence but outcomes vary by age and pathogen."

def interpret_hospadmtime(val):
    if val is None: return "Time from Hospital to ICU Admission: Not recorded."
    val = float(val)
    if val < 0: return f"Time from Hospital to ICU Admission: {abs(val):.0f} hours before hospital admission (direct ICU admit)."
    elif val == 0: return "Time from Hospital to ICU Admission: Admitted directly to ICU."
    elif val <= 6: return f"Time from Hospital to ICU Admission: {val:.0f} hours — early transfer."
    else: return f"Time from Hospital to ICU Admission: {val:.0f} hours — delayed transfer. Associated with higher mortality in severe sepsis."

def interpret_iculos(val):
    if val is None: return "ICU Length of Stay: Not recorded."
    val = float(val)
    if val <= 24: return f"ICU Length of Stay: {val:.0f} hours — early stay."
    elif 24 < val <= 72: return f"ICU Length of Stay: {val:.0f} hours — moderate stay (1-3 days). Increasing infection exposure."
    elif 72 < val <= 168: return f"ICU Length of Stay: {val:.0f} hours — prolonged stay (3-7 days). Higher device-associated infection risk."
    else: return f"ICU Length of Stay: {val:.0f} hours — extended stay (>7 days). Highest ICU-acquired infection risk."

CHOSEN_COLS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "Age", "Gender", "HospAdmTime", "ICULOS"]

col_map = {
    "HR": interpret_hr, "O2Sat": interpret_o2sat, "Temp": interpret_temp,
    "SBP": interpret_sbp, "MAP": interpret_map, "DBP": interpret_dbp,
    "Resp": interpret_resp, "Age": interpret_age, "Gender": interpret_gender,
    "HospAdmTime": interpret_hospadmtime, "ICULOS": interpret_iculos,
}

def build_contextualized_vitals(row, chosen_cols):
    lines = []
    for col in chosen_cols:
        if col in col_map:
            raw = clean_value(row[col]) if col in row.index else None
            lines.append(col_map[col](raw))
    return "\n".join(lines)


# =========================
# PROMPT BUILDERS — AGENT A UPGRADED
# Key change: rule context block added to each prompt
# =========================

def build_full_timeseries_prompt(patient_df: pd.DataFrame) -> str:
    if len(patient_df) > MAX_TIMESTEPS_PER_PATIENT:
        patient_df = patient_df.tail(MAX_TIMESTEPS_PER_PATIENT)

    patient_id = patient_df["Patient_ID"].iloc[0]
    available_cols = [c for c in CHOSEN_COLS if c in patient_df.columns]

    # --- AGENT A: get rules from LAST timestep (most recent state)
    last_row = patient_df.iloc[-1]
    patient_categories = discretize_row(last_row)
    matching_rules = get_matching_rules(patient_categories)
    rules_block = format_rules_for_prompt(matching_rules, patient_categories)

    hour_blocks = []
    for _, row in patient_df.iterrows():
        hour = row.get("Hour", "?")
        vitals = build_contextualized_vitals(row, available_cols)
        hour_blocks.append(f"--- Hour {hour} ---\n{vitals}")

    timeseries_text = "\n\n".join(hour_blocks)

    prompt = f"""
You are assisting with a sepsis prediction experiment.

Below is the time-series data for ONE patient. Each variable has been interpreted
using evidence-based clinical thresholds. Each block represents one hour of ICU monitoring.

Patient ID: {patient_id}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{rules_block}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Time-series:
{timeseries_text}

Task:
Based on the full time-series data AND the association rules above, predict whether
this patient will develop sepsis at any point.

Important:
- Respond with ONLY one label: 0 or 1
- 1 means Sepsis
- 0 means No Sepsis
- Do not explain your answer
- Not all ICU patients develop sepsis. Normal ICU vitals may appear abnormal
  compared to healthy patients — consider trends and combinations, not individual values.
- Only predict 1 if there is clear evidence of systemic infection and organ
  dysfunction across multiple variables.
    """.strip()

    return prompt


def build_final_timestep_prompt(patient_df: pd.DataFrame) -> str:
    last = patient_df.iloc[-1]
    patient_id = last["Patient_ID"]
    available_cols = [c for c in CHOSEN_COLS if c in last.index]
    vitals = build_contextualized_vitals(last, available_cols)

    # --- AGENT A: get rules for this timestep
    patient_categories = discretize_row(last)
    matching_rules = get_matching_rules(patient_categories)
    rules_block = format_rules_for_prompt(matching_rules, patient_categories)

    prompt = f"""
You are assisting with a sepsis prediction experiment.

Below is the FINAL time-step data for ONE patient. Each variable has been interpreted
using evidence-based clinical thresholds.

Patient ID: {patient_id}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{rules_block}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Final state:
{vitals}

Task:
Based on this final time-step AND the association rules above, predict whether
this patient has sepsis.

Important:
- Respond with ONLY one label: 0 or 1
- 1 means Sepsis
- 0 means No Sepsis
- Do not explain your answer
    """.strip()

    return prompt


def parse_binary_prediction(text: str):
    if text is None: return None
    text = text.strip()
    if text == "0": return 0
    if text == "1": return 1
    match = re.search(r"\b([01])\b", text)
    if match: return int(match.group(1))
    return None

def call_gemini(prompt: str) -> tuple:
    response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
    raw_text = response.text if hasattr(response, "text") else str(response)
    pred = parse_binary_prediction(raw_text)
    return pred, raw_text


# =========================
# RUN EXPERIMENT
# =========================
results = []
grouped = df.groupby("Patient_ID")
patient_ids = list(grouped.groups.keys())[:MAX_PATIENTS]

print(f"\nRunning Agent A Rule-Augmented experiment on {len(patient_ids)} patients...")
print(f"Model: {MODEL_NAME} | Rules loaded: {len(RULES)}\n")

for idx, patient_id in enumerate(patient_ids, start=1):
    patient_df = grouped.get_group(patient_id).sort_values("Hour")
    true_label = int(patient_df["SepsisLabel"].max())

    # Full timeseries
    full_prompt = build_full_timeseries_prompt(patient_df)
    try:
        full_pred, full_raw = call_gemini(full_prompt)
    except Exception as e:
        full_pred, full_raw = None, f"ERROR: {e}"
    time.sleep(SLEEP_SECONDS)

    # Final timestep
    final_prompt = build_final_timestep_prompt(patient_df)
    try:
        final_pred, final_raw = call_gemini(final_prompt)
    except Exception as e:
        final_pred, final_raw = None, f"ERROR: {e}"

    results.append({
        "Patient_ID": patient_id,
        "TrueLabel": true_label,
        "FullPromptPrediction": full_pred,
        "FinalPromptPrediction": final_pred,
        "FullPromptRaw": full_raw,
        "FinalPromptRaw": final_raw,
    })

    print(f"[{idx}/{len(patient_ids)}] Patient {patient_id} | Truth={true_label} | Full={full_pred} | Final={final_pred}")
    time.sleep(SLEEP_SECONDS)


# Save results
results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_CSV, index=False)
print(f"\nDone. Results saved to {OUTPUT_CSV}")


# =========================
# METRICS
# =========================
valid_full = results_df.dropna(subset=["FullPromptPrediction"])
valid_final = results_df.dropna(subset=["FinalPromptPrediction"])

with open(REPORT_FILE, "w") as f:
    f.write("=" * 60 + "\n")
    f.write("GEMINI AGENT A — RULE-AUGMENTED REPORT\n")
    f.write("=" * 60 + "\n")
    f.write(f"Model:                  {MODEL_NAME}\n")
    f.write(f"Dataset:                {CSV_PATH}\n")
    f.write(f"Prompt Type:            Agent A Rule-Augmented\n")
    f.write(f"FP-Growth Rules Used:   {len(RULES)}\n")
    f.write(f"MAX_PATIENTS:           {MAX_PATIENTS}\n")
    f.write(f"MAX_TIMESTEPS:          {MAX_TIMESTEPS_PER_PATIENT}\n")
    f.write(f"Total Attempted:        {len(results_df)}\n")
    f.write(f"Valid Full Predictions: {len(valid_full)}\n")
    f.write(f"Valid Final Predictions:{len(valid_final)}\n")
    f.write("=" * 60 + "\n")

    if len(valid_full) > 0:
        full_report = classification_report(valid_full["TrueLabel"], valid_full["FullPromptPrediction"], zero_division=0)
        print("\nFULL TIMESTEPS CLASSIFICATION REPORT")
        print(full_report)
        f.write("\nFULL TIMESTEPS CLASSIFICATION REPORT\n")
        f.write(full_report + "\n")

    if len(valid_final) > 0:
        final_report = classification_report(valid_final["TrueLabel"], valid_final["FinalPromptPrediction"], zero_division=0)
        print("\nFINAL TIMESTEP CLASSIFICATION REPORT")
        print(final_report)
        f.write("\nFINAL TIMESTEP CLASSIFICATION REPORT\n")
        f.write(final_report + "\n")

print(f"\nReport saved to: {REPORT_FILE}")