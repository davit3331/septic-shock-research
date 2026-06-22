import os
import re
import time
import pandas as pd
from google import genai
from sklearn.metrics import classification_report

from pathlib import Path

#########################################################
#### AGENT A — DATA-DRIVEN + FEW-SHOT
#### FP-Growth rules + raw vitals + few-shot examples
#### NO clinical English interpretation (no Simran context)
#### Same patients as document 30 (no shuffle, Patient 5, 9, 11...)
#########################################################

API_KEY = os.getenv("GEMINI_API_KEY")

MODEL_NAME = "gemini-2.5-flash"

# Repo root: walk up until we find requirements.txt. Works no matter where the
# repo is cloned or how deeply this script is nested. See README "File Paths".
ROOT_DIR = next(p for p in Path(__file__).resolve().parents if (p / "requirements.txt").exists())

CSV_PATH = ROOT_DIR / "data" / "processed" / "physionet_balanced.csv"
RULES_PATH = ROOT_DIR / "outputs" / "association_rules" / "PhysioNet_sepsis_rules.csv"

MAX_PATIENTS = 100
MAX_TIMESTEPS_PER_PATIENT = 48
N_SHOT = 1
SLEEP_SECONDS = 2.0

run_timestamp_file = time.strftime("%Y%m%d_%H%M%S")
OUTPUT_CSV = f"gemini_agentA_fewshot_results_{run_timestamp_file}.csv"
REPORT_FILE = f"gemini_agentA_fewshot_report_{run_timestamp_file}.txt"

if not API_KEY:
    raise ValueError("Missing GEMINI_API_KEY.")

client = genai.Client(api_key=API_KEY)


# =========================
# PHYSIONET THRESHOLDS
# From decision tree analysis
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
df["Patient_ID"] = df["Patient_ID"].astype(int)

required_cols = ["Patient_ID", "Hour", "SepsisLabel"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Required column '{col}' not found in CSV.")

df = df.sort_values(["Patient_ID", "Hour"]).reset_index(drop=True)


# =========================
# LOAD FP-GROWTH RULES
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


# =========================
# DISCRETIZE — AGGREGATED ACROSS ALL TIMESTEPS
# =========================
def discretize_patient(patient_df):
    categories = set()

    hr = patient_df["HR"].max() if "HR" in patient_df.columns else None
    temp = patient_df["Temp"].max() if "Temp" in patient_df.columns else None
    resp = patient_df["Resp"].max() if "Resp" in patient_df.columns else None
    o2sat = patient_df["O2Sat"].min() if "O2Sat" in patient_df.columns else None

    if pd.notna(hr):
        categories.add("HR_high" if hr > THRESHOLDS["HR"] else "HR_normal")
    if pd.notna(temp):
        categories.add("Temp_high" if temp > THRESHOLDS["Temp"] else "Temp_normal")
    if pd.notna(resp):
        categories.add("Resp_high" if resp > THRESHOLDS["Resp"] else "Resp_normal")
    if pd.notna(o2sat):
        categories.add("O2Sat_low" if o2sat < THRESHOLDS["O2Sat"] else "O2Sat_normal")

    return categories


# =========================
# MATCH RULES
# =========================
def get_matching_rules(patient_categories, top_n=5):
    matches = [r for r in RULES if r["antecedents"].issubset(patient_categories)]
    return sorted(matches, key=lambda r: r["confidence"], reverse=True)[:top_n]


def format_rules_block(matching_rules, patient_categories):
    if not matching_rules:
        return (
            f"Variable profile: {', '.join(sorted(patient_categories))}\n\n"
            "No sepsis-associated patterns found. "
            "This patient's aggregated vitals do not match any of the 19 data-mined rules. "
            "Predict 0 unless the raw vitals below strongly suggest otherwise."
        )
    lines = [f"Variable profile: {', '.join(sorted(patient_categories))}", ""]
    lines.append("Matched rules (mined from 5,864 ICU patients via FP-Growth):")
    for i, rule in enumerate(matching_rules, 1):
        antecedents_str = " + ".join(sorted(rule["antecedents"]))
        lines.append(
            f"  Rule {i}: {antecedents_str} → Sepsis "
            f"(confidence: {rule['confidence']*100:.1f}%, support: {rule['support']*100:.1f}%)"
        )
    lines.append("")
    lines.append(
        "Confidence = % of ICU patients with these vital signs who had sepsis. "
        "Support = % of all ICU patients this rule applies to."
    )
    return "\n".join(lines)


# =========================
# RAW VITALS FORMATTER (no clinical English)
# =========================
VITAL_COLS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "Age", "Gender", "HospAdmTime", "ICULOS"]

def format_raw_vitals(row):
    lines = []
    for col in VITAL_COLS:
        if col in row.index:
            val = row[col]
            if pd.isna(val):
                lines.append(f"{col}: Not recorded")
            else:
                lines.append(f"{col}: {val}")
    return "\n".join(lines)


# =========================
# FEW-SHOT EXAMPLE BUILDERS — RAW VITALS ONLY (no clinical English)
# =========================
def build_fewshot_examples(df, grouped, exclude_ids, n_per_class=1):
    patient_labels = df.groupby("Patient_ID")["SepsisLabel"].max()
    no_sepsis_ids = patient_labels[(patient_labels == 0) & (~patient_labels.index.isin(exclude_ids))].index.tolist()
    sepsis_ids = patient_labels[(patient_labels == 1) & (~patient_labels.index.isin(exclude_ids))].index.tolist()
    example_blocks = []

    for i in range(min(n_per_class, len(no_sepsis_ids))):
        ex_id = no_sepsis_ids[i]
        ex_df = grouped.get_group(ex_id).sort_values("Hour")
        if len(ex_df) > MAX_TIMESTEPS_PER_PATIENT:
            ex_df = ex_df.tail(MAX_TIMESTEPS_PER_PATIENT)
        hour_blocks = []
        for _, row in ex_df.iterrows():
            hour = row.get("Hour", "?")
            vitals = format_raw_vitals(row)
            hour_blocks.append(f"--- Hour {hour} ---\n{vitals}")
        example_blocks.append(f"EXAMPLE (No Sepsis — correct answer is 0):\n" + "\n\n".join(hour_blocks) + "\nAnswer: 0")

    for i in range(min(n_per_class, len(sepsis_ids))):
        ex_id = sepsis_ids[i]
        ex_df = grouped.get_group(ex_id).sort_values("Hour")
        if len(ex_df) > MAX_TIMESTEPS_PER_PATIENT:
            ex_df = ex_df.tail(MAX_TIMESTEPS_PER_PATIENT)
        hour_blocks = []
        for _, row in ex_df.iterrows():
            hour = row.get("Hour", "?")
            vitals = format_raw_vitals(row)
            hour_blocks.append(f"--- Hour {hour} ---\n{vitals}")
        example_blocks.append(f"EXAMPLE (Sepsis — correct answer is 1):\n" + "\n\n".join(hour_blocks) + "\nAnswer: 1")

    return "\n\n" + "="*40 + "\n\n".join(example_blocks) + "\n\n" + "="*40 + "\n\n"


def build_fewshot_examples_final(df, grouped, exclude_ids, n_per_class=1):
    patient_labels = df.groupby("Patient_ID")["SepsisLabel"].max()
    no_sepsis_ids = patient_labels[(patient_labels == 0) & (~patient_labels.index.isin(exclude_ids))].index.tolist()
    sepsis_ids = patient_labels[(patient_labels == 1) & (~patient_labels.index.isin(exclude_ids))].index.tolist()
    example_blocks = []

    for i in range(min(n_per_class, len(no_sepsis_ids))):
        ex_id = no_sepsis_ids[i]
        ex_df = grouped.get_group(ex_id).sort_values("Hour")
        last = ex_df.iloc[-1]
        vitals = format_raw_vitals(last)
        example_blocks.append(f"EXAMPLE (No Sepsis — correct answer is 0):\n{vitals}\nAnswer: 0")

    for i in range(min(n_per_class, len(sepsis_ids))):
        ex_id = sepsis_ids[i]
        ex_df = grouped.get_group(ex_id).sort_values("Hour")
        last = ex_df.iloc[-1]
        vitals = format_raw_vitals(last)
        example_blocks.append(f"EXAMPLE (Sepsis — correct answer is 1):\n{vitals}\nAnswer: 1")

    return "\n\n" + "="*40 + "\n\n".join(example_blocks) + "\n\n" + "="*40 + "\n\n"


# =========================
# PROMPT BUILDERS
# =========================
def build_full_timeseries_prompt(patient_df, patient_categories, matching_rules, few_shot_block):
    if len(patient_df) > MAX_TIMESTEPS_PER_PATIENT:
        patient_df = patient_df.tail(MAX_TIMESTEPS_PER_PATIENT)

    patient_id = patient_df["Patient_ID"].iloc[0]
    rules_block = format_rules_block(matching_rules, patient_categories)

    hour_blocks = []
    for _, row in patient_df.iterrows():
        hour = row.get("Hour", "?")
        vitals = format_raw_vitals(row)
        hour_blocks.append(f"--- Hour {hour} ---\n{vitals}")
    timeseries_text = "\n\n".join(hour_blocks)

    prompt = f"""You are a sepsis prediction system using data-driven rules mined from ICU patient records.

Below are examples of patients with and without sepsis:
{few_shot_block}

Now predict this new patient:
Patient ID: {patient_id}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DATA-DRIVEN RULES (Agent A):
{rules_block}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Raw ICU time-series (one block per hour):
{timeseries_text}

Task:
Using the examples, the data-driven rules above, and the raw vitals, predict whether this patient will develop sepsis.

Thresholds used for rule matching (from decision tree analysis):
- HR > 127 = high, else normal
- Temp > 38.45°C = high, else normal
- Resp > 20 breaths/min = high, else normal
- O2Sat < 92.6% = low, else normal

Rules:
- Respond with ONLY one label: 0 or 1
- 1 means Sepsis
- 0 means No Sepsis
- Do not explain your answer""".strip()

    return prompt


def build_final_timestep_prompt(patient_df, patient_categories, matching_rules, few_shot_block):
    last = patient_df.iloc[-1]
    patient_id = last["Patient_ID"]
    vitals = format_raw_vitals(last)
    rules_block = format_rules_block(matching_rules, patient_categories)

    prompt = f"""You are a sepsis prediction system using data-driven rules mined from ICU patient records.

Below are examples of patients with and without sepsis:
{few_shot_block}

Now predict this new patient:
Patient ID: {patient_id}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DATA-DRIVEN RULES (Agent A):
{rules_block}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Final recorded vitals:
{vitals}

Task:
Using the examples, the data-driven rules above, and the raw vitals, predict whether this patient has sepsis.

Thresholds used for rule matching (from decision tree analysis):
- HR > 127 = high, else normal
- Temp > 38.45°C = high, else normal
- Resp > 20 breaths/min = high, else normal
- O2Sat < 92.6% = low, else normal

Rules:
- Respond with ONLY one label: 0 or 1
- 1 means Sepsis
- 0 means No Sepsis
- Do not explain your answer""".strip()

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

print(f"\nRunning Agent A + Few-Shot on {len(patient_ids)} patients...")
print(f"Model: {MODEL_NAME} | Rules: {len(RULES)} | N_SHOT: {N_SHOT} | No clinical context\n")

for idx, patient_id in enumerate(patient_ids, start=1):
    patient_df = grouped.get_group(patient_id).sort_values("Hour")
    true_label = int(patient_df["SepsisLabel"].max())

    patient_categories = discretize_patient(patient_df)
    matching_rules = get_matching_rules(patient_categories)

    print(f"  → Categories: {patient_categories} | Rules matched: {len(matching_rules)}")

    exclude = set(patient_ids)
    full_few_shot = build_fewshot_examples(df, grouped, exclude, n_per_class=N_SHOT)
    final_few_shot = build_fewshot_examples_final(df, grouped, exclude, n_per_class=N_SHOT)

    full_prompt = build_full_timeseries_prompt(patient_df, patient_categories, matching_rules, full_few_shot)
    try:
        full_pred, full_raw = call_gemini(full_prompt)
    except Exception as e:
        full_pred, full_raw = None, f"ERROR: {e}"
    time.sleep(SLEEP_SECONDS)

    final_prompt = build_final_timestep_prompt(patient_df, patient_categories, matching_rules, final_few_shot)
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
    f.write("GEMINI AGENT A + FEW-SHOT — REPORT\n")
    f.write("=" * 60 + "\n")
    f.write(f"Model:                  {MODEL_NAME}\n")
    f.write(f"Dataset:                {CSV_PATH}\n")
    f.write(f"Prompt Type:            Agent A + Few-Shot (no clinical context)\n")
    f.write(f"N_SHOT:                 {N_SHOT}\n")
    f.write(f"Clinical Context:       NONE\n")
    f.write(f"FP-Growth Rules:        {len(RULES)}\n")
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