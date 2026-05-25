import os
import re
import time
import pandas as pd
from google import genai
from sklearn.metrics import classification_report

#########################################################
#### FEW-SHOT + AGENT A RULES
#### Base: document 30 (no shuffle, Patient 5, 9, 11...)
#### Added: FP-Growth rule matching per patient
#########################################################

API_KEY = os.getenv("GEMINI_API_KEY")

MODEL_NAME = "gemini-2.5-flash"

CSV_PATH = "/Users/davitpiruzyan/Desktop/septic-shock-research-github/data/processed/physionet_balanced.csv"
RULES_PATH = "/Users/davitpiruzyan/Desktop/septic-shock-research-github/outputs/association_rules/PhysioNet_sepsis_rules.csv"

MAX_PATIENTS = 100
MAX_TIMESTEPS_PER_PATIENT = 48
N_SHOT = 1
SLEEP_SECONDS = 2.0

run_timestamp_file = time.strftime("%Y%m%d_%H%M%S")
OUTPUT_CSV = f"gemini_fewshotAgentA_results_{run_timestamp_file}.csv"
REPORT_FILE = f"gemini_fewshotAgentA_report_{run_timestamp_file}.txt"

if not API_KEY:
    raise ValueError("Missing GEMINI_API_KEY.")

client = genai.Client(api_key=API_KEY)


# =========================
# PHYSIONET THRESHOLDS
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
            items_str = inner.group(1)
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
# =========================
def discretize_row(row):
    categories = set()

    def get(col):
        val = row.get(col, None)
        if val is None or pd.isna(val):
            return None
        return float(val)

    temp = get("Temp")
    if temp is not None:
        if temp > THRESHOLDS["Temp"]:       categories.add("Temp_high")
        elif temp < 36.0:                   categories.add("Temp_low")
        else:                               categories.add("Temp_normal")

    resp = get("Resp")
    if resp is not None:
        if resp > THRESHOLDS["Resp"]:       categories.add("Resp_high")
        elif resp < 12:                     categories.add("Resp_low")
        else:                               categories.add("Resp_normal")

    hr = get("HR")
    if hr is not None:
        if hr > THRESHOLDS["HR"]:           categories.add("HR_high")
        elif hr < 60:                       categories.add("HR_low")
        else:                               categories.add("HR_normal")

    o2sat = get("O2Sat")
    if o2sat is not None:
        if o2sat < THRESHOLDS["O2Sat"]:     categories.add("O2Sat_low")
        elif o2sat > 99:                    categories.add("O2Sat_high")
        else:                               categories.add("O2Sat_normal")

    return categories


# =========================
# MATCH RULES
# =========================
def get_matching_rules(patient_categories, top_n=5, min_confidence=0.70):
    matches = [r for r in RULES 
               if r["antecedents"].issubset(patient_categories) 
               and r["confidence"] >= min_confidence]
    return sorted(matches, key=lambda r: r["confidence"], reverse=True)[:top_n]

def format_rules_for_prompt(matching_rules, patient_categories):
    if not matching_rules:
        return (
            "Agent A — Data Analysis Rules (from FP-Growth mining on 5,864 ICU patients):\n"
            f"Patient variable profile: {', '.join(sorted(patient_categories))}\n\n"
            "No sepsis-predicting patterns were found in this patient's variable profile. "
            "This patient's vitals do not match any combination of variables associated with sepsis "
            "across 5,864 ICU patients. This is evidence against a sepsis prediction — predict 0 unless "
            "the clinical vitals below provide strong contradicting evidence."
        )

    lines = ["Agent A — Data Analysis Rules (from FP-Growth mining on 5,864 ICU patients):"]
    lines.append(f"Patient variable profile: {', '.join(sorted(patient_categories))}")
    lines.append("")
    for i, rule in enumerate(matching_rules, 1):
        antecedents_str = " + ".join(sorted(rule["antecedents"]))
        lines.append(f"  Rule {i}: {antecedents_str} → Sepsis (confidence: {rule['confidence']*100:.1f}%, support: {rule['support']*100:.1f}%)")
    lines.append("")
    lines.append("These rules were mined from real ICU data. Use as supporting evidence — they do not override clinical judgment.")
    return "\n".join(lines)


# =========================
# HELPER FUNCTIONS
# =========================
def clean_value(value):
    if pd.isna(value):
        return None
    return value

def interpret_hr(val):
    if val is None: return "Heart Rate: Not recorded."
    val = float(val)
    if val < 60: return f"Heart Rate: {val:.1f} bpm — BRADYCARDIC (abnormally low). Low HR in sepsis can indicate late-stage cardiovascular decompensation."
    elif 60 <= val <= 100: return f"Heart Rate: {val:.1f} bpm — NORMAL. Normal HR does not rule out early sepsis."
    elif 100 < val <= 120: return f"Heart Rate: {val:.1f} bpm — ELEVATED — mild tachycardia. Mild tachycardia is a sensitive early warning sign of systemic infection and inflammation."
    else: return f"Heart Rate: {val:.1f} bpm — CRITICALLY ELEVATED — severe tachycardia. Severe tachycardia strongly associated with sepsis and hemodynamic instability."

def interpret_o2sat(val):
    if val is None: return "Oxygen Saturation: Not recorded."
    val = float(val)
    if val >= 95: return f"Oxygen Saturation: {val:.1f}% — NORMAL. Adequate oxygenation. Note SpO2 may overestimate true saturation in sepsis."
    elif 90 <= val < 95: return f"Oxygen Saturation: {val:.1f}% — LOW — mild hypoxemia. Mild hypoxemia. Associated with increased risk of organ dysfunction in sepsis."
    else: return f"Oxygen Saturation: {val:.1f}% — CRITICALLY LOW — severe hypoxemia. Severe hypoxemia strongly associated with organ failure and mortality in sepsis."

def interpret_temp(val):
    if val is None: return "Temperature: Not recorded."
    val = float(val)
    if val < 36.0: return f"Temperature: {val:.1f}°C — HYPOTHERMIC. Hypothermia in sepsis is more dangerous than fever — associated with ~47% mortality vs ~22% with fever. Strong indicator of septic shock."
    elif 36.0 <= val <= 38.2: return f"Temperature: {val:.1f}°C — NORMAL. Normal temperature. Does not rule out sepsis — early sepsis can present without fever."
    elif 38.2 < val <= 39.5: return f"Temperature: {val:.1f}°C — FEVER — mild. Mild fever supports suspicion of systemic infection. Threshold for ICU sepsis fever is typically 38.2°C."
    else: return f"Temperature: {val:.1f}°C — FEVER — high. High fever strongly associated with systemic infection and sepsis physiology."

def interpret_sbp(val):
    if val is None: return "Systolic Blood Pressure: Not recorded."
    val = float(val)
    if val <= 90: return f"Systolic BP: {val:.1f} mmHg — CRITICALLY LOW — severe hypotension. Severe hypotension. Strong indicator of septic shock and circulatory failure."
    elif 90 < val <= 100: return f"Systolic BP: {val:.1f} mmHg — LOW — qSOFA threshold met (≤100 mmHg). Meets qSOFA criteria. Associated with poor outcomes in sepsis."
    elif 100 < val <= 120: return f"Systolic BP: {val:.1f} mmHg — BORDERLINE LOW. Below normal range. Warrants monitoring for declining trend."
    elif 120 < val <= 140: return f"Systolic BP: {val:.1f} mmHg — NORMAL. Normal systolic pressure."
    else: return f"Systolic BP: {val:.1f} mmHg — ELEVATED. Elevated SBP. Less concerning for septic shock."

def interpret_map(val):
    if val is None: return "Mean Arterial Pressure: Not recorded."
    val = float(val)
    if val < 65: return f"Mean Arterial Pressure: {val:.1f} mmHg — CRITICALLY LOW — septic shock threshold. MAP <65 mmHg meets clinical definition of septic shock. Risk rises sharply below this threshold."
    elif 65 <= val <= 70: return f"Mean Arterial Pressure: {val:.1f} mmHg — LOW — borderline shock threshold. At the lower edge of the septic shock threshold. High risk for organ hypoperfusion."
    elif 70 < val <= 82: return f"Mean Arterial Pressure: {val:.1f} mmHg — ACCEPTABLE — lowest mortality range. Within the range associated with lowest 28-day mortality in sepsis patients (70-82 mmHg)."
    elif 82 < val <= 100: return f"Mean Arterial Pressure: {val:.1f} mmHg — NORMAL. Normal MAP. Adequate organ perfusion pressure."
    else: return f"Mean Arterial Pressure: {val:.1f} mmHg — ELEVATED. Elevated MAP. May indicate vasopressor use or hypertensive state."

def interpret_dbp(val):
    if val is None: return "Diastolic Blood Pressure: Not recorded."
    val = float(val)
    if val < 40: return f"Diastolic BP: {val:.1f} mmHg — CRITICALLY LOW. Severely low diastolic pressure. Indicates poor vascular tone and impaired coronary perfusion."
    elif 40 <= val < 60: return f"Diastolic BP: {val:.1f} mmHg — LOW — below mortality change-point (~60 mmHg). Below the ~60 mmHg change-point associated with increased ICU mortality in sepsis."
    elif 60 <= val <= 80: return f"Diastolic BP: {val:.1f} mmHg — NORMAL. Normal diastolic pressure."
    else: return f"Diastolic BP: {val:.1f} mmHg — ELEVATED. Elevated diastolic pressure."

def interpret_resp(val):
    if val is None: return "Respiratory Rate: Not recorded."
    val = float(val)
    if val < 12: return f"Respiratory Rate: {val:.1f} breaths/min — LOW — bradypnea. Low respiratory rate. May indicate CNS depression or late-stage decompensation."
    elif 12 <= val <= 20: return f"Respiratory Rate: {val:.1f} breaths/min — NORMAL. Normal respiratory rate."
    elif 20 < val < 22: return f"Respiratory Rate: {val:.1f} breaths/min — BORDERLINE ELEVATED. Approaching the qSOFA threshold of 22/min."
    elif 22 <= val <= 30: return f"Respiratory Rate: {val:.1f} breaths/min — ELEVATED — qSOFA threshold met (≥22/min). Tachypnea is a compensatory mechanism for metabolic acidosis in sepsis and a sensitive early warning sign."
    else: return f"Respiratory Rate: {val:.1f} breaths/min — CRITICALLY ELEVATED — severe tachypnea. Severe tachypnea. Strongly associated with sepsis severity, ARDS risk, and mortality."

def interpret_age(val):
    if val is None: return "Age: Not recorded."
    val = float(val)
    if val < 45: return f"Age: {val:.0f} years. Young adult. Lower baseline sepsis risk."
    elif 45 <= val < 65: return f"Age: {val:.0f} years. Middle-aged. Moderate sepsis risk."
    elif 65 <= val < 85: return f"Age: {val:.0f} years. Elderly. Significantly higher sepsis risk and mortality due to immunosenescence and comorbidities."
    else: return f"Age: {val:.0f} years. Very elderly (≥85). Highest risk group — sepsis mortality ~38% vs ~10% in younger patients."

def interpret_gender(val):
    if val is None: return "Gender: Not recorded."
    val = float(val)
    if val == 1: return "Gender: Male. Male sex is associated with ~1.3x higher sepsis incidence than females, longer ICU stay, and higher mortality."
    else: return "Gender: Female. Female sex is associated with lower sepsis incidence but outcomes vary by age and pathogen."

def interpret_hospadmtime(val):
    if val is None: return "Time from Hospital to ICU Admission: Not recorded."
    val = float(val)
    if val < 0: return f"Time from Hospital to ICU Admission: {abs(val):.0f} hours before hospital admission (direct ICU admit). Suggests acute presentation."
    elif val == 0: return "Time from Hospital to ICU Admission: Admitted directly to ICU."
    elif val <= 6: return f"Time from Hospital to ICU Admission: {val:.0f} hours — early ICU transfer. Generally not associated with worse outcomes."
    else: return f"Time from Hospital to ICU Admission: {val:.0f} hours — delayed ICU transfer. Delayed admission associated with higher mortality in severe sepsis subgroups."

def interpret_iculos(val):
    if val is None: return "ICU Length of Stay: Not recorded."
    val = float(val)
    if val <= 24: return f"ICU Length of Stay: {val:.0f} hours — Early stay. Lower cumulative exposure to invasive devices and hospital pathogens."
    elif 24 < val <= 72: return f"ICU Length of Stay: {val:.0f} hours — Moderate stay (1-3 days). Increasing exposure to ventilators, central lines, and urinary catheters raises infection risk."
    elif 72 < val <= 168: return f"ICU Length of Stay: {val:.0f} hours — Prolonged stay (3-7 days). Significantly higher risk of device-associated infections that can progress to sepsis or septic shock."
    else: return f"ICU Length of Stay: {val:.0f} hours — Extended stay (>7 days). Highest risk of ICU-acquired infections. Prolonged stay often reflects illness severity or complications."

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
# FEW-SHOT EXAMPLE BUILDERS
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
        available_cols = [c for c in CHOSEN_COLS if c in ex_df.columns]
        hour_blocks = []
        for _, row in ex_df.iterrows():
            hour = row.get("Hour", "?")
            vitals = build_contextualized_vitals(row, available_cols)
            hour_blocks.append(f"--- Hour {hour} ---\n{vitals}")
        example_blocks.append(f"EXAMPLE (No Sepsis — correct answer is 0):\n" + "\n\n".join(hour_blocks) + "\nAnswer: 0")

    for i in range(min(n_per_class, len(sepsis_ids))):
        ex_id = sepsis_ids[i]
        ex_df = grouped.get_group(ex_id).sort_values("Hour")
        if len(ex_df) > MAX_TIMESTEPS_PER_PATIENT:
            ex_df = ex_df.tail(MAX_TIMESTEPS_PER_PATIENT)
        available_cols = [c for c in CHOSEN_COLS if c in ex_df.columns]
        hour_blocks = []
        for _, row in ex_df.iterrows():
            hour = row.get("Hour", "?")
            vitals = build_contextualized_vitals(row, available_cols)
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
        available_cols = [c for c in CHOSEN_COLS if c in last.index]
        vitals = build_contextualized_vitals(last, available_cols)
        example_blocks.append(f"EXAMPLE (No Sepsis — correct answer is 0):\n{vitals}\nAnswer: 0")

    for i in range(min(n_per_class, len(sepsis_ids))):
        ex_id = sepsis_ids[i]
        ex_df = grouped.get_group(ex_id).sort_values("Hour")
        last = ex_df.iloc[-1]
        available_cols = [c for c in CHOSEN_COLS if c in last.index]
        vitals = build_contextualized_vitals(last, available_cols)
        example_blocks.append(f"EXAMPLE (Sepsis — correct answer is 1):\n{vitals}\nAnswer: 1")

    return "\n\n" + "="*40 + "\n\n".join(example_blocks) + "\n\n" + "="*40 + "\n\n"


# =========================
# PROMPT BUILDERS — FEW-SHOT + AGENT A RULES
# Structure: examples → rules for THIS patient → patient vitals
# =========================
def build_full_timeseries_prompt(patient_df, few_shot_block):
    if len(patient_df) > MAX_TIMESTEPS_PER_PATIENT:
        patient_df = patient_df.tail(MAX_TIMESTEPS_PER_PATIENT)

    patient_id = patient_df["Patient_ID"].iloc[0]
    available_cols = [c for c in CHOSEN_COLS if c in patient_df.columns]

    # Agent A rules from last timestep
    agg_row = pd.Series({
        'HR':    patient_df['HR'].max(),
        'Temp':  patient_df['Temp'].max(),
        'Resp':  patient_df['Resp'].max(),
        'O2Sat': patient_df['O2Sat'].min(),
    })
    patient_categories = discretize_row(agg_row)
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

        Below are examples of patients with and without sepsis, followed by a new patient to predict.
        Each variable has been interpreted using evidence-based clinical thresholds from the medical literature.

        {few_shot_block}

        Now predict this new patient:
        Patient ID: {patient_id}

        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        {rules_block}
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        Time-series:
        {timeseries_text}

        Task:
        Based on the examples, the association rules above, AND the full time-series data,
        predict whether this patient will develop sepsis at any point.

        Important:
        - Respond with ONLY one label: 0 or 1
        - 1 means Sepsis
        - 0 means No Sepsis
        - Do not explain your answer
        - Not all ICU patients develop sepsis. Normal ICU vitals may appear abnormal
          compared to healthy patients — consider trends and combinations of variables,
          not individual values in isolation.
        - Only predict 1 if there is clear evidence of systemic infection and organ
          dysfunction across multiple variables.
        """.strip()

    return prompt


def build_final_timestep_prompt(patient_df, few_shot_block):
    last = patient_df.iloc[-1]
    patient_id = last["Patient_ID"]
    available_cols = [c for c in CHOSEN_COLS if c in last.index]
    vitals = build_contextualized_vitals(last, available_cols)

    # Agent A rules for this patient
    agg_row = pd.Series({
        'HR':    patient_df['HR'].max(),
        'Temp':  patient_df['Temp'].max(),
        'Resp':  patient_df['Resp'].max(),
        'O2Sat': patient_df['O2Sat'].min(),
    })
    patient_categories = discretize_row(agg_row)
    matching_rules = get_matching_rules(patient_categories)
    rules_block = format_rules_for_prompt(matching_rules, patient_categories)

    prompt = f"""
        You are assisting with a sepsis prediction experiment.

        Below are examples of patients with and without sepsis, followed by a new patient to predict.
        Each variable has been interpreted using evidence-based clinical thresholds from the medical literature.

        {few_shot_block}

        Now predict this new patient:
        Patient ID: {patient_id}

        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        {rules_block}
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        Final state:
        {vitals}

        Task:
        Based on the examples, the association rules above, AND this final time-step,
        predict whether this patient has sepsis.

        Important:
        - Respond with ONLY one label: 0 or 1
        - 1 means Sepsis
        - 0 means No Sepsis
        - Do not explain your answer
        - Not all ICU patients develop sepsis. Normal ICU vitals may appear abnormal
          compared to healthy patients — consider trends and combinations of variables,
          not individual values in isolation.
        - Only predict 1 if there is clear evidence of systemic infection and organ
          dysfunction across multiple variables.
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
# RUN EXPERIMENT — NO SHUFFLE (same patients as document 30 baseline)
# =========================
results = []
grouped = df.groupby("Patient_ID")
patient_ids = list(grouped.groups.keys())[:MAX_PATIENTS]

print(f"\nRunning Few-Shot + Agent A experiment on {len(patient_ids)} patients...")
print(f"Model: {MODEL_NAME} | Rules: {len(RULES)} | N_SHOT: {N_SHOT}\n")

for idx, patient_id in enumerate(patient_ids, start=1):
    patient_df = grouped.get_group(patient_id).sort_values("Hour")
    true_label = int(patient_df["SepsisLabel"].max())

    ## Debug: Show matched rules for this patient
    agg_row = pd.Series({
    'HR':    patient_df['HR'].max(),
    'Temp':  patient_df['Temp'].max(),
    'Resp':  patient_df['Resp'].max(),
    'O2Sat': patient_df['O2Sat'].min(),
    })
    debug_cats = discretize_row(agg_row)
    debug_rules = get_matching_rules(debug_cats)
    print(f"  → Categories: {debug_cats} | Rules matched: {len(debug_rules)}")

    exclude = set(patient_ids)
    full_few_shot = build_fewshot_examples(df, grouped, exclude, n_per_class=N_SHOT)
    final_few_shot = build_fewshot_examples_final(df, grouped, exclude, n_per_class=N_SHOT)

    full_prompt = build_full_timeseries_prompt(patient_df, full_few_shot)
    try:
        full_pred, full_raw = call_gemini(full_prompt)
    except Exception as e:
        full_pred, full_raw = None, f"ERROR: {e}"
    time.sleep(SLEEP_SECONDS)

    final_prompt = build_final_timestep_prompt(patient_df, final_few_shot)
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
    f.write("GEMINI FEW-SHOT + AGENT A — REPORT\n")
    f.write("=" * 60 + "\n")
    f.write(f"Model:                  {MODEL_NAME}\n")
    f.write(f"Dataset:                {CSV_PATH}\n")
    f.write(f"Prompt Type:            Few-Shot + Agent A Rules\n")
    f.write(f"N_SHOT:                 {N_SHOT}\n")
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