import os
import re
import time
import pandas as pd
from google import genai
from sklearn.metrics import classification_report

from pathlib import Path

#########################################################
#### #1 is Configuration step - set up Gemini API client
#########################################################

#API_KEY = os.getenv("GEMINI_API_KEY")


API_KEY = os.getenv("GEMINI_API_KEY")

MODEL_NAME = "gemini-2.5-flash"

# Repo root: walk up until we find requirements.txt. Works no matter where the
# repo is cloned or how deeply this script is nested. See README "File Paths".
ROOT_DIR = next(p for p in Path(__file__).resolve().parents if (p / "requirements.txt").exists())

# Input CSV path — using balanced dataset (50/50 sepsis/no-sepsis)
CSV_PATH = ROOT_DIR / "data" / "processed" / "physionet_balanced.csv"

# Limit patients for testing first. Increase Later.
MAX_PATIENTS = 100

# If a patient has many timesteps, keep only the last N to avoid huge prompts.
MAX_TIMESTEPS_PER_PATIENT = 48

# Number of few-shot examples per class (1 = 1 sepsis + 1 no-sepsis example)
N_SHOT = 1

# Sleep a bit between calls to be gentle with rate limits.
SLEEP_SECONDS = 2.0

# Output files
run_timestamp_file = time.strftime("%Y%m%d_%H%M%S")
OUTPUT_CSV = f"gemini_fewshotContextualized_results_{run_timestamp_file}.csv"
REPORT_FILE = f"gemini_fewshotContextualized_report_{run_timestamp_file}.txt"


# =========================
# 2) SAFETY CHECKS
# =========================

if not API_KEY:
    raise ValueError(
        "Missing GEMINI_API_KEY. Set it in your terminal first, e.g. "
        'export GEMINI_API_KEY="your_key_here"'
    )

client = genai.Client(api_key=API_KEY)


# =========================
# 3) LOAD DATA
# =========================

df = pd.read_csv(CSV_PATH)

df = df.set_index("Patient_ID")           # Move Patient_ID to index temporarily
df = df.groupby("Patient_ID").ffill()     # Forward fill within each patient
df = df.groupby("Patient_ID").bfill()     # Backward fill within each patient
df = df.reset_index()                      # Restore Patient_ID as a column
df["Patient_ID"] = df["Patient_ID"].astype(int)

required_cols = ["Patient_ID", "Hour", "SepsisLabel"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Required column '{col}' not found in CSV.")

df = df.sort_values(["Patient_ID", "Hour"]).reset_index(drop=True)


# =========================
# 4) HELPER FUNCTIONS
# =========================

def clean_value(value):
    """Convert NaN to None so the prompt is readable."""
    if pd.isna(value):
        return None
    return value


# -------------------------------------------------------
# CONTEXTUALIZATION FUNCTIONS
# Translates raw vitals into clinical English using
# evidence-based thresholds from Simran's variables doc.
# -------------------------------------------------------

def interpret_hr(val):
    """
    HR — Heart rate (beats per minute)
    Strong relationship to sepsis. Elevated HR (tachycardia) is a sensitive
    early indicator of systemic inflammation and hemodynamic stress.
    Source: Simran's variables document.
    """
    if val is None:
        return "Heart Rate: Not recorded."
    val = float(val)
    if val < 60:
        status = "BRADYCARDIC (abnormally low)"
        note = "Low HR in sepsis can indicate late-stage cardiovascular decompensation."
    elif 60 <= val <= 100:
        status = "NORMAL"
        note = "Normal HR does not rule out early sepsis."
    elif 100 < val <= 120:
        status = "ELEVATED — mild tachycardia"
        note = "Mild tachycardia is a sensitive early warning sign of systemic infection and inflammation."
    else:
        status = "CRITICALLY ELEVATED — severe tachycardia"
        note = "Severe tachycardia strongly associated with sepsis and hemodynamic instability."
    return f"Heart Rate: {val:.1f} bpm — {status}. {note}"


def interpret_o2sat(val):
    """
    O2Sat — Pulse oximetry (%)
    Moderate relationship. Lower SpO2 associated with organ dysfunction and
    higher mortality in sepsis. Note: SpO2 overestimates true SaO2 by ~2.75%
    in sepsis patients, so borderline values may be worse than they appear.
    Source: Simran's variables document.
    """
    if val is None:
        return "Oxygen Saturation: Not recorded."
    val = float(val)
    if val >= 95:
        status = "NORMAL"
        note = "Adequate oxygenation. Note SpO2 may overestimate true saturation in sepsis."
    elif 90 <= val < 95:
        status = "LOW — mild hypoxemia"
        note = "Mild hypoxemia. Associated with increased risk of organ dysfunction in sepsis."
    else:
        status = "CRITICALLY LOW — severe hypoxemia"
        note = "Severe hypoxemia strongly associated with organ failure and mortality in sepsis."
    return f"Oxygen Saturation: {val:.1f}% — {status}. {note}"


def interpret_temp(val):
    """
    Temp — Temperature (°C)
    Strong relationship. Both fever and hypothermia are significant.
    Hypothermia is MORE strongly associated with septic shock and higher
    mortality (~47% mortality) compared to fever (~22% mortality).
    Source: Simran's variables document.
    """
    if val is None:
        return "Temperature: Not recorded."
    val = float(val)
    if val < 36.0:
        status = "HYPOTHERMIC"
        note = "Hypothermia in sepsis is more dangerous than fever — associated with ~47% mortality vs ~22% with fever. Strong indicator of septic shock."
    elif 36.0 <= val <= 38.2:
        status = "NORMAL"
        note = "Normal temperature. Does not rule out sepsis — early sepsis can present without fever."
    elif 38.2 < val <= 39.5:
        status = "FEVER — mild"
        note = "Mild fever supports suspicion of systemic infection. Threshold for ICU sepsis fever is typically 38.2°C."
    else:
        status = "FEVER — high"
        note = "High fever strongly associated with systemic infection and sepsis physiology."
    return f"Temperature: {val:.1f}°C — {status}. {note}"


def interpret_sbp(val):
    """
    SBP — Systolic Blood Pressure (mm Hg)
    Strong relationship. qSOFA uses SBP ≤100 mmHg as a key criterion.
    Source: Simran's variables document.
    """
    if val is None:
        return "Systolic Blood Pressure: Not recorded."
    val = float(val)
    if val <= 90:
        status = "CRITICALLY LOW — severe hypotension"
        note = "Severe hypotension. Strong indicator of septic shock and circulatory failure."
    elif 90 < val <= 100:
        status = "LOW — qSOFA threshold met (≤100 mmHg)"
        note = "Meets qSOFA criteria. Associated with poor outcomes in sepsis."
    elif 100 < val <= 120:
        status = "BORDERLINE LOW"
        note = "Below normal range. Warrants monitoring for declining trend."
    elif 120 < val <= 140:
        status = "NORMAL"
        note = "Normal systolic pressure."
    else:
        status = "ELEVATED"
        note = "Elevated SBP. Less concerning for septic shock."
    return f"Systolic BP: {val:.1f} mmHg — {status}. {note}"


def interpret_map(val):
    """
    MAP — Mean Arterial Pressure (mm Hg)
    VERY STRONG relationship — one of the strongest single hemodynamic
    predictors of septic shock. MAP <65 mmHg is the clinical threshold for
    septic shock. Studies show U-shaped relationship with lowest mortality
    around 70-82 mmHg.
    Source: Simran's variables document.
    """
    if val is None:
        return "Mean Arterial Pressure: Not recorded."
    val = float(val)
    if val < 65:
        status = "CRITICALLY LOW — septic shock threshold"
        note = "MAP <65 mmHg meets clinical definition of septic shock. Risk rises sharply below this threshold."
    elif 65 <= val <= 70:
        status = "LOW — borderline shock threshold"
        note = "At the lower edge of the septic shock threshold. High risk for organ hypoperfusion."
    elif 70 < val <= 82:
        status = "ACCEPTABLE — lowest mortality range"
        note = "Within the range associated with lowest 28-day mortality in sepsis patients (70-82 mmHg)."
    elif 82 < val <= 100:
        status = "NORMAL"
        note = "Normal MAP. Adequate organ perfusion pressure."
    else:
        status = "ELEVATED"
        note = "Elevated MAP. May indicate vasopressor use or hypertensive state."
    return f"Mean Arterial Pressure: {val:.1f} mmHg — {status}. {note}"


def interpret_dbp(val):
    """
    DBP — Diastolic Blood Pressure (mm Hg)
    Moderate relationship. Clinical change-point for ICU mortality is ~60 mmHg.
    Source: Simran's variables document.
    """
    if val is None:
        return "Diastolic Blood Pressure: Not recorded."
    val = float(val)
    if val < 40:
        status = "CRITICALLY LOW"
        note = "Severely low diastolic pressure. Indicates poor vascular tone and impaired coronary perfusion."
    elif 40 <= val < 60:
        status = "LOW — below mortality change-point (~60 mmHg)"
        note = "Below the ~60 mmHg change-point associated with increased ICU mortality in sepsis."
    elif 60 <= val <= 80:
        status = "NORMAL"
        note = "Normal diastolic pressure."
    else:
        status = "ELEVATED"
        note = "Elevated diastolic pressure."
    return f"Diastolic BP: {val:.1f} mmHg — {status}. {note}"


def interpret_resp(val):
    """
    Resp — Respiratory Rate (breaths per minute)
    Strong relationship. Tachypnea is one of the EARLIEST and most sensitive
    physiologic abnormalities in sepsis. qSOFA uses RR ≥22/min as a criterion.
    Source: Simran's variables document.
    """
    if val is None:
        return "Respiratory Rate: Not recorded."
    val = float(val)
    if val < 12:
        status = "LOW — bradypnea"
        note = "Low respiratory rate. May indicate CNS depression or late-stage decompensation."
    elif 12 <= val <= 20:
        status = "NORMAL"
        note = "Normal respiratory rate."
    elif 20 < val < 22:
        status = "BORDERLINE ELEVATED"
        note = "Approaching the qSOFA threshold of 22/min."
    elif 22 <= val <= 30:
        status = "ELEVATED — qSOFA threshold met (≥22/min)"
        note = "Tachypnea is a compensatory mechanism for metabolic acidosis in sepsis and a sensitive early warning sign."
    else:
        status = "CRITICALLY ELEVATED — severe tachypnea"
        note = "Severe tachypnea. Strongly associated with sepsis severity, ARDS risk, and mortality."
    return f"Respiratory Rate: {val:.1f} breaths/min — {status}. {note}"


def interpret_age(val):
    """
    Age — Patient age (years, capped at 100)
    Strong risk factor. Incidence increases >100-fold with age.
    Mortality rises from ~10% in children to ~38% in patients ≥85.
    Source: Simran's variables document.
    """
    if val is None:
        return "Age: Not recorded."
    val = float(val)
    if val < 45:
        note = "Young adult. Lower baseline sepsis risk."
    elif 45 <= val < 65:
        note = "Middle-aged. Moderate sepsis risk."
    elif 65 <= val < 85:
        note = "Elderly. Significantly higher sepsis risk and mortality due to immunosenescence and comorbidities."
    else:
        note = "Very elderly (≥85). Highest risk group — sepsis mortality ~38% vs ~10% in younger patients."
    return f"Age: {val:.0f} years. {note}"


def interpret_gender(val):
    """
    Gender — Female (0) or Male (1)
    Moderate relationship. Male sex associated with ~1.3x higher sepsis
    incidence than females. Males show longer ICU stay and higher mortality.
    Source: Simran's variables document.
    """
    if val is None:
        return "Gender: Not recorded."
    val = float(val)
    if val == 1:
        return "Gender: Male. Male sex is associated with ~1.3x higher sepsis incidence than females, longer ICU stay, and higher mortality."
    else:
        return "Gender: Female. Female sex is associated with lower sepsis incidence but outcomes vary by age and pathogen."


def interpret_hospadmtime(val):
    """
    HospAdmTime — Hours between hospital admission and ICU admission
    Moderate relationship. Delayed ICU admission is associated with higher
    mortality in high-severity subgroups (rising lactate, septic shock).
    Source: Simran's variables document.
    """
    if val is None:
        return "Time from Hospital to ICU Admission: Not recorded."
    val = float(val)
    if val < 0:
        return f"Time from Hospital to ICU Admission: {abs(val):.0f} hours before hospital admission (direct ICU admit). Suggests acute presentation."
    elif val == 0:
        return "Time from Hospital to ICU Admission: Admitted directly to ICU."
    elif val <= 6:
        return f"Time from Hospital to ICU Admission: {val:.0f} hours — early ICU transfer. Generally not associated with worse outcomes."
    else:
        return f"Time from Hospital to ICU Admission: {val:.0f} hours — delayed ICU transfer. Delayed admission associated with higher mortality in severe sepsis subgroups."


def interpret_iculos(val):
    """
    ICULOS — ICU Length of Stay (hours)
    Related: Yes
    Strength: Moderate (indirect, exposure- and severity-mediated)
    Longer ICU stay increases exposure to invasive devices and hospital
    pathogens, raising risk of ICU-acquired infections progressing to sepsis.
    Source: Simran's variables document.
    """
    if val is None:
        return "ICU Length of Stay: Not recorded."
    val = float(val)
    if val <= 24:
        return f"ICU Length of Stay: {val:.0f} hours — Early stay. Lower cumulative exposure to invasive devices and hospital pathogens."
    elif 24 < val <= 72:
        return f"ICU Length of Stay: {val:.0f} hours — Moderate stay (1-3 days). Increasing exposure to ventilators, central lines, and urinary catheters raises infection risk."
    elif 72 < val <= 168:
        return f"ICU Length of Stay: {val:.0f} hours — Prolonged stay (3-7 days). Significantly higher risk of device-associated infections that can progress to sepsis or septic shock."
    else:
        return f"ICU Length of Stay: {val:.0f} hours — Extended stay (>7 days). Highest risk of ICU-acquired infections. Prolonged stay often reflects illness severity or complications."


def build_contextualized_vitals(row, chosen_cols):
    """
    Convert one row of raw vitals into clinical English using
    evidence-based thresholds from Simran's variables document.
    """
    col_map = {
        "HR":          lambda v: interpret_hr(v),
        "O2Sat":       lambda v: interpret_o2sat(v),
        "Temp":        lambda v: interpret_temp(v),
        "SBP":         lambda v: interpret_sbp(v),
        "MAP":         lambda v: interpret_map(v),
        "DBP":         lambda v: interpret_dbp(v),
        "Resp":        lambda v: interpret_resp(v),
        "Age":         lambda v: interpret_age(v),
        "Gender":      lambda v: interpret_gender(v),
        "HospAdmTime": lambda v: interpret_hospadmtime(v),
        "ICULOS":      lambda v: interpret_iculos(v),
    }

    lines = []

    for col in chosen_cols:

        if col in col_map:

            raw = clean_value(row[col]) if col in row.index else None

            lines.append(col_map[col](raw))

    return "\n".join(lines)


# =========================
# 5) FEW-SHOT EXAMPLE BUILDER
# =========================

# Clinically relevant columns — all backed by Simran's variables document.
CHOSEN_COLS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "Age", "Gender", "HospAdmTime", "ICULOS"]


def build_fewshot_examples(df, grouped, exclude_ids, n_per_class=1):
    """
    Pull n_per_class real patients from the dataset for each class (0 and 1)
    to use as few-shot examples. Excludes patients in exclude_ids so we don't
    use a test patient as an example.

    Returns a string block of labeled examples to prepend to the prompt.
    """
    patient_labels = df.groupby("Patient_ID")["SepsisLabel"].max()

    no_sepsis_ids = patient_labels[
        (patient_labels == 0) & (~patient_labels.index.isin(exclude_ids))
    ].index.tolist()

    sepsis_ids = patient_labels[
        (patient_labels == 1) & (~patient_labels.index.isin(exclude_ids))
    ].index.tolist()

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

        example_blocks.append(
            f"EXAMPLE (No Sepsis — correct answer is 0):\n"
            + "\n\n".join(hour_blocks)
            + "\nAnswer: 0"
        )

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

        example_blocks.append(
            f"EXAMPLE (Sepsis — correct answer is 1):\n"
            + "\n\n".join(hour_blocks)
            + "\nAnswer: 1"
        )

    return "\n\n" + "="*40 + "\n\n".join(example_blocks) + "\n\n" + "="*40 + "\n\n"


def build_fewshot_examples_final(df, grouped, exclude_ids, n_per_class=1):
    """
    Same as build_fewshot_examples but uses only the final timestep
    for each example patient — matches the final timestep prompt style.
    """
    patient_labels = df.groupby("Patient_ID")["SepsisLabel"].max()

    no_sepsis_ids = patient_labels[
        (patient_labels == 0) & (~patient_labels.index.isin(exclude_ids))
    ].index.tolist()

    sepsis_ids = patient_labels[
        (patient_labels == 1) & (~patient_labels.index.isin(exclude_ids))
    ].index.tolist()

    example_blocks = []

    for i in range(min(n_per_class, len(no_sepsis_ids))):
        ex_id = no_sepsis_ids[i]
        ex_df = grouped.get_group(ex_id).sort_values("Hour")
        last = ex_df.iloc[-1]
        available_cols = [c for c in CHOSEN_COLS if c in last.index]
        vitals = build_contextualized_vitals(last, available_cols)
        example_blocks.append(
            f"EXAMPLE (No Sepsis — correct answer is 0):\n{vitals}\nAnswer: 0"
        )

    for i in range(min(n_per_class, len(sepsis_ids))):
        ex_id = sepsis_ids[i]
        ex_df = grouped.get_group(ex_id).sort_values("Hour")
        last = ex_df.iloc[-1]
        available_cols = [c for c in CHOSEN_COLS if c in last.index]
        vitals = build_contextualized_vitals(last, available_cols)
        example_blocks.append(
            f"EXAMPLE (Sepsis — correct answer is 1):\n{vitals}\nAnswer: 1"
        )

    return "\n\n" + "="*40 + "\n\n".join(example_blocks) + "\n\n" + "="*40 + "\n\n"


# =========================
# 6) PROMPT BUILDERS
# =========================

def build_full_timeseries_prompt(patient_df, few_shot_block):
    """
    Build one prompt containing all (or last N) timesteps for a single patient.
    Uses contextualized clinical English + few-shot examples.
    """
    if len(patient_df) > MAX_TIMESTEPS_PER_PATIENT:
        patient_df = patient_df.tail(MAX_TIMESTEPS_PER_PATIENT)

    patient_id = patient_df["Patient_ID"].iloc[0]
    available_cols = [c for c in CHOSEN_COLS if c in patient_df.columns]

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

        Time-series:
        {timeseries_text}

        Task:
        Based on the full time-series data above, predict whether this patient will develop sepsis at any point.

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
    """
    Build one prompt using only the final timestep of a patient.
    Uses contextualized clinical English + few-shot examples.
    """
    last = patient_df.iloc[-1]
    patient_id = last["Patient_ID"]
    available_cols = [c for c in CHOSEN_COLS if c in last.index]
    vitals = build_contextualized_vitals(last, available_cols)

    prompt = f"""
        You are assisting with a sepsis prediction experiment.

        Below are examples of patients with and without sepsis, followed by a new patient to predict.
        Each variable has been interpreted using evidence-based clinical thresholds from the medical literature.

        {few_shot_block}

        Now predict this new patient:
        Patient ID: {patient_id}

        Final state:
        {vitals}

        Task:
        Based on this final time-step only, predict whether this patient has sepsis.

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
    """
    Extract 0 or 1 from model output.
    Returns None if parsing fails.
    """
    if text is None:
        return None
    text = text.strip()
    if text == "0":
        return 0
    if text == "1":
        return 1
    match = re.search(r"\b([01])\b", text)
    if match:
        return int(match.group(1))
    return None


def call_gemini(prompt: str) -> tuple:
    """
    Send prompt to Gemini and return (parsed_prediction, raw_text).
    """
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )
    raw_text = response.text if hasattr(response, "text") else str(response)
    pred = parse_binary_prediction(raw_text)
    return pred, raw_text


# =========================
# 7) RUN EXPERIMENT
# =========================

results = []

grouped = df.groupby("Patient_ID")

# Shuffle patient IDs for a representative random sample
import random
patient_ids = list(grouped.groups.keys())[:MAX_PATIENTS]

print(f"\nRunning few-shot contextualized experiment on {len(patient_ids)} patients...")
print(f"Model: {MODEL_NAME}")
print(f"Dataset: {CSV_PATH}")
print(f"Prompt type: {N_SHOT}-Shot Contextualized (clinical English)\n")

for idx, patient_id in enumerate(patient_ids, start=1):

    patient_df = grouped.get_group(patient_id).sort_values("Hour")

    # Ground truth: if any timestep has SepsisLabel=1, mark patient as septic
    true_label = int(patient_df["SepsisLabel"].max())

    # Build few-shot examples — exclude current patient so it's not used as its own example
    exclude = set(patient_ids)  # exclude all test patients from examples
    full_few_shot = build_fewshot_examples(df, grouped, exclude, n_per_class=N_SHOT)
    final_few_shot = build_fewshot_examples_final(df, grouped, exclude, n_per_class=N_SHOT)

    # ------- FULL TIME-SERIES VERSION -------
    full_prompt = build_full_timeseries_prompt(patient_df, full_few_shot)
    try:
        full_pred, full_raw = call_gemini(full_prompt)
    except Exception as e:
        full_pred, full_raw = None, f"ERROR: {e}"

    time.sleep(SLEEP_SECONDS)

    # ------- FINAL TIMESTEP VERSION -------
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

    print(
        f"[{idx}/{len(patient_ids)}] Patient {patient_id} | "
        f"Truth={true_label} | Full={full_pred} | Final={final_pred}"
    )

    time.sleep(SLEEP_SECONDS)


# Save results
results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_CSV, index=False)

print(f"\nDone. Results saved to {OUTPUT_CSV}")


# =========================
# 8) METRICS
# =========================
valid_full = results_df.dropna(subset=["FullPromptPrediction"])
valid_final = results_df.dropna(subset=["FinalPromptPrediction"])

with open(REPORT_FILE, "w") as f:
    f.write("=" * 60 + "\n")
    f.write("GEMINI FEW-SHOT CONTEXTUALIZED — REPORT\n")
    f.write("=" * 60 + "\n")
    f.write(f"Model:                  {MODEL_NAME}\n")
    f.write(f"Dataset:                {CSV_PATH}\n")
    f.write(f"Prompt Type:            {N_SHOT}-Shot Contextualized\n")
    f.write(f"MAX_PATIENTS:           {MAX_PATIENTS}\n")
    f.write(f"MAX_TIMESTEPS:          {MAX_TIMESTEPS_PER_PATIENT}\n")
    f.write(f"Total Attempted:        {len(results_df)}\n")
    f.write(f"Valid Full Predictions: {len(valid_full)}\n")
    f.write(f"Valid Final Predictions:{len(valid_final)}\n")
    f.write("=" * 60 + "\n")

    if len(valid_full) > 0:
        full_report = classification_report(
            valid_full["TrueLabel"],
            valid_full["FullPromptPrediction"],
            zero_division=0
        )
        print("\nFULL TIMESTEPS CLASSIFICATION REPORT")
        print(full_report)
        f.write("\nFULL TIMESTEPS CLASSIFICATION REPORT\n")
        f.write(full_report + "\n")
    else:
        print("\nNo valid full predictions.")
        f.write("\nNo valid full predictions.\n")

    if len(valid_final) > 0:
        final_report = classification_report(
            valid_final["TrueLabel"],
            valid_final["FinalPromptPrediction"],
            zero_division=0
        )
        print("\nFINAL TIMESTEP CLASSIFICATION REPORT")
        print(final_report)
        f.write("\nFINAL TIMESTEP CLASSIFICATION REPORT\n")
        f.write(final_report + "\n")
    else:
        print("\nNo valid final predictions.")
        f.write("\nNo valid final predictions.\n")

print(f"\nReport saved to: {REPORT_FILE}")