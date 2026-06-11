import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.utils import resample

# =========================
# CONFIGURATION
# =========================
ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT_DIR / "data" / "raw" / "physionet_2019.csv"
OUTPUT_DIR = ROOT_DIR / "data" / "processed" / "septic_shock"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("SEPTIC SHOCK LABEL DERIVATION")
print("=" * 60)

# =========================
# STEP 1 — LOAD RAW DATA
# =========================
print("\nLoading raw dataset...")
df = pd.read_csv(RAW_PATH)
if df.columns[0] == "Unnamed: 0":
    df = df.drop(columns=df.columns[0])
print(f"Raw shape: {df.shape}")

# =========================
# STEP 2 — ISOLATE SEPSIS PATIENTS
# =========================
print("\nIsolating sepsis patients...")
patient_labels = df.groupby("Patient_ID")["SepsisLabel"].max()
sepsis_ids = patient_labels[patient_labels == 1].index
df_sepsis = df[df["Patient_ID"].isin(sepsis_ids)].copy()
print(f"Sepsis patients: {len(sepsis_ids)}")


# =========================
# STEP 3 — WORST CASE PER PATIENT
# =========================
print("\nCalculating worst-case values per patient...")
patient_agg = df_sepsis.groupby("Patient_ID").agg(
    # Vital signs — worst case (same as existing pipeline)
    HR_max        = ("HR",       "max"),
    Temp_max      = ("Temp",     "max"),
    Resp_max      = ("Resp",     "max"),
    O2Sat_min     = ("O2Sat",    "min"),
    MAP_min       = ("MAP",      "min"),
    SBP_min       = ("SBP",      "min"),
    DBP_min       = ("DBP",      "min"),
    # Lab values — worst case (newly included)
    Lactate_max   = ("Lactate",  "max"),
    Creatinine_max= ("Creatinine","max"),
    Platelets_min = ("Platelets","min"),
    Bilirubin_max = ("Bilirubin_total", "max"),
    FiO2_min      = ("FiO2",     "min"),
    # Static
    Age           = ("Age",      "first"),
    Gender        = ("Gender",   "first"),
    HospAdmTime   = ("HospAdmTime", "first"),
    ICULOS        = ("ICULOS",   "max"),
    # Missingness indicators
    Lactate_measured   = ("Lactate",   lambda x: x.notna().any().astype(int)),
    Creatinine_measured= ("Creatinine",lambda x: x.notna().any().astype(int)),
    Platelets_measured = ("Platelets", lambda x: x.notna().any().astype(int)),
).reset_index()



# =========================
# STEP 4 — APPLY BOTH LABEL DEFINITIONS
# =========================
print("\nApplying septic shock label definitions...")

# Strict: MAP < 65 AND Lactate > 2
strict_shock = (
    (patient_agg["MAP_min"] < 65) &
    (patient_agg["Lactate_max"] > 2.0)
)

# Relaxed: MAP < 65 only
relaxed_shock = (patient_agg["MAP_min"] < 65)

patient_agg["SepticShock_strict"]  = strict_shock.astype(int)
patient_agg["SepticShock_relaxed"] = relaxed_shock.astype(int)

print(f"Strict  — Septic shock: {strict_shock.sum():,} | Non-shock: {(~strict_shock).sum():,}")
print(f"Relaxed — Septic shock: {relaxed_shock.sum():,} | Non-shock: {(~relaxed_shock).sum():,}")


# =========================
# STEP 5 — BUILD BALANCED DATASETS (ONE ROW PER PATIENT)
# For XGBoost / Random Forest
# =========================
print("\nBuilding balanced one-row-per-patient datasets...")

def build_balanced(df, label_col, label_name):
    shock     = df[df[label_col] == 1]
    non_shock = df[df[label_col] == 0]
    n = min(len(shock), len(non_shock))
    shock_sample     = resample(shock,     n_samples=n, random_state=42, replace=False)
    non_shock_sample = resample(non_shock, n_samples=n, random_state=42, replace=False)
    balanced = pd.concat([shock_sample, non_shock_sample]).sample(frac=1, random_state=42)
    balanced = balanced.rename(columns={label_col: "SepticShockLabel"})
    # Drop the other label column
    other = [c for c in ["SepticShock_strict","SepticShock_relaxed"] if c != label_col]
    balanced = balanced.drop(columns=other)
    print(f"{label_name}: {n:,} shock + {n:,} non-shock = {len(balanced):,} total")
    return balanced

strict_balanced  = build_balanced(patient_agg, "SepticShock_strict",  "Strict")
relaxed_balanced = build_balanced(patient_agg, "SepticShock_relaxed", "Relaxed")

# =========================
# STEP 5 — BUILD BALANCED DATASETS (ONE ROW PER PATIENT)
# For XGBoost / Random Forest
# =========================
print("\nBuilding balanced one-row-per-patient datasets...")

def build_balanced(df, shock_ids, nonshock_ids, label_name):
    shock     = df[df["Patient_ID"].isin(shock_ids)]
    non_shock = df[df["Patient_ID"].isin(nonshock_ids)]
    n = min(len(shock), len(non_shock))
    shock_sample     = resample(shock,     n_samples=n, random_state=42, replace=False)
    non_shock_sample = resample(non_shock, n_samples=n, random_state=42, replace=False)
    balanced = pd.concat([shock_sample, non_shock_sample]).sample(frac=1, random_state=42)
    balanced = balanced.drop(columns=["SepticShock_strict", "SepticShock_relaxed"])
    balanced["SepticShockLabel"] = [1]*n + [0]*n
    balanced = balanced.sample(frac=1, random_state=42)
    print(f"{label_name}: {n:,} shock + {n:,} non-shock = {len(balanced):,} total")
    return balanced

# Clean non-shock pool — patients whose MAP NEVER dropped below 65
# These are genuinely non-shock regardless of Lactate status
clean_nonshock_ids = patient_agg[patient_agg["MAP_min"] >= 65]["Patient_ID"]

# Strict shock — MAP < 65 AND Lactate > 2
strict_shock_ids   = patient_agg[patient_agg["SepticShock_strict"]  == 1]["Patient_ID"]

# Relaxed shock — MAP < 65
relaxed_shock_ids  = patient_agg[patient_agg["SepticShock_relaxed"] == 1]["Patient_ID"]

print(f"\nClean non-shock pool (MAP always >= 65): {len(clean_nonshock_ids):,} patients")
print(f"Strict shock pool (MAP < 65 AND Lactate > 2): {len(strict_shock_ids):,} patients")
print(f"Relaxed shock pool (MAP < 65): {len(relaxed_shock_ids):,} patients")

strict_balanced  = build_balanced(patient_agg, strict_shock_ids,  clean_nonshock_ids, "Strict")
relaxed_balanced = build_balanced(patient_agg, relaxed_shock_ids, clean_nonshock_ids, "Relaxed")

# =========================
# STEP 6 — BUILD TIME-SERIES DATASETS
# For Transformer / BiLSTM
# =========================
print("\nBuilding time-series datasets...")

def build_timeseries(shock_ids, nonshock_ids, label_name):
    n = min(len(shock_ids), len(nonshock_ids))
    shock_sample    = resample(shock_ids.values,    n_samples=n, random_state=42, replace=False)
    nonshock_sample = resample(nonshock_ids.values, n_samples=n, random_state=42, replace=False)
    all_ids = list(shock_sample) + list(nonshock_sample)
    df_ts = df_sepsis[df_sepsis["Patient_ID"].isin(all_ids)].copy()
    shock_map = {pid: 1 for pid in shock_sample}
    shock_map.update({pid: 0 for pid in nonshock_sample})
    df_ts["SepticShockLabel"] = df_ts["Patient_ID"].map(shock_map)
    print(f"{label_name} time-series: {len(all_ids):,} patients, {len(df_ts):,} rows")
    return df_ts

strict_ts  = build_timeseries(strict_shock_ids,  clean_nonshock_ids, "Strict")
relaxed_ts = build_timeseries(relaxed_shock_ids, clean_nonshock_ids, "Relaxed")

# =========================
# STEP 7 — SAVE ALL OUTPUTS
# =========================
print("\nSaving datasets...")
strict_balanced.to_csv(OUTPUT_DIR  / "strict_balanced_xgb.csv",          index=False)
relaxed_balanced.to_csv(OUTPUT_DIR / "relaxed_balanced_xgb.csv",         index=False)
strict_ts.to_csv(OUTPUT_DIR        / "strict_timeseries_transformer.csv", index=False)
relaxed_ts.to_csv(OUTPUT_DIR       / "relaxed_timeseries_transformer.csv",index=False)
print(f"Saved to: {OUTPUT_DIR}")

# =========================
# STEP 8 — SAVE SUMMARY
# =========================
summary = f"""
SEPTIC SHOCK LABEL DERIVATION SUMMARY
======================================
Source dataset: PhysioNet 2019 (raw)
Total sepsis patients: {len(sepsis_ids):,}

NON-SHOCK POOL (clean):
- Patients whose MAP never dropped below 65 mmHg: {len(clean_nonshock_ids):,}
- These patients never showed cardiovascular failure
- Lactate status irrelevant — MAP alone rules out shock

STRICT DEFINITION (MAP < 65 AND Lactate > 2):
- Septic shock patients: {len(strict_shock_ids):,}
- Non-shock patients (from clean pool): {min(len(strict_shock_ids), len(clean_nonshock_ids)):,}
- Balanced XGBoost dataset: {len(strict_balanced):,} patients
- Balanced Transformer dataset: {len(strict_ts):,} rows

RELAXED DEFINITION (MAP < 65 only):
- Septic shock patients: {len(relaxed_shock_ids):,}
- Non-shock patients (from clean pool): {min(len(relaxed_shock_ids), len(clean_nonshock_ids)):,}
- Balanced XGBoost dataset: {len(relaxed_balanced):,} patients
- Balanced Transformer dataset: {len(relaxed_ts):,} rows

PROXY JUSTIFICATION:
- No vasopressor administration data in PhysioNet 2019
- MAP < 65 mmHg used as cardiovascular failure proxy
- MAP < 65 is the established hemodynamic threshold in vasopressor protocols
- Lactate > 2 mmol/L confirms metabolic failure per Sepsis-3
- GCS excluded (not in dataset)
- Missingness indicators included for Lactate, Creatinine, Platelets

NOTES:
- Non-shock pool uses only patients with MAP always >= 65 (clean comparison group)
- Ambiguous patients (MAP < 65 but no Lactate) excluded from non-shock pool
- Lab columns included at patient level (at least one reading)
- Non-sepsis patients excluded entirely
- Worst-case aggregation used for XGBoost dataset
- Full hourly time-series preserved for Transformer dataset
"""

with open(OUTPUT_DIR / "derivation_summary.txt", "w") as f:
    f.write(summary)

print(summary)
print("=" * 60)
print("DONE")
print("=" * 60)

lactate_high_map_ok = (
    (patient_agg["MAP_min"] >= 65) & 
    (patient_agg["Lactate_max"] > 2.0)
)
print(f"MAP >= 65 but Lactate > 2: {lactate_high_map_ok.sum():,}")