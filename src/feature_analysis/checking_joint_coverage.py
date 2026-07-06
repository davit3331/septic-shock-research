import pandas as pd
import numpy as np
from pathlib import Path

# =========================
# CONFIGURATION
# =========================
RAW_PATH = Path.home() / "Desktop" / "septic-shock-research-github" / "data" / "raw" / "physionet_2019.csv"

PATIENT_ID_COL = "Patient_ID"
SEPSIS_LABEL_COL = "SepsisLabel"

# The 10 "Group A" labs from the per-feature coverage check (low sepsis/
# no-sepsis gap, high coverage both sides).
SELECTED_LABS = [
    "BUN", "Creatinine", "Hgb", "Platelets", "WBC", "Hct",
    "Potassium", "Glucose", "Magnesium", "Calcium",
]

# Also check smaller subsets, in case full-10 joint coverage is too low
# to be usable — gives a sense of the tradeoff curve rather than one
# all-or-nothing number.
SUBSETS_TO_CHECK = {
    "All 10 selected labs": SELECTED_LABS,
    "Core 6 (BUN, Creatinine, Hgb, Platelets, WBC, Hct)":
        ["BUN", "Creatinine", "Hgb", "Platelets", "WBC", "Hct"],
    "Top 4 by lowest gap (Potassium, Glucose, BUN, Creatinine)":
        ["Potassium", "Glucose", "BUN", "Creatinine"],
}

MIN_HOURS = 12  # matches existing transformer windowing requirement

print("=" * 70)
print("JOINT PATIENT-LEVEL COVERAGE CHECK")
print("(does a patient have >=1 reading for ALL labs in a given set)")
print("=" * 70)

print(f"\nLoading: {RAW_PATH}")
if not RAW_PATH.exists():
    print("*** FILE NOT FOUND. Update RAW_PATH at the top of this script. ***")
    raise SystemExit(1)

df = pd.read_csv(RAW_PATH)
print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

missing_labs = [c for c in SELECTED_LABS if c not in df.columns]
if missing_labs:
    print(f"*** WARNING: these expected lab columns are missing: {missing_labs} ***")
    SELECTED_LABS = [c for c in SELECTED_LABS if c in df.columns]

# =========================
# APPLY THE SAME >=12 HOUR FILTER your transformer pipeline already uses,
# so these numbers are comparable to your actual transformer-eligible
# population, not the full raw 40,336.
# =========================
hours_per_patient = df.groupby(PATIENT_ID_COL).size()
eligible_ids = hours_per_patient[hours_per_patient >= MIN_HOURS].index
df_eligible = df[df[PATIENT_ID_COL].isin(eligible_ids)].copy()

patient_labels = df_eligible.groupby(PATIENT_ID_COL)[SEPSIS_LABEL_COL].max()
sepsis_ids = set(patient_labels[patient_labels == 1].index)
nosepsis_ids = set(patient_labels[patient_labels == 0].index)

print(f"\nPatients with >= {MIN_HOURS}h stay (transformer-eligible): {len(patient_labels):,}")
print(f"  Sepsis: {len(sepsis_ids):,}")
print(f"  No-sepsis: {len(nosepsis_ids):,}")

# =========================
# PER-PATIENT: does this patient have >=1 reading for EACH lab in the set?
# =========================
def has_reading(group, col):
    return group[col].notna().any()

print("\nComputing per-patient per-lab coverage (this may take a moment)...")
coverage_by_patient = df_eligible.groupby(PATIENT_ID_COL)[SELECTED_LABS].apply(
    lambda g: g.notna().any()
)
# coverage_by_patient: index = Patient_ID, columns = SELECTED_LABS, values = bool

# =========================
# REPORT JOINT COVERAGE FOR EACH SUBSET
# =========================
print("\n" + "=" * 70)
print("JOINT COVERAGE RESULTS")
print("=" * 70)

for label, labs in SUBSETS_TO_CHECK.items():
    labs_present = [l for l in labs if l in coverage_by_patient.columns]
    if not labs_present:
        continue

    has_all = coverage_by_patient[labs_present].all(axis=1)

    sepsis_has_all = has_all.loc[has_all.index.isin(sepsis_ids)]
    nosepsis_has_all = has_all.loc[has_all.index.isin(nosepsis_ids)]

    n_sepsis_qualifying = sepsis_has_all.sum()
    n_nosepsis_qualifying = nosepsis_has_all.sum()
    pct_sepsis = n_sepsis_qualifying / len(sepsis_ids) * 100
    pct_nosepsis = n_nosepsis_qualifying / len(nosepsis_ids) * 100

    # what a balanced (50/50) dataset built from these patients would look like
    balanced_n_per_class = min(n_sepsis_qualifying, n_nosepsis_qualifying)
    balanced_total = balanced_n_per_class * 2

    print(f"\n--- {label} ---")
    print(f"Labs: {labs_present}")
    print(f"Sepsis patients with ALL labs covered:    {n_sepsis_qualifying:>6,} / {len(sepsis_ids):,} ({pct_sepsis:.1f}%)")
    print(f"No-sepsis patients with ALL labs covered: {n_nosepsis_qualifying:>6,} / {len(nosepsis_ids):,} ({pct_nosepsis:.1f}%)")
    print(f"=> Balanced 50/50 dataset size if built from this subset: "
          f"{balanced_total:,} total ({balanced_n_per_class:,} per class)")

print("\n" + "=" * 70)
print("HOW TO READ THIS")
print("=" * 70)
print("""
- "Balanced 50/50 dataset size" is the number you'd actually have to train
  on if you restrict to patients with real (non-imputed) values for every
  lab in that subset.
- Compare this to your current transformer dataset size (4,827 patients,
  ~3,861 train / 966 test after the 80/20 split) to judge whether this is
  enough to train on, or too small.
- If "All 10 selected labs" comes back too small, the smaller subsets give
  you a sense of the tradeoff: fewer labs but a larger, less-restricted
  dataset, vs. more labs but fewer patients to learn from.
- This approach avoids imputation entirely (only real measured values),
  but introduces a different caveat: patients who got a full lab panel
  drawn may be more closely/differently monitored than typical patients,
  which is a selection effect worth disclosing if you go this route.
""")