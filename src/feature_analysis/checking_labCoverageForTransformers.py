import pandas as pd
import numpy as np
from pathlib import Path

# =========================
# CONFIGURATION
# =========================
RAW_PATH = Path.home() / "Desktop" / "septic-shock-research-github" / "data" / "raw" / "physionet_2019.csv"

# Columns we expect to be identifiers/labels rather than clinical features.
# Anything not in this list is treated as a candidate clinical variable.
KNOWN_NON_CLINICAL = [
    "Patient_ID", "patient_id", "Hour", "hour",
    "SepsisLabel", "sepsis_label", "Unnamed: 0",
]

# Patient ID / label columns — script will try to auto-detect these from
# the actual header if the exact names below aren't present.
PATIENT_ID_GUESSES = ["Patient_ID", "patient_id", "PatientID"]
SEPSIS_LABEL_GUESSES = ["SepsisLabel", "sepsis_label", "Sepsis_Label"]

print("=" * 70)
print("STEP 1 — INSPECT THE RAW FILE")
print("=" * 70)

print(f"\nLoading: {RAW_PATH}")
if not RAW_PATH.exists():
    print(f"\n*** FILE NOT FOUND at this path. ***")
    print("Update RAW_PATH at the top of this script to the correct location,")
    print("then rerun.")
    raise SystemExit(1)

df = pd.read_csv(RAW_PATH)
print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"\nAll columns found ({len(df.columns)}):")
for c in df.columns:
    print(f"  - {c}")

# =========================
# AUTO-DETECT PATIENT ID + SEPSIS LABEL COLUMNS
# =========================
patient_id_col = next((c for c in PATIENT_ID_GUESSES if c in df.columns), None)
sepsis_label_col = next((c for c in SEPSIS_LABEL_GUESSES if c in df.columns), None)

if patient_id_col is None or sepsis_label_col is None:
    print("\n*** Could not auto-detect Patient_ID and/or SepsisLabel columns. ***")
    print(f"Patient_ID guess result: {patient_id_col}")
    print(f"SepsisLabel guess result: {sepsis_label_col}")
    print("Update PATIENT_ID_GUESSES / SEPSIS_LABEL_GUESSES at the top of this")
    print("script with the correct column names from the list above, then rerun.")
    raise SystemExit(1)

print(f"\nUsing Patient ID column: '{patient_id_col}'")
print(f"Using Sepsis label column: '{sepsis_label_col}'")

# =========================
# IDENTIFY CANDIDATE CLINICAL COLUMNS
# (everything that isn't an identifier/label/known non-clinical column)
# =========================
candidate_cols = [c for c in df.columns if c not in KNOWN_NON_CLINICAL
                  and c != patient_id_col and c != sepsis_label_col]

print(f"\nCandidate clinical columns to check ({len(candidate_cols)}):")
print(candidate_cols)

# =========================
# STEP 2 — RECORD-LEVEL MISSINGNESS (quick overview)
# =========================
print("\n" + "=" * 70)
print("STEP 2 — RECORD-LEVEL MISSINGNESS (% of all hourly rows that are null)")
print("=" * 70)
print(f"{'Column':<20} {'% missing (record-level)':>28}")
print("-" * 50)
record_missing = {}
for col in candidate_cols:
    pct_missing = df[col].isna().mean() * 100
    record_missing[col] = pct_missing
    print(f"{col:<20} {pct_missing:>27.1f}%")

# =========================
# STEP 3 — PATIENT-LEVEL LABELS
# =========================
patient_labels = df.groupby(patient_id_col)[sepsis_label_col].max()
sepsis_ids = patient_labels[patient_labels == 1].index
nosepsis_ids = patient_labels[patient_labels == 0].index

print("\n" + "=" * 70)
print("STEP 3 — PATIENT POPULATION")
print("=" * 70)
print(f"Total patients: {len(patient_labels):,}")
print(f"Sepsis patients: {len(sepsis_ids):,}")
print(f"No-sepsis patients: {len(nosepsis_ids):,}")

# =========================
# STEP 4 — PATIENT-LEVEL COVERAGE, SPLIT BY SEPSIS VS NO-SEPSIS
# (does this patient have at least one non-null reading during their stay)
# =========================
def patient_level_coverage(df, patient_ids, col, patient_id_col):
    sub = df[df[patient_id_col].isin(patient_ids)]
    has_reading = sub.groupby(patient_id_col)[col].apply(lambda x: x.notna().any())
    return has_reading.mean() * 100

print("\n" + "=" * 70)
print("STEP 4 — PATIENT-LEVEL COVERAGE (>=1 reading during stay), split by class")
print("=" * 70)
print(f"{'Column':<20} {'Sepsis cov.':>14} {'No-sepsis cov.':>17} {'Gap':>8} {'Record-level missing':>22}")
print("-" * 85)

rows = []
for col in candidate_cols:
    sepsis_cov = patient_level_coverage(df, sepsis_ids, col, patient_id_col)
    nosepsis_cov = patient_level_coverage(df, nosepsis_ids, col, patient_id_col)
    gap = sepsis_cov - nosepsis_cov
    rows.append((col, sepsis_cov, nosepsis_cov, gap, record_missing[col]))

# Sort by no-sepsis coverage descending — the binding constraint for adding
# a feature to a mixed-class transformer population is the WORSE-covered
# class, which (per Phase 7 findings) is usually no-sepsis.
rows.sort(key=lambda r: r[2], reverse=True)

for col, sepsis_cov, nosepsis_cov, gap, rec_missing in rows:
    print(f"{col:<20} {sepsis_cov:>13.1f}% {nosepsis_cov:>16.1f}% {gap:>+7.1f}% {rec_missing:>21.1f}%")

# =========================
# INTERPRETATION GUIDE
# =========================
print("\n" + "=" * 70)
print("HOW TO READ THIS")
print("=" * 70)
print("""
- Sorted by no-sepsis coverage (descending) — that's the binding constraint
  for the transformer's mixed-class population (40% sepsis / 60% no-sepsis).
- High coverage in BOTH columns = safest to add (real signal, low imputation).
- High sepsis coverage but low no-sepsis coverage = risky: the model may
  learn "this lab was measured at all" as a proxy for sepsis, which is a
  shortcut rather than real physiological signal.
- Record-level missing % (rightmost column) explains why a column was
  dropped during the ORIGINAL 11-feature pruning (>80% missing overall) —
  but patient-level coverage is the number that actually matters for
  whether it's usable as a model input.
- Compare the known Phase 7 numbers (computed only among sepsis patients)
  as a sanity check: Creatinine ~93.2%, Platelets ~91.7%, FiO2 ~74.9%,
  Lactate ~62%. If this script's 'Sepsis cov.' column roughly matches those,
  the script is working correctly on the same population.
""")