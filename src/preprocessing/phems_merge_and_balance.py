import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.utils import resample

# =========================
# CONFIGURATION
# =========================
ROOT_DIR = Path(__file__).resolve().parents[2]

INPUT_DIR = ROOT_DIR / "data" / "raw" / "phems_data" / "training_data"
OUTPUT_DIR = ROOT_DIR / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FLAT = OUTPUT_DIR / "phems_flat.csv"
OUTPUT_CLEAN = OUTPUT_DIR / "phems_clean.csv"
BALANCED_PATH = OUTPUT_DIR / "phems_balanced.csv"

# =========================
# 1) LOAD ALL FILES
# =========================
print("\nLoading files...")

meds  = pd.read_csv(INPUT_DIR / "measurement_meds_train.csv")
labs  = pd.read_csv(INPUT_DIR / "measurement_lab_train.csv")
obs   = pd.read_csv(INPUT_DIR / "measurement_observation_train.csv")
demo  = pd.read_csv(INPUT_DIR / "person_demographics_episode_train.csv")
label = pd.read_csv(INPUT_DIR / "SepsisLabel_train.csv")

print(f"  measurement_meds_train:        {meds.shape}")
print(f"  measurement_lab_train:         {labs.shape}")
print(f"  measurement_observation_train: {obs.shape}")
print(f"  person_demographics_episode:   {demo.shape}")
print(f"  SepsisLabel_train:             {label.shape}")

# =========================
# 2) STANDARDIZE DATETIME
# =========================
print("\nStandardizing datetime columns...")
for df, col in [(meds, "measurement_datetime"),
                (labs, "measurement_datetime"),
                (obs,  "measurement_datetime"),
                (label,"measurement_datetime")]:
    df[col] = pd.to_datetime(df[col])

# =========================
# 3) MERGE TIME-SERIES FILES
# =========================
# All time-series files share person_id + measurement_datetime as join keys
# We use outer joins so no rows are lost from any file
print("\nMerging time-series files...")

JOIN_KEYS = ["person_id", "measurement_datetime"]

# Start with meds (vitals) as base — largest file
df = meds.copy()
print(f"  Base (meds):         {len(df):,} rows")

# Merge labs
df = df.merge(
    labs.drop(columns=["visit_occurrence_id"], errors="ignore"),
    on=JOIN_KEYS, how="outer"
)
print(f"  After labs merge:    {len(df):,} rows")

# Merge observations (GCS, pupils, pulse etc.)
df = df.merge(
    obs.drop(columns=["visit_occurrence_id"], errors="ignore"),
    on=JOIN_KEYS, how="outer"
)
print(f"  After obs merge:     {len(df):,} rows")

# Merge sepsis label
df = df.merge(label, on=JOIN_KEYS, how="left")
print(f"  After label merge:   {len(df):,} rows")

# =========================
# 4) MERGE DEMOGRAPHICS
# =========================
# Demographics are per-patient (not per-timestamp), join on person_id only
print("\nMerging demographics...")
demo_slim = demo[["person_id", "age_in_months", "gender"]].drop_duplicates(subset="person_id")
df = df.merge(demo_slim, on="person_id", how="left")
print(f"  After demo merge:    {len(df):,} rows")

# =========================
# 5) CLEAN UP DUPLICATE COLUMNS
# =========================
print("\nCleaning duplicate columns...")

# visit_occurrence_id may appear twice (from meds and labs) — keep one
visit_cols = [c for c in df.columns if "visit_occurrence_id" in c.lower()]
if len(visit_cols) > 1:
    df = df.drop(columns=visit_cols[1:])
if visit_cols:
    df = df.rename(columns={visit_cols[0]: "visit_occurrence_id"})

print(f"  Columns after cleanup: {len(df.columns)}")

# =========================
# 6) SORT BY PATIENT + TIME
# =========================
df = df.sort_values(["person_id", "measurement_datetime"]).reset_index(drop=True)

# =========================
# 7) SAVE FLAT FILE
# =========================
print(f"\nSaving flat file...")
df.to_csv(OUTPUT_FLAT, index=False)
print(f"  Saved: PHEMS_flat.csv  ({len(df):,} rows x {len(df.columns)} cols)")

# =========================
# 8) PRINT SUMMARY STATS
# =========================
print("\n--- FLAT FILE SUMMARY ---")
n_patients = df["person_id"].nunique()
label_counts = df["SepsisLabel"].value_counts(dropna=True)
print(f"Total rows:         {len(df):,}")
print(f"Total patients:     {n_patients:,}")
print(f"Total columns:      {len(df.columns)}")
print(f"SepsisLabel=0:      {int(label_counts.get(0.0, 0)):,}")
print(f"SepsisLabel=1:      {int(label_counts.get(1.0, 0)):,}")
print(f"Label missing:      {df['SepsisLabel'].isnull().sum():,}")

print("\nAll columns:")
for c in df.columns:
    missing_pct = df[c].isnull().mean() * 100
    print(f"  {c:<55} {missing_pct:5.1f}% missing")

# =========================
# 9) PRODUCE CLEAN VERSION
# =========================
# Drop columns with >80% missing (same threshold as PhysioNet script)
print("\n--- PRODUCING CLEAN VERSION (drop >80% missing) ---")
MISSING_THRESHOLD = 0.80
NON_CLINICAL = ["visit_occurrence_id", "person_id", "measurement_datetime",
                "visit_start_date", "birth_datetime", "SepsisLabel"]

feature_cols = [c for c in df.columns if c not in NON_CLINICAL]
missing_pct  = df[feature_cols].isnull().mean()
drop_cols    = missing_pct[missing_pct > MISSING_THRESHOLD].index.tolist()

print(f"Columns dropped (>{MISSING_THRESHOLD*100:.0f}% missing): {len(drop_cols)}")
for c in drop_cols:
    print(f"  - {c}  ({missing_pct[c]*100:.1f}% missing)")

df_clean = df.drop(columns=drop_cols)
remaining = [c for c in df_clean.columns if c not in NON_CLINICAL]
print(f"\nRemaining clinical columns ({len(remaining)}):")
for c in remaining:
    print(f"  {c}")

df_clean.to_csv(OUTPUT_CLEAN, index=False)
print(f"\nSaved: PHEMS_clean.csv  ({len(df_clean):,} rows x {len(df_clean.columns)} cols)")


# =========================
# 10) BALANCE DATASET (50/50)
# =========================
print("\n--- BALANCING DATASET (50/50) ---")

# Balance at patient level using person_id
patient_labels = df_clean.groupby("person_id")["SepsisLabel"].max()
sepsis_ids    = patient_labels[patient_labels == 1].index.tolist()
no_sepsis_ids = patient_labels[patient_labels == 0].index.tolist()

print(f"Sepsis patients:    {len(sepsis_ids):,}")
print(f"No-sepsis patients: {len(no_sepsis_ids):,}")

no_sepsis_sampled = resample(no_sepsis_ids,
                             n_samples=len(sepsis_ids),
                             random_state=42,
                             replace=False)
balanced_ids  = list(sepsis_ids) + list(no_sepsis_sampled)
df_balanced   = df_clean[df_clean["person_id"].isin(balanced_ids)].copy()

bal_counts = df_balanced.groupby("person_id")["SepsisLabel"].max().value_counts()
print(f"Balanced: {int(bal_counts.get(0.0,0)):,} no-sepsis + {int(bal_counts.get(1.0,0)):,} sepsis patients")
print(f"Total rows: {len(df_balanced):,}")

df_balanced.to_csv(BALANCED_PATH, index=False)
print(f"Saved: PHEMS_balanced.csv  ({len(df_balanced):,} rows x {len(df_balanced.columns)} cols)")

# =========================
# DONE
# =========================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Flat file:   PHEMS_flat.csv   — all variables, all rows")
print(f"Clean file:    PHEMS_clean.csv    — variables with <80% missing only")
print(f"Balanced file: PHEMS_balanced.csv — clean + 50/50 patient balance")
print(f"\nNext step: Task 2 — Random Forest on PHEMS_balanced.csv and Dataset_balanced.csv")
print("\nDONE!")
