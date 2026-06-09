import pandas as pd
import numpy as np
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT_DIR / "data" / "raw" / "physionet_2019.csv"

print("Loading raw dataset...")
df = pd.read_csv(CSV_PATH)
if df.columns[0] == "Unnamed: 0":
    df = df.drop(columns=df.columns[0])

# ─────────────────────────────────────────
# STEP 1: Isolate sepsis patients only
# ─────────────────────────────────────────
patient_labels = df.groupby("Patient_ID")["SepsisLabel"].max()
sepsis_ids = patient_labels[patient_labels == 1].index
df_sepsis = df[df["Patient_ID"].isin(sepsis_ids)].copy()

print(f"\nTotal sepsis patients: {len(sepsis_ids):,}")

# ─────────────────────────────────────────
# STEP 2: Per patient — get worst case values
# (min MAP, max Lactate — dangerous directions)
# ─────────────────────────────────────────
patient_stats = df_sepsis.groupby("Patient_ID").agg(
    MAP_min=("MAP", "min"),
    Lactate_max=("Lactate", "max"),
    MAP_available=("MAP", lambda x: x.notna().any()),
    Lactate_available=("Lactate", lambda x: x.notna().any()),
).reset_index()

# ─────────────────────────────────────────
# STEP 3: Check how many meet each criterion
# ─────────────────────────────────────────
map_low = patient_stats["MAP_min"] < 65
lactate_high = patient_stats["Lactate_max"] > 2.0
both = map_low & lactate_high
lactate_missing = ~patient_stats["Lactate_available"]
map_low_only = map_low & lactate_missing

print("\n--- SEPTIC SHOCK COHORT ANALYSIS ---")
print(f"Sepsis patients with MAP < 65 (at any point):         {map_low.sum():,} ({map_low.mean()*100:.1f}%)")
print(f"Sepsis patients with Lactate > 2 (at any point):      {lactate_high.sum():,} ({lactate_high.mean()*100:.1f}%)")
print(f"Sepsis patients with BOTH MAP < 65 AND Lactate > 2:   {both.sum():,} ({both.mean()*100:.1f}%)")
print(f"Sepsis patients with MAP < 65 but NO Lactate reading:  {map_low_only.sum():,} ({map_low_only.mean()*100:.1f}%)")
print(f"Sepsis patients with missing Lactate entirely:         {lactate_missing.sum():,} ({lactate_missing.mean()*100:.1f}%)")

# ─────────────────────────────────────────
# STEP 4: Show distribution of MAP and Lactate
# among sepsis patients to understand the spread
# ─────────────────────────────────────────
print("\n--- MAP DISTRIBUTION (min per sepsis patient) ---")
print(patient_stats["MAP_min"].describe().round(2))

print("\n--- LACTATE DISTRIBUTION (max per sepsis patient, where available) ---")
print(patient_stats[patient_stats["Lactate_available"]]["Lactate_max"].describe().round(2))

# ─────────────────────────────────────────
# STEP 5: Simulate septic shock label
# and show what the cohort would look like
# ─────────────────────────────────────────
print("\n--- SIMULATED SEPTIC SHOCK LABEL ---")
print("Definition: SepsisLabel=1 AND MAP_min < 65 AND Lactate_max > 2")
print(f"Patients labeled septic shock: {both.sum():,}")
print(f"Patients labeled sepsis only:  {(~both).sum():,} (out of {len(sepsis_ids):,} sepsis patients)")
print(f"\nNote: {map_low_only.sum():,} additional patients have MAP < 65 but no Lactate reading.")
print("These are ambiguous — could be septic shock but we cannot confirm.")