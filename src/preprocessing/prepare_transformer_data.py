import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# ============================================================
# CONFIGURATION
# ============================================================
ROOT_DIR = Path(__file__).resolve().parents[2]

# Read from the RAW physionet file (all 44 columns), NOT physionet_balanced.csv.
# physionet_balanced.csv only has the original 11 vitals — the labs we want to
# add were already pruned out of it.
CSV_PATH = ROOT_DIR / "data" / "raw" / "physionet_2019.csv"
OUTPUT_DIR = ROOT_DIR / "data" / "processed" / "transformer"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_SIZE = 12       # hours before label — matching KA-Transformer paper
MIN_HOURS = 12         # exclude patients with fewer than this many hours
TEST_SIZE = 0.2        # 80/20 train/test split
RANDOM_STATE = 42

# The original 11 features (unchanged from the vitals-only baseline).
BASE_FEATURES = [
    "HR", "O2Sat", "Temp", "SBP", "MAP",
    "DBP", "Resp", "Age", "Gender",
    "HospAdmTime", "ICULOS",
]

# The 10 NEW lab features being added. These were selected because they have
# high patient-level coverage in BOTH the sepsis and no-sepsis groups (small
# coverage gap), so adding them does not leak label information through
# missingness patterns the way sparse labs (FiO2, Lactate, pH, PaCO2) would.
LAB_FEATURES = [
    "BUN", "Creatinine", "Hgb", "Platelets", "WBC",
    "Hct", "Potassium", "Glucose", "Magnesium", "Calcium",
]

# Full feature list the transformer will see: 11 + 10 = 21 features.
FEATURE_COLS = BASE_FEATURES + LAB_FEATURES

print("=" * 60)
print("TRANSFORMER DATA PREPARATION (21-feature version)")
print(f"Window size: {WINDOW_SIZE} hours")
print(f"Base features: {len(BASE_FEATURES)} | Lab features: {len(LAB_FEATURES)} "
      f"| Total: {len(FEATURE_COLS)}")
print("=" * 60)


# ============================================================
# STEP 1 — LOAD RAW DATA
# ============================================================
print("\n[1] Loading raw PhysioNet dataset...")
df = pd.read_csv(CSV_PATH)

# Drop the leftover unnamed index column if present.
if df.columns[0] == "Unnamed: 0":
    df = df.drop(columns=df.columns[0])

print(f"    Raw shape: {df.shape}")

# Patient-level label: a patient is "sepsis" if they are ever labeled sepsis.
patient_labels = df.groupby("Patient_ID")["SepsisLabel"].max()


# ============================================================
# STEP 2 — EXCLUDE SHORT STAYS (< 12 hours)
# ============================================================
print("\n[2] Excluding patients with < 12 hours of data...")
hours_per_patient = df.groupby("Patient_ID").size()
long_enough_ids = hours_per_patient[hours_per_patient >= MIN_HOURS].index
df = df[df["Patient_ID"].isin(long_enough_ids)].copy()

print(f"    Patients remaining: {len(long_enough_ids):,}")


# ============================================================
# STEP 3 — JOINT COVERAGE FILTER
# Keep only patients who have at least one real reading for EVERY lab in
# LAB_FEATURES. This is what lets us avoid imputing labs for patients who
# never had them measured — we only keep patients with genuine values.
# ============================================================
print("\n[3] Filtering to patients with all 10 labs measured...")

# For each patient, check whether every lab column has >= 1 non-null value.
lab_coverage = df.groupby("Patient_ID")[LAB_FEATURES].apply(
    lambda g: g.notna().any()  # per-lab: does this patient have any reading?
)
has_all_labs = lab_coverage.all(axis=1)  # True only if ALL labs are covered
covered_ids = has_all_labs[has_all_labs].index

df = df[df["Patient_ID"].isin(covered_ids)].copy()

# Recompute labels for the surviving population.
patient_labels = df.groupby("Patient_ID")["SepsisLabel"].max()
n_sepsis = int((patient_labels == 1).sum())
n_nosepsis = int((patient_labels == 0).sum())

print(f"    Patients with all labs covered: {len(covered_ids):,}")
print(f"      Sepsis:    {n_sepsis:,}")
print(f"      No-sepsis: {n_nosepsis:,}")


# ============================================================
# STEP 4 — BALANCE 50/50
# Undersample the majority (no-sepsis) class down to the sepsis count, so the
# final dataset is a balanced 50/50 split — consistent with the rest of the
# project (and the balanced-dataset caveat that gets disclosed in the paper).
# ============================================================
print("\n[4] Balancing dataset to 50/50...")

sepsis_ids = patient_labels[patient_labels == 1].index.tolist()
nosepsis_ids = patient_labels[patient_labels == 0].index.tolist()

# Reproducible undersample of the no-sepsis group.
rng = np.random.RandomState(RANDOM_STATE)
nosepsis_sampled = rng.choice(nosepsis_ids, size=len(sepsis_ids), replace=False)

balanced_ids = list(sepsis_ids) + list(nosepsis_sampled)
df = df[df["Patient_ID"].isin(balanced_ids)].copy()

print(f"    Balanced total: {len(balanced_ids):,} "
      f"({len(sepsis_ids):,} sepsis + {len(nosepsis_sampled):,} no-sepsis)")


# ============================================================
# STEP 5 — FORWARD FILL + BACKWARD FILL (per patient)
# Labs are only measured once or twice a day, so most hourly rows are NaN even
# for a "covered" patient. ffill carries the last known value forward through
# the patient's timeline; bfill fills any gaps at the very start of the stay.
# This is the SAME imputation approach used in the XGBoost pipeline.
# ============================================================
print("\n[5] Forward/backward filling missing values per patient...")

df = df.sort_values(["Patient_ID", "Hour"]).reset_index(drop=True)
df = df.set_index("Patient_ID")
df = df.groupby("Patient_ID").ffill()   # fill forward within each patient
df = df.groupby("Patient_ID").bfill()   # fill backward within each patient
df = df.reset_index()

# Any value still missing here means it was NaN for that patient's entire stay.
# For labs that shouldn't happen (STEP 3 guaranteed >=1 reading), but vitals
# could still have rare all-NaN cases — those get zeroed after normalization.


# ============================================================
# STEP 6 — EXTRACT THE 12-HOUR WINDOW PER PATIENT
# Sepsis patients: the 12 hours immediately BEFORE their first sepsis label
#                  (predict-at-onset framing — we never show the model the
#                  hours where sepsis is already confirmed).
# No-sepsis patients: the last 12 hours of their stay.
# ============================================================
print(f"\n[6] Extracting {WINDOW_SIZE}-hour window per patient...")

X_list = []
y_list = []
patient_id_list = []
skipped = 0

for patient_id, group in df.groupby("Patient_ID"):
    group = group.sort_values("Hour").reset_index(drop=True)
    label = patient_labels[patient_id]

    if label == 1:
        # ----- Sepsis patient: 12 hours before first sepsis label -----
        sepsis_rows = group[group["SepsisLabel"] == 1]
        if len(sepsis_rows) == 0:
            skipped += 1
            continue

        first_sepsis_idx = sepsis_rows.index[0]
        pre_sepsis = group.loc[:first_sepsis_idx - 1]  # rows strictly before onset

        if len(pre_sepsis) < WINDOW_SIZE:
            skipped += 1
            continue

        window = pre_sepsis.tail(WINDOW_SIZE)[FEATURE_COLS].values

    else:
        # ----- No-sepsis patient: last 12 hours of stay -----
        if len(group) < WINDOW_SIZE:
            skipped += 1
            continue

        window = group.tail(WINDOW_SIZE)[FEATURE_COLS].values

    # Safety check: window must be exactly [12 timesteps x 21 features].
    if window.shape != (WINDOW_SIZE, len(FEATURE_COLS)):
        skipped += 1
        continue

    X_list.append(window)
    y_list.append(label)
    patient_id_list.append(patient_id)

X = np.array(X_list)   # shape: [N, 12, 21]
y = np.array(y_list)   # shape: [N]

print(f"    Patients included: {len(X_list):,}")
print(f"    Patients skipped:  {skipped:,} (insufficient pre-onset hours)")
print(f"    X shape: {X.shape}  (patients x timesteps x features)")
print(f"    Sepsis: {y.sum():,} ({y.mean()*100:.1f}%) | "
      f"No-sepsis: {(1-y).sum():,} ({(1-y).mean()*100:.1f}%)")


# ============================================================
# STEP 7 — TRAIN/TEST SPLIT (before normalization, to avoid leakage)
# Split FIRST so normalization statistics are computed on train data only.
# ============================================================
print("\n[7] Splitting data (80/20, stratified)...")

N, T, F = X.shape
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

print(f"    Train: {X_train_raw.shape} | Sepsis: {y_train.sum():,} ({y_train.mean()*100:.1f}%)")
print(f"    Test:  {X_test_raw.shape}  | Sepsis: {y_test.sum():,} ({y_test.mean()*100:.1f}%)")


# ============================================================
# STEP 8 — NORMALIZE (mean/std fit on TRAIN only)
# z-score each feature using statistics computed from the training set only,
# then apply those same statistics to the test set.
# ============================================================
print("\n[8] Normalizing features (train-only statistics)...")

X_train_flat = X_train_raw.reshape(-1, F)

feat_mean = np.nanmean(X_train_flat, axis=0)
feat_std = np.nanstd(X_train_flat, axis=0)
feat_std[feat_std == 0] = 1  # guard against divide-by-zero

X_train = ((X_train_raw.reshape(-1, F) - feat_mean) / feat_std).reshape(X_train_raw.shape)
X_test = ((X_test_raw.reshape(-1, F) - feat_mean) / feat_std).reshape(X_test_raw.shape)

# Zero-fill any residual NaNs (e.g. a vital that was all-NaN for a patient).
X_train = np.nan_to_num(X_train, nan=0.0)
X_test = np.nan_to_num(X_test, nan=0.0)

print(f"    NaNs after normalization — train: {np.isnan(X_train).sum()} | "
      f"test: {np.isnan(X_test).sum()}")


# ============================================================
# STEP 9 — SAVE
# ============================================================
print("\n[9] Saving .npy files...")

np.save(OUTPUT_DIR / "X_train.npy", X_train)
np.save(OUTPUT_DIR / "X_test.npy", X_test)
np.save(OUTPUT_DIR / "y_train.npy", y_train)
np.save(OUTPUT_DIR / "y_test.npy", y_test)
np.save(OUTPUT_DIR / "feat_mean.npy", feat_mean)
np.save(OUTPUT_DIR / "feat_std.npy", feat_std)

with open(OUTPUT_DIR / "feature_cols.txt", "w") as f:
    for col in FEATURE_COLS:
        f.write(col + "\n")

print(f"    Saved to: {OUTPUT_DIR}")


# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Total patients used:     {N:,}")
print(f"Window size:             {WINDOW_SIZE} hours before sepsis onset")
print(f"Features ({len(FEATURE_COLS)}):           {FEATURE_COLS}")
print(f"Input shape per patient: [{WINDOW_SIZE} timesteps x {len(FEATURE_COLS)} features]")
print(f"Train set:               {X_train.shape[0]:,} patients")
print(f"Test set:                {X_test.shape[0]:,} patients")
print("=" * 60)
print("DONE")