import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# =========================
# CONFIGURATION
# =========================
ROOT_DIR = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT_DIR / "data" / "processed" / "physionet_balanced.csv"
OUTPUT_DIR = ROOT_DIR / "data" / "processed" / "transformer"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_SIZE = 12  # hours before label — matching KA-Transformer paper
MIN_HOURS = 12    # exclude patients with fewer than this
TEST_SIZE = 0.2   # 80/20 split
RANDOM_STATE = 42

# Features to use — your 11 retained features
FEATURE_COLS = [
    "HR", "O2Sat", "Temp", "SBP", "MAP",
    "DBP", "Resp", "Age", "Gender",
    "HospAdmTime", "ICULOS"
]

print("=" * 60)
print("TRANSFORMER DATA PREPARATION")
print(f"Window size: {WINDOW_SIZE} hours")
print(f"Features: {len(FEATURE_COLS)}")
print("=" * 60)

# =========================
# STEP 1 — LOAD DATA
# =========================
print("\nLoading balanced dataset...")
df = pd.read_csv(CSV_PATH)
print(f"Raw shape: {df.shape}")

# Get patient-level label
patient_labels = df.groupby("Patient_ID")["SepsisLabel"].max()

# =========================
# STEP 2 — EXCLUDE SHORT STAYS
# =========================
print("\nExcluding patients with < 12 hours...")
hours_per_patient = df.groupby("Patient_ID").size()
valid_ids = hours_per_patient[hours_per_patient >= MIN_HOURS].index
df = df[df["Patient_ID"].isin(valid_ids)].copy()

print(f"Patients remaining: {len(valid_ids):,}")
print(f"Patients excluded: {(hours_per_patient < MIN_HOURS).sum():,}")

# =========================
# STEP 3 — EXTRACT LAST 12 HOURS PER PATIENT
# =========================
print(f"\nExtracting last {WINDOW_SIZE} hours per patient...")

X_list = []
y_list = []
patient_id_list = []

for patient_id, group in df.groupby("Patient_ID"):
    # Sort by hour to ensure correct time order
    group = group.sort_values("Hour").reset_index(drop=True)

    # Take last WINDOW_SIZE hours
    window = group.tail(WINDOW_SIZE)[FEATURE_COLS].values

    # Should always be exactly WINDOW_SIZE rows since we excluded < 12h patients
    # but double check
    if len(window) < WINDOW_SIZE:
        continue  # safety check — shouldn't happen

    X_list.append(window)
    y_list.append(patient_labels[patient_id])
    patient_id_list.append(patient_id)

X = np.array(X_list)  # shape: [N, 12, 11]
y = np.array(y_list)  # shape: [N]

print(f"X shape: {X.shape}  (patients × timesteps × features)")
print(f"y shape: {y.shape}")
print(f"Sepsis patients: {y.sum():,} ({y.mean()*100:.1f}%)")
print(f"No-sepsis patients: {(1-y).sum():,} ({(1-y).mean()*100:.1f}%)")

# =========================
# STEP 4 — NORMALIZE FEATURES
# =========================
print("\nNormalizing features (per feature, across all patients and timesteps)...")

# Reshape to 2D for normalization: [N*T, F]
N, T, F = X.shape
X_flat = X.reshape(-1, F)

# Compute mean and std per feature (ignoring NaN)
feat_mean = np.nanmean(X_flat, axis=0)
feat_std  = np.nanstd(X_flat, axis=0)
feat_std[feat_std == 0] = 1  # prevent division by zero

# Normalize
X_flat_norm = (X_flat - feat_mean) / feat_std

# Reshape back to 3D
X_norm = X_flat_norm.reshape(N, T, F)

# Replace any remaining NaN with 0 (mean after normalization)
X_norm = np.nan_to_num(X_norm, nan=0.0)

print(f"NaN values after normalization: {np.isnan(X_norm).sum()}")

# =========================
# STEP 5 — TRAIN/TEST SPLIT
# =========================
print(f"\nSplitting data (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X_norm, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y  # preserve class balance in both splits
)

print(f"Train: {X_train.shape} | Sepsis: {y_train.sum():,} ({y_train.mean()*100:.1f}%)")
print(f"Test:  {X_test.shape}  | Sepsis: {y_test.sum():,} ({y_test.mean()*100:.1f}%)")

# =========================
# STEP 6 — SAVE
# =========================
print("\nSaving...")
np.save(OUTPUT_DIR / "X_train.npy", X_train)
np.save(OUTPUT_DIR / "X_test.npy",  X_test)
np.save(OUTPUT_DIR / "y_train.npy", y_train)
np.save(OUTPUT_DIR / "y_test.npy",  y_test)

# Save normalization stats for inference later
np.save(OUTPUT_DIR / "feat_mean.npy", feat_mean)
np.save(OUTPUT_DIR / "feat_std.npy",  feat_std)

# Save feature names for reference
with open(OUTPUT_DIR / "feature_cols.txt", "w") as f:
    for col in FEATURE_COLS:
        f.write(col + "\n")

print(f"Saved to: {OUTPUT_DIR}")

# =========================
# SUMMARY
# =========================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Total patients used:     {N:,}")
print(f"Excluded (< 12h stay):   561 (9.6%)")
print(f"Window size:             {WINDOW_SIZE} hours")
print(f"Features:                {F} {FEATURE_COLS}")
print(f"Input shape per patient: [{WINDOW_SIZE} timesteps × {F} features]")
print(f"Train set:               {X_train.shape[0]:,} patients")
print(f"Test set:                {X_test.shape[0]:,} patients")
print(f"Files saved:")
print(f"  X_train.npy  {X_train.shape}")
print(f"  X_test.npy   {X_test.shape}")
print(f"  y_train.npy  {y_train.shape}")
print(f"  y_test.npy   {y_test.shape}")
print(f"  feat_mean.npy / feat_std.npy (for inference normalization)")
print("=" * 60)
print("DONE")