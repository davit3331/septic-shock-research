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
# Separate folder from the fixed-12h-window version's output
# (data/processed/transformer/) — running this script must never silently
# overwrite those files. Keeping both lets you compare results and re-run
# either pipeline without losing the other's data.
OUTPUT_DIR = ROOT_DIR / "data" / "processed" / "transformer_sliding"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_SIZE = 12       # hours per window — matching KA-Transformer paper
STRIDE = 6             # hours between the START of consecutive sliding
                        # windows, TRAIN PATIENTS ONLY. 6h stride with a 12h
                        # window = 50% overlap between consecutive windows,
                        # i.e. a patient with a 30h pre-onset/stay region
                        # contributes multiple windows instead of just one.
MAX_WINDOWS_PER_PATIENT = 5
                        # Caps how many sliding windows a single TRAIN
                        # patient can contribute. Without this, sepsis
                        # patients (whose pre-onset region tends to be much
                        # longer than a no-sepsis patient's whole stay) end
                        # up massively over-represented — a first run of
                        # this pipeline produced 65% sepsis windows from a
                        # 50/50 patient split, with sepsis patients averaging
                        # ~10 windows each vs ~5 for no-sepsis. Capping pulls
                        # the window-level balance back toward the
                        # patient-level balance and prevents a handful of
                        # long-stay patients from dominating training with
                        # many near-duplicate (50%-overlapping) windows.

LOOKBACK_HOURS = 36    # SEPSIS TRAIN windows only: restrict sliding windows
                        # to the last LOOKBACK_HOURS before onset, instead of
                        # sliding across a sepsis patient's ENTIRE pre-onset
                        # region. Reasoning: sepsis patients can look
                        # physiologically stable for a long time before onset
                        # — a window from 40 hours before onset may look
                        # almost identical to a patient who never develops
                        # sepsis at all, yet earlier code labeled it
                        # "sepsis = 1" just like the window right before
                        # onset. That's a weak/noisy positive example. This
                        # bound keeps every sepsis training window within a
                        # clinically-relevant distance of actual onset. 36h
                        # is chosen so the max 5 windows (36-12)/6+1 = 5 lines
                        # up exactly with MAX_WINDOWS_PER_PATIENT. No-sepsis
                        # patients are NOT bounded this way — every hour of a
                        # no-sepsis patient's stay is validly labeled
                        # "no sepsis," so there's no equivalent weak-label
                        # problem there.
MIN_HOURS = 12          # exclude patients with fewer than this many hours
TEST_SIZE = 0.2        # 80/20 train/test split, at the PATIENT level
RANDOM_STATE = 42

# The original 11 features (unchanged from the vitals-only baseline).
BASE_FEATURES = [
    "HR", "O2Sat", "Temp", "SBP", "MAP",
    "DBP", "Resp", "Age", "Gender",
    "HospAdmTime", "ICULOS",
]

# The 10 lab features (unchanged from the 21-feature version). Selected for
# high patient-level coverage in BOTH sepsis and no-sepsis groups (small
# coverage gap), so adding them does not leak label information through
# missingness patterns the way sparse labs (FiO2, Lactate, pH, PaCO2) would.
LAB_FEATURES = [
    "BUN", "Creatinine", "Hgb", "Platelets", "WBC",
    "Hct", "Potassium", "Glucose", "Magnesium", "Calcium",
]

# Full feature list the transformer will see: 11 + 10 = 21 features.
FEATURE_COLS = BASE_FEATURES + LAB_FEATURES

print("=" * 60)
print("TRANSFORMER DATA PREPARATION (sliding-window version)")
print(f"Window size: {WINDOW_SIZE}h | Train stride: {STRIDE}h | Test: 1 window/patient")
print(f"Base features: {len(BASE_FEATURES)} | Lab features: {len(LAB_FEATURES)} "
      f"| Total: {len(FEATURE_COLS)}")
print("=" * 60)


# ============================================================
# STEP 1 — LOAD RAW DATA
# ============================================================
print("\n[1] Loading raw PhysioNet dataset...")
df = pd.read_csv(CSV_PATH)

if df.columns[0] == "Unnamed: 0":
    df = df.drop(columns=df.columns[0])

print(f"    Raw shape: {df.shape}")

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
# LAB_FEATURES, so we never impute a lab for a patient who never had it
# measured — only genuine values get carried forward.
# ============================================================
print("\n[3] Filtering to patients with all 10 labs measured...")

lab_coverage = df.groupby("Patient_ID")[LAB_FEATURES].apply(
    lambda g: g.notna().any()
)
has_all_labs = lab_coverage.all(axis=1)
covered_ids = has_all_labs[has_all_labs].index

df = df[df["Patient_ID"].isin(covered_ids)].copy()

patient_labels = df.groupby("Patient_ID")["SepsisLabel"].max()
n_sepsis = int((patient_labels == 1).sum())
n_nosepsis = int((patient_labels == 0).sum())

print(f"    Patients with all labs covered: {len(covered_ids):,}")
print(f"      Sepsis:    {n_sepsis:,}")
print(f"      No-sepsis: {n_nosepsis:,}")


# ============================================================
# STEP 4 — BALANCE 50/50
# Undersample the majority (no-sepsis) class down to the sepsis count, at
# the PATIENT level, so the pool of patients we split and window from is
# balanced going in.
# ============================================================
print("\n[4] Balancing dataset to 50/50 (patient level)...")

sepsis_ids = patient_labels[patient_labels == 1].index.tolist()
nosepsis_ids = patient_labels[patient_labels == 0].index.tolist()

rng = np.random.RandomState(RANDOM_STATE)
nosepsis_sampled = rng.choice(nosepsis_ids, size=len(sepsis_ids), replace=False)

balanced_ids = list(sepsis_ids) + list(nosepsis_sampled)
df = df[df["Patient_ID"].isin(balanced_ids)].copy()

# Restrict patient_labels to exactly the balanced population — this is what
# STEP 6 (the patient-level split) uses, so it must reflect who's actually
# still in df after undersampling, not the pre-balance population.
patient_labels = patient_labels.loc[balanced_ids]

print(f"    Balanced total: {len(balanced_ids):,} "
      f"({len(sepsis_ids):,} sepsis + {len(nosepsis_sampled):,} no-sepsis)")


# ============================================================
# STEP 5 — FORWARD FILL + BACKWARD FILL (per patient)
# Same imputation approach as the XGBoost pipeline.
# ============================================================
print("\n[5] Forward/backward filling missing values per patient...")

df = df.sort_values(["Patient_ID", "Hour"]).reset_index(drop=True)
df = df.set_index("Patient_ID")
df = df.groupby("Patient_ID").ffill()
df = df.groupby("Patient_ID").bfill()
df = df.reset_index()


# ============================================================
# STEP 6 — SPLIT PATIENTS FIRST (before any windowing)
# This is the step that prevents leakage once a single patient can
# contribute MULTIPLE overlapping windows. If we generated windows first
# and split rows/windows afterward, two windows from the same patient could
# land on both sides of the split — the model could then partly recognize
# that patient's own physiology in "held-out" data rather than learning the
# general sepsis pattern. Splitting patient IDs first guarantees every
# window from a given patient lives entirely on one side.
# ============================================================
print("\n[6] Splitting patients into train/test (80/20, stratified)...")

patient_ids = patient_labels.index.values
patient_label_values = patient_labels.values

train_ids, test_ids = train_test_split(
    patient_ids,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=patient_label_values,
)
train_ids = set(train_ids)
test_ids = set(test_ids)

print(f"    Train patients: {len(train_ids):,} | Test patients: {len(test_ids):,}")


# ============================================================
# STEP 7 — WINDOW EXTRACTION
# TRAIN patients: sliding windows (overlapping, stride=STRIDE hours). This
#   is what actually grows the effective training set size — a patient with
#   a long pre-onset/stay region now contributes several training examples
#   instead of one.
# TEST patients: exactly ONE window each (same rule as the original script —
#   last 12h pre-onset for sepsis, last 12h of stay for no-sepsis). The test
#   set intentionally stays one-window-per-patient: if test patients were
#   also slid, a patient with a long stay would contribute many highly
#   correlated windows and silently get more weight in the reported metrics
#   than a patient with a short stay. One real patient = one vote in
#   held-out evaluation.
#
# CAVEAT TO WATCH: sepsis patients' eligible region is bounded by their
# (often earlier) onset time, while no-sepsis patients can draw windows from
# their entire stay. Even with 50/50 patient balance, this can produce a
# WINDOW-level class balance that isn't 50/50 — check the printed sepsis
# percentage below and revisit focal loss alpha if it's shifted a lot.
# ============================================================
print(f"\n[7] Extracting windows (train: sliding stride={STRIDE}h, test: 1/patient)...")


def get_patient_windows(group, label, feature_cols, window_size, stride=None):
    """Return a list of [window_size x n_features] arrays for one patient.

    stride=None -> single window: the last `window_size` rows of the
        eligible region (matches the original one-window-per-patient rule).
    stride=<int> -> sliding windows across the eligible region, starting
        every `stride` hours, each exactly `window_size` hours long.
    """
    group = group.sort_values("Hour").reset_index(drop=True)

    if label == 1:
        # Sepsis patient: only hours strictly BEFORE first sepsis label —
        # every window, sliding or not, must stay pre-onset.
        sepsis_rows = group[group["SepsisLabel"] == 1]
        if len(sepsis_rows) == 0:
            return []
        first_sepsis_idx = sepsis_rows.index[0]
        full_pre_onset = group.loc[:first_sepsis_idx - 1]

        if stride is not None:
            # TRAIN sliding windows: bound to the LOOKBACK_HOURS immediately
            # before onset (see LOOKBACK_HOURS comment above) — don't slide
            # across the patient's whole pre-onset history, some of which
            # may be too far from onset to carry real signal.
            eligible = full_pre_onset.tail(LOOKBACK_HOURS)
        else:
            # TEST single window: unchanged — last WINDOW_SIZE hours
            # before onset, same rule as the original one-window version.
            eligible = full_pre_onset
    else:
        # No-sepsis patient: entire stay is eligible, train or test — every
        # hour of a no-sepsis stay is a validly labeled negative, so there's
        # no equivalent reason to bound it.
        eligible = group

    n = len(eligible)
    if n < window_size:
        return []

    windows = []
    if stride is None:
        windows.append(eligible.tail(window_size)[feature_cols].values)
    else:
        start = 0
        while start + window_size <= n:
            windows.append(eligible.iloc[start:start + window_size][feature_cols].values)
            start += stride

    return windows


X_train_list, y_train_list, train_patient_id_list = [], [], []
X_test_list, y_test_list = [], []
skipped_train, skipped_test = 0, 0

for patient_id, group in df.groupby("Patient_ID"):
    label = patient_labels[patient_id]

    if patient_id in train_ids:
        windows = get_patient_windows(group, label, FEATURE_COLS, WINDOW_SIZE, stride=STRIDE)
        if not windows:
            skipped_train += 1
            continue

        # Cap: if this patient produced more than MAX_WINDOWS_PER_PATIENT,
        # keep an evenly-spaced subset (not just the first N) so the kept
        # windows still span the patient's whole eligible region rather than
        # clustering at the start of their stay.
        if len(windows) > MAX_WINDOWS_PER_PATIENT:
            keep_idx = np.linspace(0, len(windows) - 1, MAX_WINDOWS_PER_PATIENT).astype(int)
            windows = [windows[i] for i in keep_idx]

        for w in windows:
            if w.shape != (WINDOW_SIZE, len(FEATURE_COLS)):
                continue
            X_train_list.append(w)
            y_train_list.append(label)
            train_patient_id_list.append(patient_id)

    elif patient_id in test_ids:
        windows = get_patient_windows(group, label, FEATURE_COLS, WINDOW_SIZE, stride=None)
        if not windows or windows[0].shape != (WINDOW_SIZE, len(FEATURE_COLS)):
            skipped_test += 1
            continue
        X_test_list.append(windows[0])
        y_test_list.append(label)

X_train_raw = np.array(X_train_list)          # [N_train_windows, 12, 21]
y_train = np.array(y_train_list)
train_patient_ids = np.array(train_patient_id_list)  # parallel array — one ID per training window

X_test_raw = np.array(X_test_list)             # [N_test_patients, 12, 21]
y_test = np.array(y_test_list)

n_train_patients_used = len(set(train_patient_id_list))
print(f"    Train windows: {X_train_raw.shape} from {n_train_patients_used:,} patients "
      f"(avg {len(X_train_list)/max(n_train_patients_used,1):.1f} windows/patient)")
print(f"      Sepsis windows: {y_train.sum():,} ({y_train.mean()*100:.1f}%)")
print(f"    Train patients skipped (insufficient eligible hours): {skipped_train:,}")
print(f"    Test windows:  {X_test_raw.shape} (1 per patient)")
print(f"      Sepsis: {y_test.sum():,} ({y_test.mean()*100:.1f}%)")
print(f"    Test patients skipped: {skipped_test:,}")


# ============================================================
# STEP 8 — NORMALIZE (train-window stats only)
# ============================================================
print("\n[8] Normalizing features (train-only statistics)...")

_, _, n_feat = X_train_raw.shape
X_train_flat = X_train_raw.reshape(-1, n_feat)

feat_mean = np.nanmean(X_train_flat, axis=0)
feat_std = np.nanstd(X_train_flat, axis=0)
feat_std[feat_std == 0] = 1  # guard against divide-by-zero

X_train = ((X_train_raw.reshape(-1, n_feat) - feat_mean) / feat_std).reshape(X_train_raw.shape)
X_test = ((X_test_raw.reshape(-1, n_feat) - feat_mean) / feat_std).reshape(X_test_raw.shape)

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

# NEW — one patient ID per training window. Ka_transformer.py needs this to
# carve its train/val split by PATIENT (GroupShuffleSplit), not by row —
# otherwise two windows from the same patient could land on both sides of
# that split too, reintroducing the exact leakage this whole change avoids.
np.save(OUTPUT_DIR / "train_patient_ids.npy", train_patient_ids)

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
print(f"Window size:              {WINDOW_SIZE} hours | Train stride: {STRIDE} hours")
print(f"Features ({len(FEATURE_COLS)}):            {FEATURE_COLS}")
print(f"Train set:                {X_train.shape[0]:,} windows from {n_train_patients_used:,} patients")
print(f"Test set:                 {X_test.shape[0]:,} windows (1/patient)")
print("=" * 60)
print("DONE")