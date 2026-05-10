import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample
import os

# =========================
# CONFIGURATION
# =========================
from pathlib import Path

# =========================
# CONFIGURATION
# =========================
ROOT_DIR = Path(__file__).resolve().parents[2]

CSV_PATH = ROOT_DIR / "data" / "raw" / "physionet_2019.csv"
OUTPUT_DIR = ROOT_DIR / "outputs" / "physionet_analysis"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

IRRELEVANT_COLS = ["Unit1", "Unit2"]
MISSING_THRESHOLD = 0.80

print("=" * 60)
print("PHYSIONET 2019 SEPSIS DATASET ANALYSIS")
print("=" * 60)

# =========================
# 1) LOAD DATA
# =========================
print("\nLoading dataset...")
df = pd.read_csv(CSV_PATH)
if df.columns[0] == "Unnamed: 0":
    df = df.drop(columns=df.columns[0])

print(f"Raw shape: {df.shape}")
print(f"Total rows: {len(df):,}")
print(f"Total columns: {len(df.columns)}")

# =========================
# 2) BASIC STATS
# =========================
print("\n--- BASIC STATS ---")
patient_labels = df.groupby("Patient_ID")["SepsisLabel"].max()
total_patients = len(patient_labels)
class_counts = patient_labels.value_counts()
records_per_patient = df.groupby("Patient_ID").size()

print(f"Total patients:      {total_patients:,}")
print(f"Total records:       {len(df):,}")
print(f"Avg records/patient: {len(df)/total_patients:.1f}")
print(f"No Sepsis: {class_counts[0]:,} ({class_counts[0]/total_patients*100:.1f}%)")
print(f"Sepsis:    {class_counts[1]:,} ({class_counts[1]/total_patients*100:.1f}%)")
print(f"Min/Max/Median hours per patient: {records_per_patient.min()} / {records_per_patient.max()} / {records_per_patient.median():.0f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
bars = axes[0].bar(["No Sepsis", "Sepsis"],
                   [class_counts[0], class_counts[1]],
                   color=["#4472C4", "#ED7D31"], edgecolor="white", width=0.5)
for bar, count in zip(bars, [class_counts[0], class_counts[1]]):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                 f"{count:,}", ha="center", va="bottom", fontsize=12, fontweight="bold")
axes[0].set_title("Patient-Level Class Distribution", fontsize=13, fontweight="bold")
axes[0].set_ylabel("Number of Patients")
axes[0].spines[["top", "right"]].set_visible(False)

axes[1].hist(records_per_patient, bins=50, color="#4472C4", edgecolor="white", alpha=0.8)
axes[1].set_title("Hours of ICU Stay per Patient", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Number of Hours (Records)")
axes[1].set_ylabel("Number of Patients")
axes[1].spines[["top", "right"]].set_visible(False)

plt.suptitle("PhysioNet 2019 — Dataset Overview", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/1_basic_stats.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 1_basic_stats.png")

# =========================
# 3) MISSING VALUES — PER COLUMN
# =========================
print("\n--- MISSING VALUES PER COLUMN ---")
feature_cols = [c for c in df.columns if c not in ["Patient_ID", "Hour", "SepsisLabel"]]
missing_pct = df[feature_cols].isnull().mean().sort_values(ascending=False)

print("Top 10 most missing:")
print(missing_pct.head(10).to_string())

fig, ax = plt.subplots(figsize=(12, 8))
colors = ["#C00000" if v > MISSING_THRESHOLD else "#ED7D31" if v > 0.5 else "#4472C4"
          for v in missing_pct.values]
ax.barh(missing_pct.index, missing_pct.values * 100, color=colors)
ax.axvline(x=MISSING_THRESHOLD * 100, color="red", linestyle="--",
           linewidth=1.5, label=f"Drop threshold ({MISSING_THRESHOLD*100:.0f}%)")
ax.set_xlabel("% Missing Values", fontsize=11)
ax.set_title("% Null Values per Column", fontsize=14, fontweight="bold", pad=15)
ax.legend(fontsize=10)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/2_missing_per_column.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 2_missing_per_column.png")

# =========================
# 4) MISSING VALUES — BY SEPSIS LABEL
# =========================
print("\n--- MISSING VALUES BY SEPSIS LABEL ---")
sepsis_df = df[df["SepsisLabel"] == 1]
nosepsis_df = df[df["SepsisLabel"] == 0]
missing_sepsis = sepsis_df[feature_cols].isnull().mean()
missing_nosepsis = nosepsis_df[feature_cols].isnull().mean()
missing_diff = (missing_sepsis - missing_nosepsis).abs().sort_values(ascending=False)

print("Top 10 columns with biggest missingness difference:")
print(missing_diff.head(10).to_string())

top15 = missing_diff.head(15).index
fig, ax = plt.subplots(figsize=(12, 8))
x = np.arange(len(top15))
width = 0.35
ax.barh(x + width/2, missing_sepsis[top15].values * 100,
        width, label="Sepsis (label=1)", color="#ED7D31", alpha=0.8)
ax.barh(x - width/2, missing_nosepsis[top15].values * 100,
        width, label="No Sepsis (label=0)", color="#4472C4", alpha=0.8)
ax.set_yticks(x)
ax.set_yticklabels(top15)
ax.set_xlabel("% Missing", fontsize=11)
ax.set_title("Missing % by Sepsis Label (Top 15 columns)",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/3_missing_by_label.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 3_missing_by_label.png")

# =========================
# 5) MISSING VALUES — PER PATIENT
# =========================
print("\n--- MISSING VALUES PER PATIENT ---")
patient_missing = df.groupby("Patient_ID")[feature_cols].apply(
    lambda x: x.isnull().mean().mean()
)
print(f"Avg missing per patient: {patient_missing.mean()*100:.1f}%")
print(f"Patients >50% missing:   {(patient_missing > 0.5).sum():,}")
print(f"Patients >80% missing:   {(patient_missing > 0.8).sum():,}")

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(patient_missing * 100, bins=50, color="#4472C4", edgecolor="white", alpha=0.8)
ax.axvline(x=50, color="orange", linestyle="--", linewidth=1.5, label=">50% missing")
ax.axvline(x=80, color="red", linestyle="--", linewidth=1.5, label=">80% missing")
ax.set_xlabel("% Missing Values per Patient", fontsize=11)
ax.set_ylabel("Number of Patients", fontsize=11)
ax.set_title("Distribution of Missing Data per Patient", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/4_missing_per_patient.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 4_missing_per_patient.png")

# =========================
# 6) MISSING VALUES — OVER TIME
# =========================
print("\n--- MISSING VALUES OVER ICU HOURS ---")
time_missing = df.groupby("Hour")[feature_cols].apply(
    lambda x: x.isnull().mean().mean()
) * 100

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(time_missing.index[:72], time_missing.values[:72],
        color="#4472C4", linewidth=2)
ax.fill_between(time_missing.index[:72], time_missing.values[:72],
                alpha=0.2, color="#4472C4")
ax.set_xlabel("ICU Hour", fontsize=11)
ax.set_ylabel("Average % Missing", fontsize=11)
ax.set_title("Average Missing Data % Over ICU Hours (First 72h)",
             fontsize=13, fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/5_missing_over_time.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 5_missing_over_time.png")

# =========================
# 7) MISSING VALUES HEATMAP
# =========================
print("\nGenerating missing values heatmap (200 patient sample)...")
sample_patients = df["Patient_ID"].drop_duplicates().sample(200, random_state=42)
df_sample = df[df["Patient_ID"].isin(sample_patients)].groupby("Patient_ID")[feature_cols].mean()
missing_heatmap = df_sample.isnull().astype(int)

fig, ax = plt.subplots(figsize=(16, 8))
sns.heatmap(missing_heatmap.T, cmap=["#4472C4", "#C00000"],
            xticklabels=False, yticklabels=True,
            cbar_kws={"label": "Missing (1=Red) / Present (0=Blue)"},
            ax=ax)
ax.set_xlabel("Patients (sample of 200)", fontsize=11)
ax.set_ylabel("Variables", fontsize=11)
ax.set_title("Missing Values Heatmap\n(Red = Missing, Blue = Present)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/6_missing_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 6_missing_heatmap.png")

# =========================
# 8) PRUNE COLUMNS
# =========================
print("\n--- PRUNING COLUMNS ---")
cols_to_drop = [c for c in IRRELEVANT_COLS if c in df.columns]
high_missing = missing_pct[missing_pct > MISSING_THRESHOLD].index.tolist()
all_drop = cols_to_drop + high_missing
df_pruned = df.drop(columns=[c for c in all_drop if c in df.columns])
remaining_features = [c for c in df_pruned.columns
                      if c not in ["Patient_ID", "Hour", "SepsisLabel"]]
print(f"Dropped {len(all_drop)} columns")
print(f"Remaining: {remaining_features}")
print(f"Shape: {df_pruned.shape}")

# =========================
# 9) CLASS IMBALANCE — RECORD LEVEL
# =========================
print("\n--- RECORD-LEVEL CLASS IMBALANCE ---")
record_counts = df["SepsisLabel"].value_counts()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].bar(["No Sepsis", "Sepsis"], [class_counts[0], class_counts[1]],
            color=["#4472C4", "#ED7D31"], edgecolor="white")
axes[0].set_title("Patient-Level Imbalance", fontsize=12, fontweight="bold")
axes[0].set_ylabel("Number of Patients")
for i, v in enumerate([class_counts[0], class_counts[1]]):
    axes[0].text(i, v + 200, f"{v:,}\n({v/total_patients*100:.1f}%)",
                 ha="center", fontsize=10, fontweight="bold")

axes[1].bar(["No Sepsis", "Sepsis"], [record_counts[0], record_counts[1]],
            color=["#4472C4", "#ED7D31"], edgecolor="white")
axes[1].set_title("Record-Level Imbalance", fontsize=12, fontweight="bold")
axes[1].set_ylabel("Number of Records")
for i, v in enumerate([record_counts[0], record_counts[1]]):
    axes[1].text(i, v + 5000, f"{v:,}\n({v/len(df)*100:.1f}%)",
                 ha="center", fontsize=10, fontweight="bold")

for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)

plt.suptitle("Class Imbalance — Patient vs Record Level", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/7_class_imbalance.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 7_class_imbalance.png")

# =========================
# 10) CORRELATION WITH SEPSIS LABEL
# =========================
print("\n--- CORRELATION WITH SEPSIS LABEL ---")
numeric_features = [c for c in remaining_features
                    if df_pruned[c].dtype in [np.float64, np.int64]]
correlations = df_pruned[numeric_features + ["SepsisLabel"]].corr()["SepsisLabel"].drop("SepsisLabel")
correlations = correlations.abs().sort_values(ascending=True)

print("Correlations:")
print(correlations.sort_values(ascending=False).to_string())

fig, ax = plt.subplots(figsize=(10, 6))
colors = ["#C00000" if v > 0.05 else "#4472C4" for v in correlations.values]
bars = ax.barh(correlations.index, correlations.values, color=colors)
for bar, val in zip(bars, correlations.values):
    ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
            f"{val:.3f}", va="center", fontsize=9)
ax.set_xlabel("Absolute Pearson Correlation with SepsisLabel", fontsize=11)
ax.set_title("Feature Correlation with Sepsis Label", fontsize=13, fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/8_correlation.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 8_correlation.png")

# Correlation heatmap
fig, ax = plt.subplots(figsize=(12, 10))
corr_matrix = df_pruned[numeric_features + ["SepsisLabel"]].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f",
            cmap="RdBu_r", center=0, square=True,
            linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8})
ax.set_title("Correlation Heatmap — Features vs SepsisLabel",
             fontsize=13, fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/9_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 9_correlation_heatmap.png")

# =========================
# 11) BALANCE DATASET
# =========================
print("\n--- BALANCING DATASET (50/50) ---")
patient_labels_pruned = df_pruned.groupby("Patient_ID")["SepsisLabel"].max()
sepsis_ids = patient_labels_pruned[patient_labels_pruned == 1].index.tolist()
no_sepsis_ids = patient_labels_pruned[patient_labels_pruned == 0].index.tolist()

no_sepsis_sampled = resample(no_sepsis_ids, n_samples=len(sepsis_ids),
                              random_state=42, replace=False)
balanced_ids = list(sepsis_ids) + list(no_sepsis_sampled)
df_balanced = df_pruned[df_pruned["Patient_ID"].isin(balanced_ids)].copy()
bal_counts = df_balanced.groupby("Patient_ID")["SepsisLabel"].max().value_counts()

print(f"Balanced: {bal_counts[0]:,} no-sepsis + {bal_counts[1]:,} sepsis patients")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].bar(["No Sepsis", "Sepsis"], [class_counts[0], class_counts[1]],
            color=["#4472C4", "#ED7D31"], edgecolor="white")
axes[0].set_title("Original (92.7% / 7.3%)", fontsize=12, fontweight="bold")
axes[0].set_ylabel("Number of Patients")

axes[1].bar(["No Sepsis", "Sepsis"], [bal_counts[0], bal_counts[1]],
            color=["#4472C4", "#ED7D31"], edgecolor="white")
axes[1].set_title("Balanced (50/50)", fontsize=12, fontweight="bold")
axes[1].set_ylabel("Number of Patients")

for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)

plt.suptitle("Class Distribution: Before vs After Balancing", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/10_before_after_balance.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 10_before_after_balance.png")

# =========================
# 12) SAVE CLEANED DATASETS
# =========================
print("\n--- SAVING ---")
pruned_path = PROCESSED_DIR / "physionet_pruned.csv"
balanced_path = PROCESSED_DIR / "physionet_balanced.csv"

df_pruned.to_csv(pruned_path, index=False)
df_balanced.to_csv(balanced_path, index=False)
print(f"Pruned CSV:   {pruned_path}")
print(f"Balanced CSV: {balanced_path}")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Total patients:      {total_patients:,}")
print(f"Total records:       {len(df):,}")
print(f"Class imbalance:     {class_counts[0]:,} no-sepsis / {class_counts[1]:,} sepsis")
print(f"Columns dropped:     {len(all_drop)}")
print(f"Remaining features:  {remaining_features}")
print(f"Balanced patients:   {len(balanced_ids):,} (50/50)")
print(f"Charts saved to:     {OUTPUT_DIR}")
print("\nDONE!")
