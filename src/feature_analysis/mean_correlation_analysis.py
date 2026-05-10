import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

# =========================
# CONFIGURATION
# =========================
ROOT_DIR = Path(__file__).resolve().parents[2]

PHYSIONET_PATH = ROOT_DIR / "data" / "processed" / "physionet_balanced.csv"
PHEMS_PATH = ROOT_DIR / "data" / "processed" / "phems_balanced.csv"
OUTPUT_DIR = ROOT_DIR / "outputs" / "feature_analysis"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Feature columns are derived dynamically from the actual files
# They are populated after loading the datasets below
PHYSIONET_NON_FEATURE = ["Patient_ID", "Hour", "SepsisLabel", "HospAdmTime", "ICULOS"]
PHEMS_NON_FEATURE     = ["visit_occurrence_id", "person_id", "measurement_datetime", "SepsisLabel"]

# Friendly display names for PHEMS columns
PHEMS_RENAME = {
    "Body temperature":                                "Temp",
    "Respiratory rate":                                "Resp",
    "Heart rate":                                      "HR",
    "Measurement of oxygen saturation at periphery":   "O2Sat",
    "age_in_months":                                   "Age",
    "gender":                                          "Gender",
}

PHYSIONET_PATIENT_COL = "Patient_ID"
PHEMS_PATIENT_COL     = "person_id"
LABEL_COL             = "SepsisLabel"

print("=" * 60)
print("TASK 2 — STEP 1: CORRELATION ANALYSIS")
print("=" * 60)

# =========================
# FUNCTION: RUN CORRELATION ANALYSIS
# =========================
def run_correlation_analysis(df, feature_cols, patient_col, label_col, dataset_name):
    print(f"\n{'='*50}")
    print(f"DATASET: {dataset_name}")
    print(f"{'='*50}")

    # Drop rows with null labels
    df = df.dropna(subset=[label_col]).copy()
    df[label_col] = df[label_col].astype(int)
    print(f"Rows after dropping null labels: {len(df):,}")

    # Step 1: Aggregate per patient (mean of each variable)
    print("\nAggregating per patient (taking mean of each variable)...")
    agg_dict = {col: "mean" for col in feature_cols}
    agg_dict[label_col] = "max"  # patient label = 1 if ever had sepsis
    df_patient = df.groupby(patient_col).agg(agg_dict).reset_index()
    print(f"Patients: {len(df_patient):,}")
    print(f"Sepsis:    {(df_patient[label_col]==1).sum():,}")
    print(f"No-sepsis: {(df_patient[label_col]==0).sum():,}")

    # Step 2: Split into sepsis vs no-sepsis groups
    sepsis    = df_patient[df_patient[label_col] == 1]
    no_sepsis = df_patient[df_patient[label_col] == 0]

    # Step 3: For each variable compute stats
    print("\n--- CORRELATION RESULTS ---")
    results = []
    for col in feature_cols:
        s_vals  = sepsis[col].dropna()
        ns_vals = no_sepsis[col].dropna()

        if len(s_vals) < 5 or len(ns_vals) < 5:
            print(f"  {col}: not enough data, skipping")
            continue

        s_mean  = s_vals.mean()
        ns_mean = ns_vals.mean()
        s_std   = s_vals.std()
        ns_std  = ns_vals.std()
        diff    = s_mean - ns_mean
        pct_diff = (diff / ns_mean * 100) if ns_mean != 0 else 0

        # Pearson correlation with label
        merged = df_patient[[col, label_col]].dropna()
        corr, _ = stats.pearsonr(merged[col], merged[label_col])

        # t-test
        t_stat, p_val = stats.ttest_ind(s_vals, ns_vals, equal_var=False)

        results.append({
            "Variable":         col,
            "Mean (Sepsis)":    round(s_mean, 3),
            "Mean (No Sepsis)": round(ns_mean, 3),
            "Difference":       round(diff, 3),
            "% Difference":     round(pct_diff, 1),
            "Std (Sepsis)":     round(s_std, 3),
            "Std (No Sepsis)":  round(ns_std, 3),
            "Pearson r":        round(abs(corr), 4),
            "p-value":          round(p_val, 6),
            "Significant":      "YES" if p_val < 0.05 else "NO",
        })

        sig = "✓" if p_val < 0.05 else "✗"
        print(f"  {sig} {col:<10} | sepsis={s_mean:.2f}  no-sepsis={ns_mean:.2f}  "
              f"diff={diff:+.2f} ({pct_diff:+.1f}%)  r={abs(corr):.3f}  p={p_val:.4f}")

    df_results = pd.DataFrame(results).sort_values("Pearson r", ascending=False).reset_index(drop=True)

    # Save CSV
    csv_path = OUTPUT_DIR / f"{dataset_name}_correlation.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path.name}")

    # =========================
    # CHART 1: MEAN COMPARISON BAR CHART
    # =========================
    n_vars = len(df_results)
    fig, axes = plt.subplots(1, n_vars, figsize=(n_vars * 2.5, 5))
    if n_vars == 1:
        axes = [axes]

    for ax, (_, row) in zip(axes, df_results.iterrows()):
        bars = ax.bar(["Sepsis", "No Sepsis"],
                      [row["Mean (Sepsis)"], row["Mean (No Sepsis)"]],
                      color=["#C00000", "#2E75B6"], edgecolor="white", width=0.5)
        for bar, val in zip(bars, [row["Mean (Sepsis)"], row["Mean (No Sepsis)"]]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.01,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        sig_marker = " *" if row["Significant"] == "YES" else ""
        ax.set_title(f"{row['Variable']}{sig_marker}\nr={row['Pearson r']:.3f}",
                     fontsize=9, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", labelsize=8)

    plt.suptitle(f"{dataset_name} — Mean Values: Sepsis vs No-Sepsis\n(* = statistically significant p<0.05)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    chart_path = OUTPUT_DIR / f"{dataset_name}_1_mean_comparison.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {dataset_name}_1_mean_comparison.png")

    # =========================
    # CHART 2: RANKED CORRELATION BAR CHART
    # =========================
    fig, ax = plt.subplots(figsize=(8, max(4, n_vars * 0.6)))
    colors = ["#C00000" if sig == "YES" else "#AAAAAA"
              for sig in df_results["Significant"]]
    bars = ax.barh(df_results["Variable"], df_results["Pearson r"],
                   color=colors, edgecolor="white", height=0.5)
    for bar, val in zip(bars, df_results["Pearson r"]):
        ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=10)
    ax.set_xlabel("Absolute Pearson Correlation with Sepsis Label", fontsize=11)
    ax.set_title(f"{dataset_name} — Variable Correlation Ranking\n(Red = significant p<0.05, Grey = not significant)",
                 fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    chart_path = OUTPUT_DIR / f"{dataset_name}_2_correlation_ranking.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {dataset_name}_2_correlation_ranking.png")

    # =========================
    # CHART 3: BOX PLOTS
    # =========================
    fig, axes = plt.subplots(1, n_vars, figsize=(n_vars * 2.8, 5))
    if n_vars == 1:
        axes = [axes]

    for ax, col in zip(axes, df_results["Variable"]):
        s_vals  = sepsis[col].dropna()
        ns_vals = no_sepsis[col].dropna()
        bp = ax.boxplot([s_vals, ns_vals],
                        patch_artist=True,
                        tick_labels=["Sepsis", "No\nSepsis"],
                        widths=0.5)
        bp["boxes"][0].set_facecolor("#C00000")
        bp["boxes"][0].set_alpha(0.7)
        bp["boxes"][1].set_facecolor("#2E75B6")
        bp["boxes"][1].set_alpha(0.7)
        for median in bp["medians"]:
            median.set_color("white")
            median.set_linewidth(2)
        ax.set_title(col, fontsize=9, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", labelsize=8)

    plt.suptitle(f"{dataset_name} — Box Plots: Sepsis vs No-Sepsis Distribution",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    chart_path = OUTPUT_DIR / f"{dataset_name}_3_boxplots.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {dataset_name}_3_boxplots.png")

    print(f"\nTOP VARIABLES BY CORRELATION ({dataset_name}):")
    for i, (_, row) in enumerate(df_results.head(4).iterrows()):
        print(f"  {i+1}. {row['Variable']} — r={row['Pearson r']:.3f}, "
              f"sepsis mean={row['Mean (Sepsis)']:.2f} vs no-sepsis mean={row['Mean (No Sepsis)']:.2f} "
              f"({row['% Difference']:+.1f}%)")

    return df_results, df_patient

# =========================
# RUN ON PHYSIONET
# =========================
print("\nLoading PhysioNet balanced dataset...")
df_physio = pd.read_csv(PHYSIONET_PATH)
if df_physio.columns[0] == "Unnamed: 0":
    df_physio = df_physio.drop(columns=df_physio.columns[0])
PHYSIONET_FEATURES = [c for c in df_physio.columns if c not in PHYSIONET_NON_FEATURE]
print(f"PhysioNet features detected ({len(PHYSIONET_FEATURES)}): {PHYSIONET_FEATURES}")

physio_results, physio_patients = run_correlation_analysis(
    df_physio, PHYSIONET_FEATURES, PHYSIONET_PATIENT_COL, LABEL_COL, "PhysioNet"
)

# =========================
# RUN ON PHEMS
# =========================
# Encode gender string to numeric (MALE=1, FEMALE=0) for correlation analysis
print("\nLoading PHEMS balanced dataset...")
df_phems = pd.read_csv(PHEMS_PATH)
# Step 1: Encode gender string to numeric FIRST
if "gender" in df_phems.columns:
    df_phems["gender"] = df_phems["gender"].map({"MALE": 1, "FEMALE": 0})
    df_phems["gender"] = pd.to_numeric(df_phems["gender"], errors="coerce")
# Step 2: Rename columns to short names
df_phems = df_phems.rename(columns=PHEMS_RENAME)

# Step 3: Remove physiologically impossible outlier values
# PHEMS has a small number of corrupt entries with values like 156,775 bpm
# These are clearly data errors — we set them to NaN so they are ignored in aggregation
print("Removing physiologically impossible outlier values from PHEMS...")
PHEMS_VALID_RANGES = {
    "HR":     (20,  300),   # heart rate bpm
    "Temp":   (25,  45),    # body temperature celsius
    "Resp":   (4,   80),    # respiratory rate breaths/min
    "O2Sat":  (50,  100),   # oxygen saturation %
    "Age":    (0,   300),   # age in months (0-25 years)
}
for col, (min_val, max_val) in PHEMS_VALID_RANGES.items():
    if col in df_phems.columns:
        # Count how many values we are removing
        invalid = ((df_phems[col] < min_val) | (df_phems[col] > max_val))
        n_invalid = invalid.sum()
        if n_invalid > 0:
            print(f"  {col}: removing {n_invalid} outlier rows (outside {min_val}-{max_val})")
            df_phems.loc[invalid, col] = None  # set to NaN, not drop the row

# Step 4: Now detect features from renamed columns
PHEMS_FEATURES = [c for c in df_phems.columns if c not in PHEMS_NON_FEATURE]
print(f"PHEMS features detected ({len(PHEMS_FEATURES)}): {PHEMS_FEATURES}")

phems_results, phems_patients = run_correlation_analysis(
    df_phems, PHEMS_FEATURES, PHEMS_PATIENT_COL, LABEL_COL, "PHEMS"
)

# =========================
# CROSS DATASET COMPARISON
# =========================
print("\n--- CROSS DATASET COMPARISON ---")
shared = set(physio_results["Variable"]) & set(phems_results["Variable"])
print(f"Shared variables: {shared}")

if shared:
    print("\nCorrelation comparison for shared variables:")
    print(f"{'Variable':<12} {'PhysioNet r':>12} {'PHEMS r':>10} {'Agreement':>12}")
    print("-" * 50)
    for var in sorted(shared):
        p_r = physio_results[physio_results["Variable"] == var]["Pearson r"].values[0]
        h_r = phems_results[phems_results["Variable"] == var]["Pearson r"].values[0]
        agree = "✓ AGREE" if abs(p_r - h_r) < 0.1 else "~ DIFFER"
        print(f"  {var:<12} {p_r:>12.3f} {h_r:>10.3f} {agree:>12}")

print(f"\nOutputs saved to: {OUTPUT_DIR}")
print("\nDONE! Next step: Step 2 — Random Forest feature importance")
