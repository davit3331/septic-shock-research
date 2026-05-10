import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path


# =========================
# CONFIGURATION
# =========================
ROOT_DIR = Path(__file__).resolve().parents[2]

PHEMS_FLAT_PATH = ROOT_DIR / "data" / "processed" / "phems_flat.csv"
PHEMS_CLEAN_PATH = ROOT_DIR / "data" / "processed" / "phems_clean.csv"
PHYSIONET_RAW_PATH = ROOT_DIR / "data" / "raw" / "physionet_2019.csv"
PHYSIONET_CLEAN_PATH = ROOT_DIR / "data" / "processed" / "physionet_balanced.csv"

OUTPUT_DIR = ROOT_DIR / "outputs" / "variable_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# CONFIGURATION
# =========================
ROOT_DIR = Path(__file__).resolve().parents[2]

PHEMS_FLAT_PATH = ROOT_DIR / "data" / "processed" / "phems_flat.csv"
PHEMS_CLEAN_PATH = ROOT_DIR / "data" / "processed" / "phems_clean.csv"
PHYSIONET_RAW_PATH = ROOT_DIR / "data" / "raw" / "physionet_2019.csv"
PHYSIONET_CLEAN_PATH = ROOT_DIR / "data" / "processed" / "physionet_balanced.csv"

OUTPUT_DIR = ROOT_DIR / "outputs" / "variable_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("TASK 1: CROSS-DATASET VARIABLE COMPARISON")
print("PhysioNet 2019  vs  PHEMS")
print("=" * 60)

# =========================
# 1) PHYSIONET VARIABLES — loaded dynamically from both CSVs
# =========================
PHYSIONET_NON_CLINICAL = ["Patient_ID", "Hour", "SepsisLabel", "HospAdmTime", "ICULOS", "Unit1", "Unit2"]

print("\nLoading PhysioNet raw dataset...")
df_physio_raw = pd.read_csv(PHYSIONET_RAW_PATH)
if df_physio_raw.columns[0] == "Unnamed: 0":
    df_physio_raw = df_physio_raw.drop(columns=df_physio_raw.columns[0])
physionet_vars = [c for c in df_physio_raw.columns if c not in PHYSIONET_NON_CLINICAL]
print(f"PhysioNet original clinical variables ({len(physionet_vars)})")

print("Loading PhysioNet clean dataset...")
df_physio_clean = pd.read_csv(PHYSIONET_CLEAN_PATH)
if df_physio_clean.columns[0] == "Unnamed: 0":
    df_physio_clean = df_physio_clean.drop(columns=df_physio_clean.columns[0])
PHYSIONET_CLEAN_VARS = [c for c in df_physio_clean.columns if c not in PHYSIONET_NON_CLINICAL]
print(f"PhysioNet clean variables after 80% threshold ({len(PHYSIONET_CLEAN_VARS)})")

# =========================
# 2) PHEMS VARIABLES — loaded dynamically from both CSVs
# =========================
PHEMS_NON_CLINICAL = [
    "visit_occurrence_id", "person_id", "measurement_datetime",
    "visit_start_date", "birth_datetime", "SepsisLabel"
]

print("\nLoading real PHEMS flat file...")
df_phems = pd.read_csv(PHEMS_FLAT_PATH)
phems_vars = [c for c in df_phems.columns if c not in PHEMS_NON_CLINICAL]
print(f"PHEMS original clinical variables ({len(phems_vars)})")

print("Loading PHEMS clean file...")
df_phems_clean = pd.read_csv(PHEMS_CLEAN_PATH)
PHEMS_CLEAN_VARS = [c for c in df_phems_clean.columns if c not in PHEMS_NON_CLINICAL]
print(f"PHEMS clean variables after 80% threshold ({len(PHEMS_CLEAN_VARS)})")

# =========================
# 3) CANONICAL MAPPING
# =========================
PHYSIONET_TO_CANONICAL = {
    "HR":               "Heart Rate",
    "O2Sat":            "SpO2 / Oxygen Saturation",
    "Temp":             "Body Temperature",
    "SBP":              "Systolic Blood Pressure",
    "MAP":              "Mean Arterial Pressure",
    "DBP":              "Diastolic Blood Pressure",
    "Resp":             "Respiratory Rate",
    "Age":              "Age",
    "Gender":           "Gender",
    "EtCO2":            "EtCO2",
    "BaseExcess":       "Base Excess",
    "HCO3":             "Bicarbonate (HCO3)",
    "FiO2":             "FiO2 (Inhaled O2 Fraction)",
    "pH":               "pH (Arterial)",
    "PaCO2":            "PaCO2 (Arterial)",
    "SaO2":             "SaO2",
    "AST":              "AST",
    "BUN":              "BUN",
    "Alkalinephos":     "Alkaline Phosphatase",
    "Calcium":          "Calcium",
    "Chloride":         "Chloride",
    "Creatinine":       "Creatinine",
    "Bilirubin_direct": "Bilirubin (Direct)",
    "Glucose":          "Glucose",
    "Lactate":          "Lactate",
    "Magnesium":        "Magnesium",
    "Phosphate":        "Phosphate",
    "Potassium":        "Potassium",
    "Bilirubin_total":  "Bilirubin (Total)",
    "TroponinI":        "Troponin I",
    "Hct":              "Hematocrit",
    "Hgb":              "Hemoglobin",
    "PTT":              "PTT",
    "WBC":              "White Blood Cell Count",
    "Fibrinogen":       "Fibrinogen",
    "Platelets":        "Platelets",
}

PHEMS_TO_CANONICAL = {
    "Heart rate":                                          "Heart Rate",
    "Measurement of oxygen saturation at periphery":      "SpO2 / Oxygen Saturation",
    "Body temperature":                                    "Body Temperature",
    "Systolic blood pressure":                             "Systolic Blood Pressure",
    "Diastolic blood pressure":                            "Diastolic Blood Pressure",
    "Respiratory rate":                                    "Respiratory Rate",
    "Pulse":                                               "Pulse / Heart Rate (alt)",
    "Arterial pulse pressure":                             "Arterial Pulse Pressure",
    "age_in_months":                                       "Age",
    "gender":                                              "Gender",
    "Lactate [Moles/volume] in Blood":                     "Lactate",
    "Creatinine [Mass/volume] in Blood":                   "Creatinine",
    "Bilirubin.total [Moles/volume] in Serum or Plasma":   "Bilirubin (Total)",
    "Bilirubin measurement":                               "Bilirubin (General)",
    "Total white blood count":                             "White Blood Cell Count",
    "White blood cell count":                              "White Blood Cell Count",
    "Neutrophil Ab [Units/volume] in Serum":               "Neutrophils",
    "Platelet count":                                      "Platelets",
    "Procalcitonin [Mass/volume] in Serum or Plasma":      "Procalcitonin",
    "C reactive protein [Mass/volume] in Serum or Plasma": "CRP (C-Reactive Protein)",
    "Glasgow coma scale":                                  "Glasgow Coma Scale",
    "Glucose [Moles/volume] in Serum or Plasma":           "Glucose",
    "Sodium [Moles/volume] in Serum or Plasma":            "Sodium",
    "Potassium [Moles/volume] in Blood":                   "Potassium",
    "Chloride [Moles/volume] in Blood":                    "Chloride",
    "Calcium [Moles/volume] in Serum or Plasma":           "Calcium",
    "Ionised calcium measurement":                         "Ionised Calcium",
    "Magnesium [Moles/volume] in Blood":                   "Magnesium",
    "Phosphate [Moles/volume] in Serum or Plasma":         "Phosphate",
    "Albumin [Mass/volume] in Serum or Plasma":            "Albumin",
    "Hemoglobin [Moles/volume] in Blood":                  "Hemoglobin",
    "Hematocrit [Volume Fraction] of Blood":               "Hematocrit",
    "Bicarbonate [Moles/volume] in Arterial blood":        "Bicarbonate (Arterial)",
    "Bicarbonate [Moles/volume] in Venous blood":          "Bicarbonate (Venous)",
    "Blood arterial pH":                                   "pH (Arterial)",
    "Blood venous pH":                                     "pH (Venous)",
    "Oxygen [Partial pressure] in Arterial blood":         "PaO2 (Arterial)",
    "Oxygen [Partial pressure] in Venous blood":           "PvO2 (Venous)",
    "Carbon dioxide [Partial pressure] in Arterial blood": "PaCO2 (Arterial)",
    "Carbon dioxide [Partial pressure] in Venous blood":   "PvCO2 (Venous)",
    "Base excess in Arterial blood by calculation":        "Base Excess (Arterial)",
    "Base excess in Venous blood by calculation":          "Base Excess (Venous)",
    "Oxygen/Gas total [Pure volume fraction] Inhaled gas": "FiO2 (Inhaled O2 Fraction)",
    "Prothrombin time (PT)":                               "Prothrombin Time (PT)",
    "Partial thromboplastin time":                         "PTT",
    "activated":                                           "aPTT (Activated PTT)",
    "Fibrinogen measurement":                              "Fibrinogen",
    "D-dimer level":                                       "D-Dimer",
    "Interleukin 6 [Mass/volume] in Body fluid":           "Interleukin-6 (IL-6)",
    "Left pupil Diameter Auto":                            "Left Pupil Diameter",
    "Right pupil Diameter Auto":                           "Right Pupil Diameter",
    "Left pupil Pupillary response":                       "Left Pupillary Response",
    "Right pupil Pupillary response":                      "Right Pupillary Response",
    "Capillary refill [Time]":                             "Capillary Refill Time",
}

# Canonical sets for clean variables
PHYSIONET_CLEAN_CANONICAL = {PHYSIONET_TO_CANONICAL.get(v, v) for v in PHYSIONET_CLEAN_VARS}
PHEMS_CLEAN_CANONICAL     = {PHEMS_TO_CANONICAL.get(v, v)     for v in PHEMS_CLEAN_VARS}

# =========================
# 4) BUILD RAW COMPARISON TABLE
# =========================
print("\n--- COMPARISON 1: RAW DATASETS ---")
physio_canonical = {PHYSIONET_TO_CANONICAL.get(v, v): v for v in physionet_vars}
phems_canonical  = {PHEMS_TO_CANONICAL.get(v, v): v    for v in phems_vars}
all_canonical    = sorted(set(list(physio_canonical.keys()) + list(phems_canonical.keys())))

rows = []
for canonical in all_canonical:
    in_physio = canonical in physio_canonical
    in_phems  = canonical in phems_canonical
    if in_physio and in_phems:
        status = "BOTH"
    elif in_physio:
        status = "PhysioNet only"
    else:
        status = "PHEMS only"
    rows.append({
        "Variable (Canonical Name)": canonical,
        "PhysioNet 2019":            "✓" if in_physio else "—",
        "PHEMS":                     "✓" if in_phems  else "—",
        "Status":                    status,
        "PhysioNet Column":          physio_canonical.get(canonical, "—"),
        "PHEMS Column":              phems_canonical.get(canonical, "—"),
    })

df_raw   = pd.DataFrame(rows)
order    = {"BOTH": 0, "PhysioNet only": 1, "PHEMS only": 2}
df_raw["_sort"] = df_raw["Status"].map(order)
df_raw   = df_raw.sort_values("_sort").drop(columns="_sort").reset_index(drop=True)

both_raw        = df_raw[df_raw["Status"] == "BOTH"]
physio_only_raw = df_raw[df_raw["Status"] == "PhysioNet only"]
phems_only_raw  = df_raw[df_raw["Status"] == "PHEMS only"]

print(f"Variables in BOTH (raw):         {len(both_raw)}")
print(f"PhysioNet only (raw):            {len(physio_only_raw)}")
print(f"PHEMS only (raw):                {len(phems_only_raw)}")
print(f"Total unique variables (raw):    {len(df_raw)}")
print("\nShared variables (raw):")
print(both_raw[["Variable (Canonical Name)", "PhysioNet Column", "PHEMS Column"]].to_string(index=False))

df_raw.to_csv(f"{OUTPUT_DIR}/raw_variable_comparison.csv", index=False)
print("Saved: raw_variable_comparison.csv")

# =========================
# 5) BUILD CLEAN COMPARISON TABLE
# =========================
print("\n--- COMPARISON 2: CLEAN DATASETS (after 80% missing threshold) ---")

all_clean_canonical = sorted(PHYSIONET_CLEAN_CANONICAL | PHEMS_CLEAN_CANONICAL)
clean_rows = []
for canonical in all_clean_canonical:
    in_physio = canonical in PHYSIONET_CLEAN_CANONICAL
    in_phems  = canonical in PHEMS_CLEAN_CANONICAL
    if in_physio and in_phems:
        status = "BOTH"
    elif in_physio:
        status = "PhysioNet only"
    else:
        status = "PHEMS only"
    clean_rows.append({
        "Variable (Canonical Name)": canonical,
        "PhysioNet 2019 (clean)":    "✓" if in_physio else "—",
        "PHEMS (clean)":             "✓" if in_phems  else "—",
        "Status":                    status,
    })

df_clean = pd.DataFrame(clean_rows)
df_clean["_sort"] = df_clean["Status"].map(order)
df_clean = df_clean.sort_values("_sort").drop(columns="_sort").reset_index(drop=True)

both_clean        = df_clean[df_clean["Status"] == "BOTH"]
physio_only_clean = df_clean[df_clean["Status"] == "PhysioNet only"]
phems_only_clean  = df_clean[df_clean["Status"] == "PHEMS only"]

print(f"Variables in BOTH (clean):       {len(both_clean)}")
print(f"PhysioNet only (clean):          {len(physio_only_clean)}")
print(f"PHEMS only (clean):              {len(phems_only_clean)}")
print("\nShared usable variables (clean):")
print(both_clean[["Variable (Canonical Name)"]].to_string(index=False))
if len(physio_only_clean) > 0:
    print("\nPhysioNet only after cleaning:")
    print(physio_only_clean[["Variable (Canonical Name)"]].to_string(index=False))
if len(phems_only_clean) > 0:
    print("\nPHEMS only after cleaning:")
    print(phems_only_clean[["Variable (Canonical Name)"]].to_string(index=False))

df_clean.to_csv(f"{OUTPUT_DIR}/clean_variable_comparison.csv", index=False)
print("Saved: clean_variable_comparison.csv")

# =========================
# 6) CHART — RAW COMPARISON MATRIX
# =========================
print("\nGenerating charts...")
color_map = {"BOTH": "#2E75B6", "PhysioNet only": "#ED7D31", "PHEMS only": "#70AD47"}
legend_elements = [
    mpatches.Patch(color="#2E75B6", label="PhysioNet 2019"),
    mpatches.Patch(color="#70AD47", label="PHEMS"),
    mpatches.Patch(color="#E0E0E0", label="Absent"),
]

fig, ax = plt.subplots(figsize=(10, max(12, len(df_raw) * 0.32)))
display_vars = df_raw["Variable (Canonical Name)"].tolist()
n = len(display_vars)
for i, (_, row) in enumerate(df_raw.iterrows()):
    y = n - i - 1
    physio_val = 1 if row["PhysioNet 2019"] == "✓" else 0
    phems_val  = 1 if row["PHEMS"] == "✓" else 0
    ax.barh(y - 0.2, physio_val, height=0.35,
            color="#2E75B6" if physio_val else "#E0E0E0", alpha=0.85)
    ax.barh(y + 0.2, phems_val,  height=0.35,
            color="#70AD47" if phems_val  else "#E0E0E0", alpha=0.85)
    ax.text(1.05, y, row["Status"], va="center", fontsize=8,
            color=color_map[row["Status"]], fontweight="bold")
ax.set_yticks(range(n))
ax.set_yticklabels(reversed(display_vars), fontsize=8)
ax.set_xticks([0, 1])
ax.set_xticklabels(["Absent", "Present"])
ax.set_title("Comparison 1: Raw Variable Presence\nPhysioNet 2019 vs PHEMS (before cleaning)",
             fontsize=13, fontweight="bold", pad=15)
ax.legend(handles=legend_elements, loc="lower right", fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/1_raw_comparison_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 1_raw_comparison_matrix.png")

# =========================
# 7) CHART — CLEAN COMPARISON MATRIX
# =========================
fig, ax = plt.subplots(figsize=(10, max(6, len(df_clean) * 0.5)))
display_clean = df_clean["Variable (Canonical Name)"].tolist()
nc = len(display_clean)
for i, (_, row) in enumerate(df_clean.iterrows()):
    y = nc - i - 1
    physio_val = 1 if row["PhysioNet 2019 (clean)"] == "✓" else 0
    phems_val  = 1 if row["PHEMS (clean)"] == "✓" else 0
    ax.barh(y - 0.2, physio_val, height=0.35,
            color="#2E75B6" if physio_val else "#E0E0E0", alpha=0.85)
    ax.barh(y + 0.2, phems_val,  height=0.35,
            color="#70AD47" if phems_val  else "#E0E0E0", alpha=0.85)
    ax.text(1.05, y, row["Status"], va="center", fontsize=9,
            color=color_map[row["Status"]], fontweight="bold")
ax.set_yticks(range(nc))
ax.set_yticklabels(reversed(display_clean), fontsize=10)
ax.set_xticks([0, 1])
ax.set_xticklabels(["Absent", "Present"])
ax.set_title("Comparison 2: Usable Variables After 80% Missing Threshold\nPhysioNet 2019 vs PHEMS (after cleaning)",
             fontsize=13, fontweight="bold", pad=15)
ax.legend(handles=legend_elements, loc="lower right", fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/2_clean_comparison_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 2_clean_comparison_matrix.png")

# =========================
# 8) CHART — VARIABLE COUNT BEFORE vs AFTER CLEANING
# =========================
fig, ax = plt.subplots(figsize=(8, 5))
categories = ["PhysioNet\n(Raw)", "PhysioNet\n(Clean)", "PHEMS\n(Raw)", "PHEMS\n(Clean)"]
values     = [len(physionet_vars), len(PHYSIONET_CLEAN_VARS), len(phems_vars), len(PHEMS_CLEAN_VARS)]
colors     = ["#2E75B6", "#9DC3E6", "#70AD47", "#A9D18E"]
bars = ax.bar(categories, values, color=colors, edgecolor="white", width=0.5)
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            str(val), ha="center", va="bottom", fontsize=13, fontweight="bold")
ax.set_ylabel("Number of Variables", fontsize=11)
ax.set_title("Variables Before vs After 80% Missing Threshold\nBoth Datasets Collapse to Core Vitals",
             fontsize=13, fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/3_before_after_cleaning.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 3_before_after_cleaning.png")

# =========================
# 9) CHART — CLINICALLY CRITICAL VARIABLES LOST IN CLEANING
# =========================
physio_raw_canonical = set(physio_canonical.keys())
physio_lost          = physio_raw_canonical - PHYSIONET_CLEAN_CANONICAL
CRITICAL_VARS = [
    "Lactate", "Creatinine", "Bilirubin (Total)", "White Blood Cell Count",
    "Platelets", "Hematocrit", "Hemoglobin", "PTT", "Fibrinogen",
    "pH (Arterial)", "PaCO2 (Arterial)", "FiO2 (Inhaled O2 Fraction)",
    "Glucose", "Calcium", "Potassium", "Magnesium", "Phosphate", "Chloride",
]
lost_critical = [v for v in CRITICAL_VARS if v in physio_lost]

fig, ax = plt.subplots(figsize=(9, max(5, len(lost_critical) * 0.45)))
ax.barh(range(len(lost_critical)), [1]*len(lost_critical),
        color="#C00000", alpha=0.8, height=0.5)
ax.set_yticks(range(len(lost_critical)))
ax.set_yticklabels(lost_critical, fontsize=10)
ax.set_xticks([])
ax.set_title("Clinically Critical Variables Lost After Cleaning\n(>80% missing — key limitation to flag on poster)",
             fontsize=12, fontweight="bold")
ax.spines[["top", "right", "bottom", "left"]].set_visible(False)
for i, v in enumerate(lost_critical):
    ax.text(0.02, i, v, va="center", fontsize=10, color="white", fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/4_critical_vars_lost.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 4_critical_vars_lost.png")


# =========================
# 10) CHART — PIE CHART: RAW VARIABLE DISTRIBUTION
# =========================
fig, ax = plt.subplots(figsize=(7, 5))
counts = [len(both_raw), len(physio_only_raw), len(phems_only_raw)]
labels = [f"In Both\n({len(both_raw)})",
          f"PhysioNet Only\n({len(physio_only_raw)})",
          f"PHEMS Only\n({len(phems_only_raw)})"]
colors_pie = ["#2E75B6", "#ED7D31", "#70AD47"]
ax.pie(counts, labels=labels, colors=colors_pie, autopct="%1.0f%%",
       startangle=140, textprops={"fontsize": 11})
ax.set_title("Variable Distribution Across Datasets\n(Raw)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/5_variable_status_pie.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 5_variable_status_pie.png")

# =========================
# 11) CHART — CLINICALLY CRITICAL VARIABLES IN PHEMS BUT NOT PHYSIONET (RAW)
# =========================
# These exist in PHEMS raw but were never in PhysioNet at all
CRITICAL_PHEMS_ONLY = [
    "Lactate", "Creatinine", "Bilirubin (Total)", "White Blood Cell Count",
    "Neutrophils", "Platelets", "Procalcitonin", "CRP (C-Reactive Protein)",
    "Glasgow Coma Scale", "pH (Arterial)", "PaO2 (Arterial)",
]
phems_only_canonical = set(phems_only_raw["Variable (Canonical Name)"].tolist())
present_critical = [v for v in CRITICAL_PHEMS_ONLY if v in phems_only_canonical]

fig, ax = plt.subplots(figsize=(9, max(5, len(present_critical) * 0.55)))
ax.barh(range(len(present_critical)), [1]*len(present_critical),
        color="#C00000", alpha=0.8, height=0.5)
ax.set_yticks(range(len(present_critical)))
ax.set_yticklabels(present_critical, fontsize=10)
ax.set_xticks([])
ax.set_title("Clinically Critical Variables in PHEMS\nbut Not in PhysioNet (raw datasets)",
             fontsize=12, fontweight="bold")
ax.spines[["top", "right", "bottom", "left"]].set_visible(False)
for i, v in enumerate(present_critical):
    ax.text(0.02, i, v, va="center", fontsize=10, color="white", fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/6_critical_phems_only_vars.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 6_critical_phems_only_vars.png")

# =========================
# FINAL SUMMARY
# =========================
print("\n" + "=" * 60)
print("TASK 1 FINAL SUMMARY")
print("=" * 60)
print(f"\nRAW DATASETS:")
print(f"  PhysioNet original variables:  {len(physionet_vars)}")
print(f"  PHEMS original variables:      {len(phems_vars)}")
print(f"  Shared (raw):                  {len(both_raw)}")
print(f"  PhysioNet only (raw):          {len(physio_only_raw)}")
print(f"  PHEMS only (raw):              {len(phems_only_raw)}")
print(f"\nCLEAN DATASETS (after 80% threshold):")
print(f"  PhysioNet usable variables:    {len(PHYSIONET_CLEAN_VARS)}")
print(f"  PHEMS usable variables:        {len(PHEMS_CLEAN_VARS)}")
print(f"  Shared (clean):                {len(both_clean)}")
print(f"  PhysioNet only (clean):        {len(physio_only_clean)}")
print(f"  PHEMS only (clean):            {len(phems_only_clean)}")
print(f"\nKEY FINDING FOR POSTER:")
print(f"  Both datasets independently collapse to the same core vitals")
print(f"  after cleaning. This is not a PhysioNet limitation alone — it")
print(f"  reflects the reality of sparse lab data in real-world ICU records.")
print(f"  {len(lost_critical)} clinically critical variables were present but too sparse to use.")
print(f"\nOutputs saved to: {OUTPUT_DIR}")
print("\nDONE! Next step: Task 2 — Random Forest on clean shared variables")