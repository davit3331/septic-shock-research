import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT_DIR / "data" / "raw" / "physionet_2019.csv"

df = pd.read_csv(CSV_PATH)
if df.columns[0] == "Unnamed: 0":
    df = df.drop(columns=df.columns[0])

# SOFA relevant columns to check
sofa_cols = ['Lactate', 'Bilirubin_total', 'Bilirubin_direct', 
             'Creatinine', 'Platelets', 'FiO2', 'PaO2', 
             'GCS', 'MAP', 'SBP']

# Check missingness overall vs sepsis patients only
patient_labels = df.groupby("Patient_ID")["SepsisLabel"].max()
sepsis_ids = patient_labels[patient_labels == 1]


df_sepsis = df[df["Patient_ID"].isin(sepsis_ids.index)]

print("Column | Overall Missing | Sepsis-Only Missing | Sepsis Coverage")
print("-" * 70)
for col in sofa_cols:
    if col in df.columns:
        overall = df[col].isnull().mean() * 100
        sepsis_miss = df_sepsis[col].isnull().mean() * 100
        sepsis_cov = 100 - sepsis_miss
        print(f"{col:20} | {overall:6.1f}%          | {sepsis_miss:6.1f}%              | {sepsis_cov:6.1f}%")


print("\nPatient-level coverage (at least 1 reading):")
print("-" * 70)
for col in sofa_cols:
    if col in df.columns:
        # Overall: patients with at least 1 non-null reading
        overall_cov = df.groupby("Patient_ID")[col].apply(
            lambda x: x.notna().any()
        ).mean() * 100
        
        # Sepsis patients with at least 1 non-null reading
        sepsis_cov = df_sepsis.groupby("Patient_ID")[col].apply(
            lambda x: x.notna().any()
        ).mean() * 100
        
        print(f"{col:20} | Overall: {overall_cov:5.1f}% | Sepsis: {sepsis_cov:5.1f}%")



import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT_DIR / "data" / "raw" / "physionet_2019.csv"

df = pd.read_csv(CSV_PATH, nrows=1)
print(list(df.columns))