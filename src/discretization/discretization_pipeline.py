import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from pathlib import Path

# =========================
# CONFIGURATION
# =========================
ROOT_DIR = Path(__file__).resolve().parents[2]

PHYSIONET_PATH = ROOT_DIR / "data" / "processed" / "physionet_balanced.csv"
PHEMS_PATH = ROOT_DIR / "data" / "processed" / "phems_balanced.csv"
OUTPUT_DIR = ROOT_DIR / "outputs" / "discretization"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Column name mapping: PHEMS uses long names, we rename to short names
PHEMS_RENAME = {
    "Body temperature":                               "Temp",
    "Respiratory rate":                               "Resp",
    "Heart rate":                                     "HR",
    "Measurement of oxygen saturation at periphery":  "O2Sat",
    "age_in_months":                                  "Age",
    "gender":                                         "Gender",
}

#clean Phems outliers
def phems_outlier_filtering(phems_df):
    VALID_RANGES = {
        "HR":    (20,  300),
        "Temp":  (25,  45),
        "Resp":  (4,   80),
        "O2Sat": (50,  100),
        "Age":   (0,   300),
    }

    for col, (min_val, max_val) in VALID_RANGES.items():
        if col in phems_df.columns:
            invalid = (phems_df[col] < min_val) | (phems_df[col] > max_val)
            phems_df.loc[invalid, col] = None

    return phems_df

def aggregation_rules():
    agg_rules = {
        "HR":    "max",   # changed from mean
        "Temp":  "max",   # changed from mean
        "Resp":  "max",   # changed from mean
        "O2Sat": "min",   # changed from mean — min captures worst low reading
        "Age":   "first",
    }
    agg_rules["SepsisLabel"] = "max"
    return agg_rules

#Load and preprocess data
def load_and_preprocess_data(physionet_path, phems_path):
    # Load datasets from path
    physionet_df = pd.read_csv(physionet_path)
    phems_df = pd.read_csv(phems_path)

    # Drop rows with missing values in the target column
    physionet_df.dropna(subset=["SepsisLabel"], inplace=True)
    phems_df.dropna(subset=["SepsisLabel"], inplace=True)

    print("rows after dropping missing target - physionet:", len(physionet_df))
    print("rows after dropping missing target - phems:", len(phems_df))

    
    #rename phems columns to match physionet
    phems_df = phems_df.rename(columns=PHEMS_RENAME)
    #clean phems outliers
    phems_df = phems_outlier_filtering(phems_df)

    # Aggregate PhysioNet by Patient_ID
    agg_rules = aggregation_rules()
    phsyionet_agg = physionet_df.groupby("Patient_ID").agg(agg_rules).reset_index()


    # Aggregate PHEMS by Patient_ID
    phems_agg = phems_df.groupby("person_id").agg(agg_rules).reset_index()

    return phsyionet_agg, phems_agg


def discretize_physionet(df):

    df["HR_categorized"] = np.where(
        df["HR"] > 127, "HR_high",
        np.where(df["HR"] < 60, "HR_low", "HR_normal")
    )
    
    df["Temp_categorized"] = np.where(
        df["Temp"] > 38.45, "Temp_high",
        np.where(df["Temp"] < 36.0, "Temp_low", "Temp_normal")
)

    df["Resp_categorized"] = np.where(
        df["Resp"] > 20, "Resp_high",
        np.where(df["Resp"] < 12, "Resp_low", "Resp_normal")
    )
    df["O2Sat_categorized"] = np.where(
        df["O2Sat"] < 92.6, "O2Sat_low",
        np.where(df["O2Sat"] > 99, "O2Sat_high", "O2Sat_normal")
    )

    """
    HR:   > 127   = high,  < 60   = low,   else = normal
    Temp:  > 38.45 = high,  < 36.0 = low,  else = normal
    Resp:  > 20    = high,  < 12   = low,   else = normal
    O2Sat: < 92.6  = low,   > 99   = high,  else = normal
    """
    return df


def discretize_phems(df):
    # PHEMS thresholds (pediatric ICU)
    df["HR_categorized"] = np.where(
        df["HR"] > 106, "HR_high",
        np.where(df["HR"] < 60, "HR_low", "HR_normal")
    )
    df["Temp_categorized"] = np.where(
        df["Temp"] > 37.29, "Temp_high",
        np.where(df["Temp"] < 35.0, "Temp_low", "Temp_normal")
    )
    df["Resp_categorized"] = np.where(
        df["Resp"] > 19, "Resp_high",
        np.where(df["Resp"] < 12, "Resp_low", "Resp_normal")
    )
    df["O2Sat_categorized"] = np.where(
        df["O2Sat"] < 97.01, "O2Sat_low",
        np.where(df["O2Sat"] > 99, "O2Sat_high", "O2Sat_normal")
    )
    return df

# Main execution
physionet_agg, phems_agg = load_and_preprocess_data(PHYSIONET_PATH, PHEMS_PATH)

physionet_agg = discretize_physionet(physionet_agg)
phems_agg = discretize_phems(phems_agg)

physionet_agg.to_csv(OUTPUT_DIR / "PhysioNet_discretized.csv", index=False)
phems_agg.to_csv(OUTPUT_DIR / "PHEMS_discretized.csv", index=False)
print("Saved discretized datasets")