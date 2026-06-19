import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from pathlib import Path

# =========================
# CONFIGURATION
# =========================
# Repo root: walk up until we find requirements.txt (works at any nesting depth).
ROOT_DIR = next(p for p in Path(__file__).resolve().parents if (p / "requirements.txt").exists())

PHYSIONET_PATH = ROOT_DIR / "data" / "processed" / "physionet_balanced.csv"
PHEMS_PATH = ROOT_DIR / "data" / "processed" / "phems_balanced.csv"
OUTPUT_DIR = ROOT_DIR / "outputs" / "decision_trees"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# These are the non-feature columns we want to ignore
PHYSIONET_NON_FEATURE = ["Patient_ID", "Hour", "SepsisLabel", "HospAdmTime", "ICULOS"]
PHEMS_NON_FEATURE     = ["visit_occurrence_id", "person_id", "measurement_datetime", "SepsisLabel"]


# Column name mapping: PHEMS uses long names, we rename to short names
PHEMS_RENAME = {
    "Body temperature":                               "Temp",
    "Respiratory rate":                               "Resp",
    "Heart rate":                                     "HR",
    "Measurement of oxygen saturation at periphery":  "O2Sat",
    "age_in_months":                                  "Age",
    "gender":                                         "Gender",
}


##Outlier filtering for Phsems
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


def load_and_preprocess_data(physionet_path, phems_path):
    # Load datasets
    physionet_df = pd.read_csv(physionet_path)
    phems_df = pd.read_csv(phems_path)

    # Drop rows with missing values in the target column
    physionet_df.dropna(subset=["SepsisLabel"], inplace=True)
    phems_df.dropna(subset=["SepsisLabel"], inplace=True)

    print("rows after dropping missing target - physionet:", len(physionet_df))
    print("rows after dropping missing target - phems:", len(phems_df))

    # --- PhysioNet aggregation rules ---
    # Built from PhysioNet columns only
    physionet_features = physionet_df.drop(columns=PHYSIONET_NON_FEATURE)
    physionet_agg_rules = {}
    for col in physionet_features.columns:
        if col in ["Temp", "HR", "O2Sat", "Resp"]:
            physionet_agg_rules[col] = 'mean'
        elif col == "SepsisLabel":
            physionet_agg_rules[col] = 'max'
        else:
            physionet_agg_rules[col] = 'first'
    physionet_agg_rules["SepsisLabel"] = "max"

    # Aggregate PhysioNet by Patient_ID
    physionet_agg = physionet_df.groupby("Patient_ID").agg(physionet_agg_rules).reset_index()

    # --- PHEMS aggregation rules ---
    # Rename columns first, then build rules from PHEMS columns only
    phems_df = phems_df.rename(columns=PHEMS_RENAME)

    ## call the function we wrote to filter outliers
    phems_df = phems_outlier_filtering(phems_df)

    phems_features = phems_df.drop(columns=PHEMS_NON_FEATURE)
    phems_agg_rules = {}
    for col in phems_features.columns:
        if col in ["Temp", "HR", "O2Sat", "Resp"]:
            phems_agg_rules[col] = 'mean'
        elif col == "SepsisLabel":
            phems_agg_rules[col] = 'max'
        else:
            phems_agg_rules[col] = 'first'
    phems_agg_rules["SepsisLabel"] = "max"

    # Aggregate PHEMS by person_id
    phems_agg = phems_df.groupby("person_id").agg(phems_agg_rules).reset_index()

    phems_features = phems_agg.drop(columns=["person_id"])

    print("PhysioNet agg columns:", physionet_agg.columns.tolist())
    print("PHEMS agg columns:", phems_agg.columns.tolist())
    return physionet_agg[["Temp", "HR", "Resp", "O2Sat", "SepsisLabel"]], phems_agg[["Temp", "HR", "Resp", "O2Sat", "SepsisLabel"]]


#decision tree
def train_decision_tree(dataframe, label_column):

    X_train = dataframe.drop(columns=[label_column])
    y_train = dataframe[label_column]
    model = DecisionTreeClassifier(random_state=42, max_depth=4)
    model.fit(X_train, y_train)

    return model



def extractThresholds(decision_tree_model, feature_names):
    thresholds = {}
    for i in range(decision_tree_model.tree_.node_count):
        if decision_tree_model.tree_.feature[i] != -2:  # Check if it's not a leaf node
            feature_index = decision_tree_model.tree_.feature[i]
            threshold_value = decision_tree_model.tree_.threshold[i]
            thresholds[feature_names[feature_index]] = threshold_value
    return thresholds


def visualize_decision_tree(model, feature_names, dataset_name, output_dir):
    
    # Create a large figure so the tree is readable
    figure, axis = plt.subplots(figsize=(20, 10))
    
    # Draw the tree
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=["No Sepsis", "Sepsis"],
        filled=True,          # color the nodes
        rounded=True,         # rounded corners
        fontsize=10,
        ax=axis
    )
    
    # Add a title
    axis.set_title(
        f"{dataset_name} — Decision Tree Structure",
        fontsize=14,
        fontweight="bold"
    )
    
    # Save the chart
    plt.tight_layout()
    chart_path = output_dir / f"{dataset_name}_5_decision_tree.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {dataset_name}_5_decision_tree.png")



# Main execution
physionet_data, phems_data = load_and_preprocess_data(PHYSIONET_PATH, PHEMS_PATH)

# Train decision tree on PhysioNet data
physionet_tree = train_decision_tree(physionet_data, "SepsisLabel")
# Train decision tree on PHEMS data
phems_tree = train_decision_tree(phems_data, "SepsisLabel")

# Extract thresholds
physionet_thresholds = extractThresholds(physionet_tree, physionet_data.drop(columns=["SepsisLabel"]).columns)
phems_thresholds = extractThresholds(phems_tree, phems_data.drop(columns=["SepsisLabel"]).columns)

print("\n" + "=" * 50)
print("DECISION TREE THRESHOLDS")
print("=" * 50)


print("\nPhysioNet Thresholds:")
for variable, threshold in physionet_thresholds.items():
    if threshold != float('inf'): #if the threshold is not infinity, which means it's a valid split
        print(f"PhysioNet - Variable: {variable:<10} Threshold: {round(threshold, 2)}")

print("\nPHEMS Thresholds:")
for variable, threshold in phems_thresholds.items():
    if threshold != float('inf'):
       print(f"PHEMS - Variable: {variable:<10}  Threshold: {round(threshold, 2)}")



# Visualize the trees
visualize_decision_tree(
    physionet_tree,
    physionet_data.drop(columns=["SepsisLabel"]).columns.tolist(),
    "PhysioNet",
    OUTPUT_DIR
)

visualize_decision_tree(
    phems_tree,
    phems_data.drop(columns=["SepsisLabel"]).columns.tolist(),
    "PHEMS",
    OUTPUT_DIR
)