import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from pathlib import Path

# =====================================================
# CONFIGURATION
# =====================================================
# Repo root: walk up until we find requirements.txt (works at any nesting depth).
ROOT_DIR = next(p for p in Path(__file__).resolve().parents if (p / "requirements.txt").exists())

PHYSIONET_PATH = ROOT_DIR / "data" / "processed" / "physionet_balanced.csv"
PHEMS_PATH = ROOT_DIR / "data" / "processed" / "phems_balanced.csv"
OUTPUT_DIR = ROOT_DIR / "outputs" / "feature_analysis" / "random_forest"

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

# Physiologically valid ranges for PHEMS outlier removal
VALID_RANGES = {
    "HR":    (20,  300),
    "Temp":  (25,  45),
    "Resp":  (4,   80),
    "O2Sat": (50,  100),
    "Age":   (0,   150),
}

# How many trees to build in the Random Forest
NUMBER_OF_TREES = 500

# Random seed so results are reproducible every run
RANDOM_SEED = 42

print("=" * 60)
print("TASK 2 — STEP 2: RANDOM FOREST FEATURE IMPORTANCE")
print("=" * 60)


# =====================================================
# FUNCTION: PREPARE DATA
# Aggregates rows to one row per patient
# Returns X (features) and y (labels)
# =====================================================
def prepare_patient_data(dataframe, feature_columns, patient_id_column, label_column, dataset_name):

    print(f"\n--- Preparing data for {dataset_name} ---")

    # Drop any rows where the label is missing
    # We cannot train on rows with unknown labels
    dataframe = dataframe.dropna(subset=[label_column])
    print(f"Rows after removing missing labels: {len(dataframe):,}")

    # Group by patient and take the mean of each variable
    # This gives us one row per patient instead of one row per hour
    # Also take the MAX of the label (if patient ever had sepsis = 1)
    aggregation_rules = {}
    for column in feature_columns:
        aggregation_rules[column] = "mean"
    aggregation_rules[label_column] = "max"

    # Apply the aggregation, after these 2 lines every patient has ONE ROW
    df_per_patient = dataframe.groupby(patient_id_column).agg(aggregation_rules)
    df_per_patient = df_per_patient.reset_index()

    # Count how many patients in each class
    number_of_sepsis    = (df_per_patient[label_column] == 1).sum()
    number_of_nosepsis  = (df_per_patient[label_column] == 0).sum()
    print(f"Total patients: {len(df_per_patient):,}")
    print(f"Sepsis patients: {number_of_sepsis:,}")
    print(f"No-sepsis patients: {number_of_nosepsis:,}")

    # Separate features (X) from labels (y)
    # X is the table of variable values
    # y is the list of sepsis labels (0 or 1)
    X = df_per_patient[feature_columns]
    y = df_per_patient[label_column]

    print(f"Feature matrix shape (patients x variables): {X.shape}")
    print(f"Label vector shape: {y.shape}")

    return X, y


# =====================================================
# FUNCTION: RUN RANDOM FOREST
# Trains the model and returns feature importance scores
# =====================================================
def run_random_forest(X, y, feature_columns, dataset_name):

    print(f"\n--- Running Random Forest for {dataset_name} ---")

    # Some patients may have missing values for some variables
    # Random Forest cannot handle NaN values
    # We fill missing values with the mean of that column
    # This is called "mean imputation"
    print("Filling missing values with column means (imputation)...")
    imputer = SimpleImputer(strategy="mean")
    X_filled = imputer.fit_transform(X)
    print(f"Missing values filled. Matrix shape: {X_filled.shape}")

    # Convert the label to integer (0 or 1)
    y_integer = y.astype(int)

    # Create the Random Forest model
    # n_estimators = how many trees to build (500 is reliable)
    # random_state = fixed seed so we get same results every run
    print(f"Building Random Forest with {NUMBER_OF_TREES} trees...")
    random_forest_model = RandomForestClassifier(
        n_estimators=NUMBER_OF_TREES,
        random_state=RANDOM_SEED
    )

    # Train the model on all our data
    # The model learns which variables best split sepsis vs no-sepsis
    random_forest_model.fit(X_filled, y_integer)
    print("Training complete.")

    # Extract the feature importance scores
    # These are numbers between 0 and 1 that add up to 1.0
    # Higher number = more important for predicting sepsis
    importance_scores = random_forest_model.feature_importances_

    # Put the scores into a table with variable names
    results_table = pd.DataFrame({
        "Variable":   feature_columns,
        "Importance": importance_scores
    })

    # Sort from most important to least important
    results_table = results_table.sort_values("Importance", ascending=False)
    results_table = results_table.reset_index(drop=True)

    # Print the results
    print(f"\n--- FEATURE IMPORTANCE RESULTS ({dataset_name}) ---")
    for position, row in results_table.iterrows():
        bar = "█" * int(row["Importance"] * 100)
        print(f"  {position+1}. {row['Variable']:<10} {row['Importance']:.4f}  {bar}")

    return results_table


# =====================================================
# FUNCTION: MAKE CHART
# Saves a horizontal bar chart of feature importances
# =====================================================
def make_importance_chart(results_table, dataset_name):

    print(f"\nGenerating importance chart for {dataset_name}...")

    # Get the variable names and their scores
    variable_names  = results_table["Variable"].tolist()
    importance_vals = results_table["Importance"].tolist()

    # Assign colors — top 3 variables get red, rest get blue
    bar_colors = []
    for i in range(len(variable_names)):
        if i < 3:
            bar_colors.append("#C00000")   # red = top 3
        else:
            bar_colors.append("#2E75B6")   # blue = others

    # Create the figure
    figure, axis = plt.subplots(figsize=(9, 5))

    # Draw the horizontal bars
    axis.barh(
        variable_names,
        importance_vals,
        color=bar_colors,
        edgecolor="white",
        height=0.5
    )

    # Add the importance score as a number next to each bar
    for index, value in enumerate(importance_vals):
        axis.text(
            value + 0.002,
            index,
            f"{value:.4f}",
            va="center",
            fontsize=10
        )

    # Add labels and title
    axis.set_xlabel("Feature Importance Score (higher = more predictive)", fontsize=11)
    axis.set_title(
        f"{dataset_name} — Random Forest Feature Importance\n(Red = Top 3 variables)",
        fontsize=13,
        fontweight="bold"
    )
    axis.spines[["top", "right"]].set_visible(False)
    axis.invert_yaxis()  # most important variable at the top

    # Save the chart
    plt.tight_layout()
    chart_filename = f"{dataset_name}_4_random_forest_importance.png"
    chart_path = OUTPUT_DIR / chart_filename
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {chart_filename}")


# =====================================================
# STEP A: LOAD AND PREPARE PHYSIONET DATA
# =====================================================
print("\n\nLOADING PHYSIONET DATASET")
print("-" * 40)

# Load the balanced PhysioNet CSV file
df_physionet = pd.read_csv(PHYSIONET_PATH)

# Remove the unnamed index column if it exists
if df_physionet.columns[0] == "Unnamed: 0":
    df_physionet = df_physionet.drop(columns=df_physionet.columns[0])

print(f"Loaded PhysioNet: {df_physionet.shape[0]:,} rows, {df_physionet.shape[1]} columns")

# Detect feature columns dynamically from the file
# Keep any column that is not in the non-feature list
physionet_feature_columns = []
for column_name in df_physionet.columns:
    if column_name not in PHYSIONET_NON_FEATURE:
        physionet_feature_columns.append(column_name)

print(f"Feature columns detected ({len(physionet_feature_columns)}): {physionet_feature_columns}")


# =====================================================
# STEP B: LOAD AND PREPARE PHEMS DATA
# =====================================================
print("\n\nLOADING PHEMS DATASET")
print("-" * 40)

# Load the balanced PHEMS CSV file
df_phems = pd.read_csv(PHEMS_PATH)
print(f"Loaded PHEMS: {df_phems.shape[0]:,} rows, {df_phems.shape[1]} columns")

# Step 1: Encode gender from text to number
# MALE = 1, FEMALE = 0
# We do this BEFORE renaming so we target the right column name
if "gender" in df_phems.columns:
    print("Encoding gender column (MALE=1, FEMALE=0)...")
    df_phems["gender"] = df_phems["gender"].map({"MALE": 1, "FEMALE": 0})
    df_phems["gender"] = pd.to_numeric(df_phems["gender"], errors="coerce")

# Step 2: Rename PHEMS columns to short names so they match PhysioNet
print("Renaming PHEMS columns to short names...")
df_phems = df_phems.rename(columns=PHEMS_RENAME)

# Step 3: Remove physiologically impossible outlier values
# Some PHEMS rows have corrupt entries like HR = 156,775
# These would destroy our averages so we set them to NaN
print("Removing physiologically impossible outlier values...")
for column_name, (min_value, max_value) in VALID_RANGES.items():
    if column_name in df_phems.columns:
        # Find rows where the value is outside the valid range
        invalid_mask = (df_phems[column_name] < min_value) | (df_phems[column_name] > max_value)
        number_of_invalid = invalid_mask.sum()
        if number_of_invalid > 0:
            print(f"  {column_name}: setting {number_of_invalid} outlier values to NaN")
            df_phems.loc[invalid_mask, column_name] = None

# Step 4: Detect feature columns from the renamed file
phems_feature_columns = []
for column_name in df_phems.columns:
    if column_name not in PHEMS_NON_FEATURE:
        phems_feature_columns.append(column_name)

print(f"Feature columns detected ({len(phems_feature_columns)}): {phems_feature_columns}")


# =====================================================
# STEP C: RUN ANALYSIS ON BOTH DATASETS
# =====================================================

# --- PhysioNet ---
X_physionet, y_physionet = prepare_patient_data(
    df_physionet,
    physionet_feature_columns,
    "Patient_ID",
    "SepsisLabel",
    "PhysioNet"
)
physionet_results = run_random_forest(X_physionet, y_physionet, physionet_feature_columns, "PhysioNet")
make_importance_chart(physionet_results, "PhysioNet")

# --- PHEMS ---
X_phems, y_phems = prepare_patient_data(
    df_phems,
    phems_feature_columns,
    "person_id",
    "SepsisLabel",
    "PHEMS"
)
phems_results = run_random_forest(X_phems, y_phems, phems_feature_columns, "PHEMS")
make_importance_chart(phems_results, "PHEMS")


# =====================================================
# STEP D: COMPARE STEP 1 VS STEP 2 RANKINGS
# =====================================================
print("\n\n" + "=" * 60)
print("COMPARISON: CORRELATION (Step 1) vs RANDOM FOREST (Step 2)")
print("=" * 60)

# PhysioNet comparison
print("\nPhysioNet ranking comparison:")
print(f"{'Rank':<6} {'Step 1 (Correlation)':<25} {'Step 2 (Random Forest)':<25}")
print("-" * 56)

# Step 1 ranking from our previous results
step1_physionet = ["HR", "Resp", "Temp", "MAP", "DBP", "SBP", "Gender", "O2Sat", "Age"]

for rank in range(len(physionet_results)):
    step1_var = step1_physionet[rank] if rank < len(step1_physionet) else "-"
    step2_var = physionet_results.iloc[rank]["Variable"]
    match = "✓" if step1_var == step2_var else "~"
    print(f"  {rank+1:<4} {step1_var:<25} {step2_var:<25} {match}")


# =====================================================
# FINAL SUMMARY
# =====================================================
print("\n\n" + "=" * 60)
print("STEP 2 FINAL SUMMARY")
print("=" * 60)

print("\nPhysioNet Top 4 Variables (Random Forest):")
for i, row in physionet_results.head(4).iterrows():
    print(f"  {i+1}. {row['Variable']:<10} importance = {row['Importance']:.4f} ({row['Importance']*100:.1f}%)")

print("\nPHEMS Top 4 Variables (Random Forest):")
for i, row in phems_results.head(4).iterrows():
    print(f"  {i+1}. {row['Variable']:<10} importance = {row['Importance']:.4f} ({row['Importance']*100:.1f}%)")

print(f"\nOutputs saved to: {OUTPUT_DIR}")
print("\nDONE! Next step: Step 3 — Decision Tree thresholds on top variables")