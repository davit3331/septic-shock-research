import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules

from pathlib import Path

# =========================
# CONFIGURATION
# =========================
# Repo root: walk up until we find requirements.txt (works at any nesting depth).
ROOT_DIR = next(p for p in Path(__file__).resolve().parents if (p / "requirements.txt").exists())

PHYSIONET_PATH = ROOT_DIR / "outputs" / "discretization" / "PhysioNet_discretized.csv"
PHEMS_PATH = ROOT_DIR / "outputs" / "discretization" / "PHEMS_discretized.csv"
OUTPUT_DIR = ROOT_DIR / "outputs" / "association_rules"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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

    return physionet_df, phems_df

#For each patient row, create a list of their category labels + "Sepsis" or "No_Sepsis"
def create_transaction_list(df, feature_cols):
    transactions = []
    for _, row in df.iterrows():
        transaction = []
        for col in feature_cols:
            transaction.append(f"{row[col]}")
        transaction.append("Sepsis" if row["SepsisLabel"] == 1 else "No_Sepsis")
        transactions.append(transaction)
    return transactions


# FP-Growth algorithm implementation, with a minimum support threshold of 0.05 (5% of the dataset))
def perform_fp_growth(transactions, min_support=0.05):
    # Convert transactions to one-hot encoded DataFrame
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_ary, columns=te.columns_)

    # Apply FP-Growth algorithm
    frequent_itemsets = fpgrowth(df, min_support=min_support, use_colnames=True)
    
    return frequent_itemsets

#Generate Association Rules from frequent itemsets
def generate_association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5):
    rules = association_rules(frequent_itemsets, metric=metric, min_threshold=min_threshold)
    return rules

#Filter rules where consequent (right side) is "Sepsis"
def filter_sepsis_rules(rules):
    sepsis_rules = rules[rules['consequents'] == frozenset({'Sepsis'})]
    return sepsis_rules

#sort by confidence descending
def sort_rules_by_confidence(rules):
    sorted_rules = rules.sort_values(by='confidence', ascending=False)
    return sorted_rules

def printTop10Rules(sorted_rules):
    print("Top 10 Association Rules Predicting Sepsis:")
    for idx, row in sorted_rules.head(10).iterrows():
        antecedents = ', '.join(list(row['antecedents']))
        consequents = ', '.join(list(row['consequents']))
        print(f"Rule: {antecedents} => {consequents} | Support: {row['support']:.4f} | Confidence: {row['confidence']:.4f} | Lift: {row['lift']:.4f}")



#main  execution

physionet_df, phems_df = load_and_preprocess_data(PHYSIONET_PATH, PHEMS_PATH)
# Define feature columns (excluding target)
feature_cols = [col for col in physionet_df.columns if col.endswith("_categorized")]

# Create transactions for FP-Growth
physionet_transactions = create_transaction_list(physionet_df, feature_cols)

# Perform FP-Growth
physionet_frequent_itemsets = perform_fp_growth(physionet_transactions, min_support=0.05)
# Generate Association Rules
physionet_rules = generate_association_rules(physionet_frequent_itemsets, metric="confidence", min_threshold=0.5)
# Filter rules predicting Sepsis
physionet_sepsis_rules = filter_sepsis_rules(physionet_rules)
# Sort rules by confidence
physionet_sorted_rules = sort_rules_by_confidence(physionet_sepsis_rules)
# Print top 10 rules
print("Top 10 Association Rules Predicting Sepsis from PhysioNet:")
printTop10Rules(physionet_sorted_rules)

# Repeat the same process for PHEMS dataset
# Define feature columns (excluding target)
feature_cols_phems = [col for col in phems_df.columns if col.endswith("_categorized")]
# Create transactions for FP-Growth
phems_transactions = create_transaction_list(phems_df, feature_cols_phems)
# Perform FP-Growth
phems_frequent_itemsets = perform_fp_growth(phems_transactions, min_support=0.05)
# Generate Association Rules
phems_rules = generate_association_rules(phems_frequent_itemsets, metric="confidence", min_threshold=0.5)
# Filter rules predicting Sepsis
phems_sepsis_rules = filter_sepsis_rules(phems_rules)
# Sort rules by confidence
phems_sorted_rules = sort_rules_by_confidence(phems_sepsis_rules)
# Print top 10 rules
print("\nTop 10 Association Rules Predicting Sepsis from PHEMS:")
printTop10Rules(phems_sorted_rules)


# Save final association rule outputs
physionet_sorted_rules.to_csv(
    OUTPUT_DIR / "PhysioNet_sepsis_rules.csv",
    index=False
)

phems_sorted_rules.to_csv(
    OUTPUT_DIR / "PHEMS_sepsis_rules.csv",
    index=False
)

print("\nSaved association rule CSVs:")
print(f"- {OUTPUT_DIR / 'PhysioNet_sepsis_rules.csv'}")
print(f"- {OUTPUT_DIR / 'PHEMS_sepsis_rules.csv'}")

