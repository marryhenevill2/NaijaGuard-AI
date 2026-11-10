

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from pathlib import Path

warnings.filterwarnings("ignore")

print("NaijaGuard AI development started!")

# Get the main project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Get the report directory
REPORT_DIR = BASE_DIR / "reports" / "eda"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Locate the dataset
DATA_PATH = BASE_DIR / "data" / "PS_20174392717_1491204439457.csv"

# Load the dataset
df = pd.read_csv(DATA_PATH)


# DATASET DIAGNOSTICS

print("\n========== DATASET DIAGNOSTICS ==========")

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 Records:")
print(df.head())

print("\nColumn Names:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nFraud Column Information:")
print(df["isFraud"].value_counts(dropna=False))

print("\nFraud Column Data Type:")
print(df["isFraud"].dtype)

print("\nIs Dataset Empty?")
print(df.empty)

print("\n========== END DIAGNOSTICS ==========")


# Display the first 5 rows
print("\nFirst 5 rows of the dataset:")
print(df.head())

# Display dataset information
print("\nDataset information:")
print(df.info())

# Display dataset shape
print("\nDataset shape:")
print(df.shape)

# Display column names
print("\nDataset columns:")
print(df.columns.tolist())

# Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# check duplicates
print("\nDuplicate rows:")
print(df.duplicated().sum())
 

# display head of the dataset
print("\nFirst 5 rows of the dataset:")
print(df.head())

# display tail of the dataset
print("\nLast 5 rows of the dataset:")
print(df.tail())

# display statistics summary of the dataset
print("\nStatistics summary of the dataset:")
print(df.describe(include="all"))

# display unique values of each column
print("\nUnique values of each column:")
print(df.nunique())

# display total missing values in the dataset
print("\nTotal missing values in the dataset:")
print(df.isnull().sum().sum())

# display value counts of each column
print("\nValue counts of each column:")
for col in df.columns:
    print(f"\nValue counts for '{col}':")
    print(df[col].value_counts(dropna=False))

print("\n========== Data Understanding Completed ==========")

# Display fraud distribution
print("========== Fraud Distribition ==========")

fraud_counts = df["isFraud"].value_counts()
fraud_labels = {
    0: "Legitimate",
    1: "Fraudulent",
}

print("\nFraud Percentage:")
print(df["isFraud"].value_counts(normalize=True).mul(100).rename(index=fraud_labels))

# visualize fraud distribution
plt.figure(figsize=(8, 5))
sns.countplot(
    x="isFraud",
    data=df,
    hue="isFraud",
    palette="Set2",
    legend=False,
)

plt.title("Legitimate vs Fraudulent Transactions")
plt.xlabel("Transaction Type")
plt.ylabel("Numbers of Transaction")

plt.xticks(ticks=[0, 1], labels=["Legitimate", "Fraudulent"])

plt.tight_layout()

plt.savefig(
    REPORT_DIR / "fraudulent_distribution.png",
    dpi=300
)
plt.show()

# Transaction type distribution
print("\n========== Transaction Analysis ===========")
plt.figure(figsize=(10, 6))
sns.countplot(
    data=df,
    x="type",
    order=df["type"].value_counts().index,
    hue="type",
    palette="viridis",
    legend=False
)

plt.title("Distribution of Transaction Types")
plt.xlabel("Transaction Type")
plt.ylabel("Number of Transactions")

plt.tight_layout()

plt.savefig(
    REPORT_DIR / "transaction_type_distribution.png",
    dpi=300
)

plt.show()

# Fraud by transaction type
print("\n========== Fraud by Transaction Type ==========")

fraud_by_type = pd.crosstab(
    df["type"],
    df["isFraud"]
)
print("\nFraud by Transaction Type:")
print(fraud_by_type)

fraud_rate_by_type = (
    df.groupby("type")["isFraud"]
    .mean()
    .mul(100)
    .sort_values(ascending=False)
)
print("\nFraud Rate by Transaction Type (%):")
print(fraud_rate_by_type)

plt.figure(figsize=(10, 6))
sns.barplot(
    x=fraud_rate_by_type.index,
    y=fraud_rate_by_type.values,
    hue=fraud_rate_by_type.index,
    palette="rocket",
    legend=False
)
plt.title("Fraud Rate by Transaction Type")
plt.xlabel("Transantion Type")
plt.ylabel("Fraud Rate (%)")

plt.tight_layout()

plt.savefig(
    REPORT_DIR / "fraud_by_transaction_type.png",
    dpi=300
)

plt.show()

# Transaction amount analysis
print("\n========== Transaction Amount Analysis ============")
print("\nAverage Transaction Amount:")
print(df["amount"].mean())

print("\nMedian transaction Amount:")
print(df["amount"].median())

print("\nmaximum transaction Amount:")
print(df["amount"].max())

# Amount distribution using log scale/
plt.figure(figsize=(10, 6))

sns.histplot(
    x="amount",
    data=df,
    bins=50,
    log_scale=(True, False)
)
plt.title("Distribution of Transaction Amount")
plt.xlabel("Transaction Amount (log scale)")
plt.ylabel("frequency")

plt.tight_layout()

plt.savefig(
    REPORT_DIR / "transaction_amount_distribution.png",
    dpi=300
)

plt.show()

# Fraudulent vs legitimate amount
print("\n========== Amount by Fraud Calass ==========")
amount_comparison = (
    df.groupby("isFraud")["amount"]
    .agg(["count", "mean", "median", "min", "max"])
)

amount_comparison.index = [
    "Legitimate",
    "Fraudulent"
]
print(amount_comparison)

plt.figure(figsize=(10, 6))

sns.boxplot(
    data=df,
    x="isFraud",
    y="amount",
    hue="isFraud",
    showfliers=False,
    palette="Set2",
    legend=False
)
plt.yscale("log")

plt.title("Transaction Amount by Fraud Classification")
plt.xlabel("Transaction Class")
plt.ylabel("Transaction Amount (log scale)")
plt.xticks(
    [0, 1],
    ["Legitimate", "Fraudulent"]
)

plt.tight_layout()

plt.savefig(
    REPORT_DIR / "amount_by_fraud.png",
    dpi=300
)

plt.show()

# Fraud over time
print("\n======= Fraud Over Time ========")

fraud_over_time = (
    df.groupby("step")["isFraud"]
    .sum()
)

print("\nFraudulent Transactions by Step:")
print(fraud_over_time.describe())

plt.figure(figsize=(12, 6))
plt.plot(
    fraud_over_time.index,
    fraud_over_time.values
)

plt.title("Fraudulent Transaction Over Time")
plt.xlabel("Transaction Step")
plt.ylabel("number of Fraudulent Transactions")

plt.tight_layout()

plt.savefig(
    REPORT_DIR / "fraud_over_time.png",
    dpi=300
)
plt.show()

# Correlation analysis
print("\n====== Correlation Analysis =========")

numeric_df = df.select_dtypes(include=np.number)

correlation = numeric_df.corr()
print("\nCorrelation with Fraud:")
print(
    correlation["isFraud"]
    .sort_values(ascending=False)
)

plt.figure(figsize=(12, 8))
sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Numerical Feature Correlation Heatmap")

plt.tight_layout()

plt.savefig(
    REPORT_DIR / "correlation_heatmap.png",
    dpi=300
)

plt.show()

# Save EDA results

print("\n======= EDA Completed ========")
print("\nChart save inside:")
print(REPORT_DIR)

print("\nNaijaGuard AI EDA completed successfully!")