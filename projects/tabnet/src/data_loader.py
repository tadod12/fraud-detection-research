"""
Data loader for the bank transaction fraud detection dataset.

Downloads from Kaggle if credentials available, otherwise generates
a synthetic dataset matching the paper's description for testing.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

from config import DATA_DIR, RANDOM_SEED


def download_dataset() -> Path:
    """
    Download the bank transaction fraud dataset from Kaggle.

    Tries kagglehub first; if not available, generates synthetic data.
    Returns the path to the CSV file.
    """
    csv_path = DATA_DIR / "bank_transactions.csv"

    if csv_path.exists():
        print(f"[INFO] Dataset already exists at {csv_path}")
        return csv_path

    print("[INFO] Attempting to download dataset from Kaggle...")
    try:
        import kagglehub

        dataset_path = kagglehub.dataset_download(
            "valakhorasani/bank-transaction-fraud-detection"
        )
        print(f"[INFO] Dataset downloaded to {dataset_path}")

        dataset_path = Path(dataset_path)
        csv_files = list(dataset_path.rglob("*.csv"))

        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {dataset_path}")

        import shutil
        source_csv = csv_files[0]
        shutil.copy2(source_csv, csv_path)
        print(f"[INFO] Dataset saved to {csv_path}")
        return csv_path

    except Exception as e:
        print(f"[WARN] Kaggle download failed: {e}")
        print("[INFO] Generating synthetic dataset matching paper description...")
        return generate_synthetic_dataset(csv_path)


def generate_synthetic_dataset(csv_path: Path, n_samples: int = 100000) -> Path:
    """
    Generate a synthetic bank transaction fraud dataset.

    Mimics the distribution described in the paper:
      - Mix of numerical and categorical features
      - ~2% fraud ratio (heavily imbalanced)
      - Features typical of Indian bank transactions
    """
    np.random.seed(RANDOM_SEED)

    n_fraud = int(n_samples * 0.02)  # ~2% fraud
    n_legit = n_samples - n_fraud

    # --- Numerical features ---
    transaction_amount_legit = np.random.lognormal(mean=7, sigma=1.5, size=n_legit)
    transaction_amount_fraud = np.random.lognormal(mean=9, sigma=1.8, size=n_fraud)

    account_balance_legit = np.random.lognormal(mean=10, sigma=1.2, size=n_legit)
    account_balance_fraud = np.random.lognormal(mean=9, sigma=1.5, size=n_fraud)

    transaction_hour_legit = np.random.normal(14, 4, n_legit).clip(0, 23).astype(int)
    transaction_hour_fraud = np.random.choice([0, 1, 2, 3, 22, 23], size=n_fraud)

    num_transactions_day_legit = np.random.poisson(3, n_legit)
    num_transactions_day_fraud = np.random.poisson(8, n_fraud)

    days_since_last_txn_legit = np.random.exponential(2, n_legit)
    days_since_last_txn_fraud = np.random.exponential(15, n_fraud)

    login_attempts_legit = np.random.poisson(1, n_legit)
    login_attempts_fraud = np.random.poisson(4, n_fraud)

    avg_txn_amount_legit = transaction_amount_legit * np.random.uniform(0.8, 1.2, n_legit)
    avg_txn_amount_fraud = transaction_amount_fraud * np.random.uniform(0.2, 0.5, n_fraud)

    amount_deviation_legit = np.abs(np.random.normal(0, 0.3, n_legit))
    amount_deviation_fraud = np.abs(np.random.normal(2, 1, n_fraud))

    # --- Categorical features ---
    transaction_types = ["Transfer", "Withdrawal", "Deposit", "Payment", "POS"]
    merchant_categories = ["Retail", "Food", "Travel", "Entertainment", "Online", "Utilities", "ATM"]
    channels = ["Mobile", "Internet", "Branch", "ATM", "POS"]
    cities = ["Mumbai", "Delhi", "Bangalore", "Chennai", "Kolkata", "Hyderabad", "Pune", "Ahmedabad"]
    account_types = ["Savings", "Current", "Salary"]

    txn_type_legit = np.random.choice(transaction_types, n_legit, p=[0.25, 0.2, 0.2, 0.2, 0.15])
    txn_type_fraud = np.random.choice(transaction_types, n_fraud, p=[0.4, 0.3, 0.05, 0.15, 0.1])

    merchant_legit = np.random.choice(merchant_categories, n_legit)
    merchant_fraud = np.random.choice(["Online", "ATM", "Travel"], n_fraud)

    channel_legit = np.random.choice(channels, n_legit, p=[0.3, 0.3, 0.15, 0.15, 0.1])
    channel_fraud = np.random.choice(channels, n_fraud, p=[0.15, 0.4, 0.05, 0.3, 0.1])

    city_legit = np.random.choice(cities, n_legit)
    city_fraud = np.random.choice(cities, n_fraud)

    acct_type_legit = np.random.choice(account_types, n_legit, p=[0.5, 0.3, 0.2])
    acct_type_fraud = np.random.choice(account_types, n_fraud, p=[0.4, 0.4, 0.2])

    is_international_legit = np.random.choice([0, 1], n_legit, p=[0.95, 0.05])
    is_international_fraud = np.random.choice([0, 1], n_fraud, p=[0.6, 0.4])

    is_weekend_legit = np.random.choice([0, 1], n_legit, p=[0.7, 0.3])
    is_weekend_fraud = np.random.choice([0, 1], n_fraud, p=[0.4, 0.6])

    # --- Combine ---
    def concat_arrays(legit, fraud):
        return np.concatenate([legit, fraud])

    data = {
        "transaction_amount": concat_arrays(transaction_amount_legit, transaction_amount_fraud),
        "account_balance": concat_arrays(account_balance_legit, account_balance_fraud),
        "transaction_hour": concat_arrays(transaction_hour_legit, transaction_hour_fraud),
        "num_transactions_day": concat_arrays(num_transactions_day_legit, num_transactions_day_fraud),
        "days_since_last_transaction": concat_arrays(days_since_last_txn_legit, days_since_last_txn_fraud),
        "login_attempts": concat_arrays(login_attempts_legit, login_attempts_fraud),
        "avg_transaction_amount": concat_arrays(avg_txn_amount_legit, avg_txn_amount_fraud),
        "amount_deviation": concat_arrays(amount_deviation_legit, amount_deviation_fraud),
        "transaction_type": concat_arrays(txn_type_legit, txn_type_fraud),
        "merchant_category": concat_arrays(merchant_legit, merchant_fraud),
        "channel": concat_arrays(channel_legit, channel_fraud),
        "city": concat_arrays(city_legit, city_fraud),
        "account_type": concat_arrays(acct_type_legit, acct_type_fraud),
        "is_international": concat_arrays(is_international_legit, is_international_fraud),
        "is_weekend": concat_arrays(is_weekend_legit, is_weekend_fraud),
        "is_fraud": np.concatenate([np.zeros(n_legit, dtype=int), np.ones(n_fraud, dtype=int)]),
    }

    df = pd.DataFrame(data)
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    # Save
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"[INFO] Synthetic dataset generated: {n_samples} samples, "
          f"{n_fraud} fraud ({n_fraud/n_samples*100:.1f}%)")
    print(f"[INFO] Saved to {csv_path}")
    return csv_path


def load_dataset(csv_path: Path = None) -> pd.DataFrame:
    """Load the dataset from CSV."""
    if csv_path is None:
        csv_path = download_dataset()

    print(f"[INFO] Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"[INFO] Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def profile_dataset(df: pd.DataFrame) -> dict:
    """Generate a basic profile of the dataset."""
    target_candidates = [
        col for col in df.columns
        if any(kw in col.lower() for kw in ["fraud", "target", "label", "class"])
    ]
    target_col = target_candidates[0] if target_candidates else None

    profile = {
        "n_rows": df.shape[0],
        "n_columns": df.shape[1],
        "columns": list(df.columns),
        "dtypes": df.dtypes.value_counts().to_dict(),
        "missing_values": df.isnull().sum().sum(),
        "missing_by_column": df.isnull().sum()[df.isnull().sum() > 0].to_dict(),
        "target_column": target_col,
    }

    if target_col:
        value_counts = df[target_col].value_counts()
        profile["class_distribution"] = value_counts.to_dict()
        fraud_count = value_counts.get(1, 0)
        total = len(df)
        profile["fraud_ratio"] = fraud_count / total if total > 0 else 0

    print("\n" + "=" * 60)
    print("DATASET PROFILE")
    print("=" * 60)
    print(f"  Rows:              {profile['n_rows']:,}")
    print(f"  Columns:           {profile['n_columns']}")
    print(f"  Missing values:    {profile['missing_values']:,}")
    if target_col:
        print(f"  Target column:     {target_col}")
        print(f"  Class distribution:")
        for cls, count in profile["class_distribution"].items():
            print(f"    {cls}: {count:,} ({count/profile['n_rows']*100:.2f}%)")
        print(f"  Fraud ratio:       {profile['fraud_ratio']:.4f}")
    print("=" * 60 + "\n")

    return profile


if __name__ == "__main__":
    df = load_dataset()
    profile = profile_dataset(df)
