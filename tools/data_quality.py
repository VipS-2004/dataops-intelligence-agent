
import pandas as pd


def analyze_data_quality(df):

    results = {
        "missing_values": {},
        "duplicate_records": 0,
        "duplicate_transaction_ids": 0,
        "invalid_values": {},
        "range_violations": {}
    }

    # 1. Missing value detection
    missing = df.isnull().sum()

    results["missing_values"] = {
        column: int(count)
        for column, count in missing.items()
        if count > 0
    }

    # 2. Duplicate record detection
    results["duplicate_records"] = int(
        df.duplicated().sum()
    )

    # 3. Duplicate transaction ID detection
    if "transaction_id" in df.columns:
        results["duplicate_transaction_ids"] = int(
            df["transaction_id"].duplicated().sum()
        )

    # 4. Invalid purchase amounts
    if "purchase_amount" in df.columns:
        invalid_amounts = pd.to_numeric(
            df["purchase_amount"], errors="coerce"
        )

        results["invalid_values"]["negative_purchase_amounts"] = int(
            (invalid_amounts < 0).sum()
        )

    # 5. Age range validation
    if "age" in df.columns:
        ages = pd.to_numeric(
            df["age"], errors="coerce"
        )

        results["range_violations"]["invalid_ages"] = int(
            ((ages < 0) | (ages > 120)).sum()
        )

    return results
