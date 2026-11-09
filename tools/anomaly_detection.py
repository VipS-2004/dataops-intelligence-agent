
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def detect_anomalies(
    df,
    contamination=0.1,
    random_state=42
):

    if df.empty or len(df) < 8:
        raise ValueError(
            "Dataset must contain at least 8 records"
        )

    if not 0 < contamination <= 0.5:
        raise ValueError(
            "Contamination must be between 0 and 0.5"
        )

    # Convert columns to numeric where possible
    numeric_df = df.apply(
        lambda column: pd.to_numeric(
            column, errors="coerce"
        )
    )

    # Select columns with at least 80% numeric values
    valid_columns = [
        column
        for column in numeric_df.columns
        if numeric_df[column].notna().mean() >= 0.8
    ]

    if not valid_columns:
        raise ValueError(
            "No suitable numerical columns found"
        )

    X = numeric_df[valid_columns].copy()

    # Fill missing values using column medians
    X = X.fillna(X.median())

    # Remove constant columns
    X = X.loc[:, X.nunique() > 1]

    if X.empty:
        raise ValueError(
            "No useful numerical features found"
        )

    # Create and train Isolation Forest
    model = IsolationForest(
        contamination=contamination,
        random_state=random_state
    )

    predictions = model.fit_predict(X)

    # Higher score means more anomalous
    anomaly_scores = -model.decision_function(X)

    # Extract suspicious records
    anomaly_indices = np.where(predictions == -1)[0]

    anomalies = []

    for index in anomaly_indices:

        anomalies.append({
            "row_index": df.index[index],
            "anomaly_score": float(
                anomaly_scores[index]
            ),
            "values": X.iloc[index].to_dict()
        })

    results = {
        "features_used": X.columns.tolist(),
        "total_records": len(df),
        "anomaly_count": len(anomalies),
        "anomaly_rate": len(anomalies) / len(df),
        "anomalies": anomalies,
        "warning": (
            "Small dataset: results are exploratory"
            if len(df) < 100 else None
        )
    }

    return results
