
import numpy as np
import pandas as pd


def calculate_psi(reference, current, bins=10):

    # Create bin boundaries using historical data
    quantiles = np.linspace(0, 1, bins + 1)

    edges = np.unique(
        np.quantile(reference, quantiles)
    )

    if len(edges) < 2:
        edges = np.array([-np.inf, np.inf])
    else:
        edges[0] = -np.inf
        edges[-1] = np.inf

    # Count values in each bin
    reference_counts, _ = np.histogram(
        reference, bins=edges
    )

    current_counts, _ = np.histogram(
        current, bins=edges
    )

    # Convert counts into proportions
    reference_pct = reference_counts / len(reference)
    current_pct = current_counts / len(current)

    # Prevent log(0)
    epsilon = 1e-6

    reference_pct = np.clip(reference_pct, epsilon, None)
    current_pct = np.clip(current_pct, epsilon, None)

    # Calculate PSI
    psi = np.sum(
        (current_pct - reference_pct)
        * np.log(current_pct / reference_pct)
    )

    return float(psi)


def calculate_ecdf_distance(reference, current):

    reference = np.sort(reference)
    current = np.sort(current)

    # Combine unique values from both distributions
    combined = np.sort(
        np.unique(np.concatenate((reference, current)))
    )

    # Calculate empirical CDFs
    reference_cdf = (
        np.searchsorted(reference, combined, side="right")
        / len(reference)
    )

    current_cdf = (
        np.searchsorted(current, combined, side="right")
        / len(current)
    )

    # Maximum difference between distributions
    distance = np.max(
        np.abs(reference_cdf - current_cdf)
    )

    return float(distance)


def detect_drift(
    baseline_df,
    current_df,
    psi_threshold=0.2,
    ecdf_threshold=0.1
):

    results = {
        "columns": {},
        "drifted_columns": [],
        "skipped_columns": {}
    }

    common_columns = sorted(
        set(baseline_df.columns)
        & set(current_df.columns)
    )

    for column in common_columns:

        baseline_values = pd.to_numeric(
            baseline_df[column],
            errors="coerce"
        )

        current_values = pd.to_numeric(
            current_df[column],
            errors="coerce"
        )

        # Calculate how much of each column is numeric
        baseline_ratio = (
            baseline_values.notna().mean()
        )

        current_ratio = (
            current_values.notna().mean()
        )

        if baseline_ratio < 0.8 or current_ratio < 0.8:
            results["skipped_columns"][column] = (
                "Not sufficiently numeric"
            )
            continue

        baseline_values = baseline_values.dropna().to_numpy()
        current_values = current_values.dropna().to_numpy()

        if len(baseline_values) < 8 or len(current_values) < 8:
            results["skipped_columns"][column] = (
                "Insufficient observations"
            )
            continue

        # Calculate PSI
        psi = calculate_psi(
            baseline_values,
            current_values
        )

        # Calculate ECDF distance
        ecdf_distance = calculate_ecdf_distance(
            baseline_values,
            current_values
        )

        psi_detected = psi >= psi_threshold

        ecdf_detected = (
            ecdf_distance >= ecdf_threshold
        )

        drift_detected = (
            psi_detected or ecdf_detected
        )

        results["columns"][column] = {
            "psi": psi,
            "ecdf_distance": ecdf_distance,
            "psi_detected": bool(psi_detected),
            "ecdf_detected": bool(ecdf_detected),
            "drift_detected": bool(drift_detected)
        }

        if drift_detected:
            results["drifted_columns"].append(column)

    return results
