
import pandas as pd


def validate_schema(df, expected_schema):

    results = {
        "missing_columns": [],
        "unexpected_columns": [],
        "type_mismatches": []
    }

    actual_columns = set(df.columns)
    expected_columns = set(expected_schema.keys())

    # Check missing columns
    results["missing_columns"] = list(
        expected_columns - actual_columns
    )

    # Check unexpected columns
    results["unexpected_columns"] = list(
        actual_columns - expected_columns
    )

    # Check data types
    for column, expected_type in expected_schema.items():

        if column not in df.columns:
            continue

        series = df[column].dropna()

        if expected_type == "integer":
            valid = (
                pd.api.types.is_numeric_dtype(series)
                and (series % 1 == 0).all()
            )

        elif expected_type == "float":
            valid = pd.api.types.is_numeric_dtype(series)

        elif expected_type == "string":
            valid = (
                pd.api.types.is_string_dtype(df[column])
                or df[column].dtype == "object"
            )

        else:
            raise ValueError(
                f"Unsupported expected type: {expected_type}"
            )

        if not valid:
            results["type_mismatches"].append({
                "column": column,
                "expected": expected_type,
                "actual": str(df[column].dtype)
            })

    results["is_valid"] = not any([
        results["missing_columns"],
        results["unexpected_columns"],
        results["type_mismatches"]
    ])

    return results
