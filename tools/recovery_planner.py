
import json
from pathlib import Path
from numbers import Integral
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage

from tools.llm_client import get_llm


class RecoveryAction(BaseModel):
    action_type: Literal[
        "coerce_numeric_with_quarantine",
        "review_missing_values",
        "review_duplicates",
        "review_business_rule_violations",
        "investigate_distribution_drift",
        "review_anomalous_records"
    ]

    target_column: str | None = None
    rationale: str
    proposed_steps: list[str]
    parameters: dict[str, Any] = Field(default_factory=dict)
    risk_level: Literal["low", "medium", "high", "critical"]
    reversible: bool
    validation_check: str


class RecoveryPlan(BaseModel):
    overall_strategy: str
    actions: list[RecoveryAction]
    manual_investigations: list[str]
    overall_risk: Literal["low", "medium", "high", "critical"]


def create_recovery_plan(
    root_cause_results: dict,
    incident_summary: dict,
    incidents: list[dict]
) -> dict:

    if not incidents:
        return {
            "overall_strategy": "No recovery required.",
            "actions": [],
            "manual_investigations": [],
            "overall_risk": "low",
            "execution_allowed": False
        }

    llm = get_llm()

    structured_llm = llm.with_structured_output(
        RecoveryPlan,
        method="function_calling"
    )

    system_prompt = """
    You are a senior DataOps recovery engineer.

    Create a structured recovery plan using the supplied evidence.

    Rules:
    1. Never invent replacement values.
    2. Never automatically delete duplicates.
    3. Never convert negative purchases to absolute values.
    4. Never automatically correct invalid ages.
    5. Preserve original values during numeric conversion.
    6. Investigate distribution drift without modifying distributions.
    7. Statistical anomalies are not necessarily errors.
    8. Every action requires human approval.
    9. Include a validation check for every action.
    10. Do not execute recovery operations.
    11. Use only these exact values for risk_level and overall_risk:
        low, medium, high, or critical. Do not combine labels.
    """

    human_prompt = f"""
    Root cause investigation:
    {json.dumps(root_cause_results, indent=2)}

    Incident summary:
    {json.dumps(incident_summary, indent=2)}

    Individual incidents:
    {json.dumps(incidents, indent=2)}

    Return a structured recovery plan.
    """

    result = structured_llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ])

    plan = result.model_dump()

    for action in plan["actions"]:
        action["requires_human_approval"] = True

    plan["execution_allowed"] = False

    return plan


def execute_recovery(
    dataset_path: str,
    recovery_plan: dict,
    approval_result: dict
) -> dict:

    if (
        not isinstance(approval_result, dict)
        or approval_result.get("decision") != "approve"
        or approval_result.get("approved") is not True
    ):
        raise PermissionError("Recovery not approved.")

    source_path = Path(dataset_path)
    output_path = Path("outputs/recovered_current.csv")

    if not source_path.exists():
        raise FileNotFoundError(source_path)

    if source_path.resolve() == output_path.resolve():
        raise ValueError("Cannot overwrite original dataset.")

    df = pd.read_csv(source_path)
    recovered = df.copy(deep=True)

    applied_actions = []
    manual_reviews = []
    corrected_values = {}
    flagged_values = {}

    supported_actions = {
        "coerce_numeric_with_quarantine",
        "review_missing_values",
        "review_duplicates",
        "review_business_rule_violations",
        "investigate_distribution_drift",
        "review_anomalous_records"
    }

    def action_columns(action, params):
        columns = params.get("columns")

        if columns is not None and not isinstance(columns, list):
            raise ValueError("Action columns must be a list.")

        if not columns:
            column = params.get("column") or action.get("target_column")
            columns = [column] if column is not None else []

        if any(not isinstance(column, str) for column in columns):
            raise ValueError("Action columns must contain column names.")

        columns = list(dict.fromkeys(columns))

        for column in columns:
            if column not in df.columns:
                raise ValueError(f"Unknown column: {column}")

        return columns

    def set_flag(name, values):
        if name in df.columns:
            raise ValueError(
                f"Cannot overwrite original column with recovery flag: {name}"
            )

        values = pd.Series(values, index=df.index, dtype=bool)
        if name in recovered.columns:
            recovered[name] = recovered[name].astype(bool) | values
        else:
            recovered[name] = values

    def add_flagged_value(name, count):
        flagged_values[name] = flagged_values.get(name, 0) + int(count)

    for action in recovery_plan.get("actions", []):

        if not isinstance(action, dict):
            raise ValueError("Recovery actions must be objects.")

        if action.get("requires_human_approval") is not True:
            raise PermissionError("Action lacks human approval.")

        action_type = action.get("action_type")
        if not isinstance(action_type, str) or action_type not in supported_actions:
            raise ValueError(f"Unsupported action: {action_type}")

        params = action.get("parameters") or {}
        if not isinstance(params, dict):
            raise ValueError("Action parameters must be an object.")

        columns = action_columns(action, params)

        if action_type == "coerce_numeric_with_quarantine":

            if not columns:
                raise ValueError("Numeric conversion requires at least one column.")

            for column in columns:
                original = df[column].copy()
                converted = pd.to_numeric(original, errors="coerce")
                original_column = f"{column}_original"
                quarantine_column = f"{column}_quarantined"

                if original_column in df.columns or quarantine_column in df.columns:
                    raise ValueError(
                        f"Recovery output column already exists for {column}."
                    )

                recovered[original_column] = original
                recovered[column] = converted
                invalid_values = original.notna() & converted.isna()
                set_flag(quarantine_column, invalid_values)
                corrected_values[column] = int(
                    (original.notna() & converted.notna()).sum()
                )
                add_flagged_value(
                    f"unconvertible_{column}_values", invalid_values.sum()
                )

        elif action_type == "review_missing_values":

            if not columns:
                columns = [
                    column for column in df.columns
                    if df[column].isna().any()
                ]

            missing_count = 0
            for col in columns:
                missing = df[col].isna()
                set_flag(f"recovery_missing_{col}", missing)
                add_flagged_value(f"missing_{col}", missing.sum())
                missing_count += int(missing.sum())

            if missing_count:
                manual_reviews.append(
                    "Review flagged missing values; no values were imputed "
                    "and no rows were deleted."
                )

        elif action_type == "review_duplicates":

            duplicate_records = df.duplicated(keep=False)
            set_flag("recovery_duplicate_flag", duplicate_records)
            add_flagged_value(
                "duplicate_record_rows_flagged", duplicate_records.sum()
            )

            duplicate_ids = pd.Series(False, index=df.index)
            if "transaction_id" in df.columns:
                duplicate_ids = df["transaction_id"].duplicated(keep=False)
                set_flag("recovery_duplicate_id_flag", duplicate_ids)
                add_flagged_value(
                    "duplicate_transaction_id_rows_flagged",
                    duplicate_ids.sum()
                )

            if duplicate_records.any() or duplicate_ids.any():
                manual_reviews.append(
                    "Review duplicate records and transaction IDs; "
                    "none were deleted."
                )

        elif action_type == "review_business_rule_violations":

            if not columns:
                columns = [
                    column for column in ("purchase_amount", "age")
                    if column in df.columns
                ]

            for column in columns:
                if column not in ("purchase_amount", "age"):
                    raise ValueError(
                        f"Unsupported business-rule column: {column}"
                    )

                values = pd.to_numeric(df[column], errors="coerce")
                if column == "purchase_amount":
                    violations = values < 0
                    flag_name = "recovery_negative_purchase_amount_flag"
                    issue = "negative_purchase_amounts"
                else:
                    violations = (values < 0) | (values > 120)
                    flag_name = "recovery_invalid_age_flag"
                    issue = "invalid_ages"

                set_flag(flag_name, violations)
                count = int(violations.sum())
                add_flagged_value(issue, count)
                if count:
                    manual_reviews.append(
                        f"Review {count} {issue.replace('_', ' ')}; "
                        "none were corrected."
                    )

        elif action_type == "investigate_distribution_drift":

            drift_columns = columns or ["unspecified columns"]
            for column in drift_columns:
                manual_reviews.append(
                    f"Distribution drift investigation required for {column}; "
                    "the distribution was not modified."
                )

        elif action_type == "review_anomalous_records":

            rows = params.get(
                "row_indices",
                params.get("rows", params.get("anomalous_rows", []))
            )
            if not isinstance(rows, list):
                raise ValueError("Anomaly row indices must be a list.")

            if any(
                not isinstance(row, Integral)
                or isinstance(row, bool)
                or row not in df.index
                for row in rows
            ):
                raise ValueError("Invalid anomaly row index.")

            anomaly_flags = df.index.isin(rows)
            set_flag("recovery_anomaly_flag", anomaly_flags)
            add_flagged_value("anomalous_records", anomaly_flags.sum())
            if anomaly_flags.any():
                manual_reviews.append(
                    f"Review {int(anomaly_flags.sum())} anomalous record "
                    "indices; none were deleted."
                )

        applied_actions.append(action_type)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    recovered.to_csv(output_path, index=False)

    return {
        "status": "completed",
        "recovered_file": str(output_path),
        "original_preserved": True,
        "actions_applied": len(applied_actions),
        "corrected_values": corrected_values,
        "flagged_values": flagged_values,
        "manual_reviews": manual_reviews
    }
