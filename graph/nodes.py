
from graph.state import DataOpsState

from tools.ingestion import load_dataset
from tools.schema_validation import validate_schema
from tools.data_quality import analyze_data_quality
from tools.drift_detection import detect_drift
from tools.anomaly_detection import detect_anomalies
from tools.root_cause import investigate_root_cause
from tools.recovery_planner import create_recovery_plan
from langgraph.types import interrupt
from tools.recovery_planner import execute_recovery

from collections import Counter

# Load the current dataset
def load_current_dataset(state):

    df, metadata = load_dataset(
        state["dataset_path"]
    )

    return df


# 1. Schema analysis node
def schema_analysis_node(state: DataOpsState):

    df = load_current_dataset(state)

    results = validate_schema(
        df,
        state["expected_schema"]
    )

    incidents = []

    for column in results["missing_columns"]:
        incidents.append({
            "source": "schema",
            "issue": "missing_column",
            "column": column
        })

    for column in results["unexpected_columns"]:
        incidents.append({
            "source": "schema",
            "issue": "unexpected_column",
            "column": column
        })

    for mismatch in results["type_mismatches"]:
        incidents.append({
            "source": "schema",
            "issue": "type_mismatch",
            **mismatch
        })

    return {
        "schema_results": results,
        "incidents": incidents
    }


# 2. Data quality analysis node
def quality_analysis_node(state: DataOpsState):

    df = load_current_dataset(state)

    results = analyze_data_quality(df)

    incidents = []

    for column, count in results["missing_values"].items():
        incidents.append({
            "source": "data_quality",
            "issue": "missing_values",
            "column": column,
            "count": count
        })

    if results["duplicate_records"] > 0:
        incidents.append({
            "source": "data_quality",
            "issue": "duplicate_records",
            "count": results["duplicate_records"]
        })

    if results["duplicate_transaction_ids"] > 0:
        incidents.append({
            "source": "data_quality",
            "issue": "duplicate_transaction_ids",
            "count": results["duplicate_transaction_ids"]
        })

    for issue, count in results["invalid_values"].items():
        if count > 0:
            incidents.append({
                "source": "data_quality",
                "issue": issue,
                "count": count
            })

    for issue, count in results["range_violations"].items():
        if count > 0:
            incidents.append({
                "source": "data_quality",
                "issue": issue,
                "count": count
            })

    return {
        "quality_results": results,
        "incidents": incidents
    }


# 3. Drift detection node
def drift_analysis_node(state: DataOpsState):

    current_df = load_current_dataset(state)

    baseline_df, _ = load_dataset(
        state["baseline_path"]
    )

    results = detect_drift(
        baseline_df,
        current_df
    )

    incidents = []

    for column in results["drifted_columns"]:

        metrics = results["columns"][column]

        incidents.append({
            "source": "drift_detection",
            "issue": "distribution_drift",
            "column": column,
            "psi": metrics["psi"],
            "ecdf_distance": metrics["ecdf_distance"]
        })

    return {
        "drift_results": results,
        "incidents": incidents
    }


# 4. Anomaly detection node
def anomaly_analysis_node(state: DataOpsState):

    df = load_current_dataset(state)

    results = detect_anomalies(df)

    incidents = []

    for anomaly in results["anomalies"]:

        incidents.append({
            "source": "anomaly_detection",
            "issue": "unusual_record",
            "row_index": anomaly["row_index"],
            "anomaly_score": anomaly["anomaly_score"]
        })

    return {
        "anomaly_results": results,
        "incidents": incidents
    }





def aggregate_findings_node(state: DataOpsState):

    incidents = state.get("incidents", [])

    # Count findings by analysis source
    source_counts = Counter(
        incident["source"]
        for incident in incidents
    )

    # Count findings by issue type
    issue_counts = Counter(
        incident["issue"]
        for incident in incidents
    )

    # Identify affected columns
    affected_columns = sorted({
        incident["column"]
        for incident in incidents
        if "column" in incident
    })

    # Identify affected rows
    affected_rows = sorted({
        incident["row_index"]
        for incident in incidents
        if "row_index" in incident
    })

    summary = {
        "total_incidents": len(incidents),
        "source_counts": dict(source_counts),
        "issue_counts": dict(issue_counts),
        "affected_columns": affected_columns,
        "affected_rows": affected_rows
    }

    return {
        "incident_summary": summary,
        "has_incidents": len(incidents) > 0
    }


def root_cause_node(state: DataOpsState):

    print("\nRunning LLM root-cause investigation...")

    results = investigate_root_cause(
        incidents=state.get("incidents", []),
        incident_summary=state.get("incident_summary", {})
    )

    return {
        "root_cause_results": results
    }

def healthy_report_node(state: DataOpsState):
    print("\nHealthy dataset branch selected.")

    return {
        "root_cause_results": {
            "status": "not_required",
            "incident_count": 0
        }
    }


def recovery_planning_node(state: DataOpsState):

    print("\nGenerating recovery plan...")

    plan = create_recovery_plan(
        root_cause_results=state.get("root_cause_results", {}),
        incident_summary=state.get("incident_summary", {}),
        incidents=state.get("incidents", [])
    )

    return {
        "recovery_plan": plan
    }

def human_approval_node(state: DataOpsState):

    plan = state.get("recovery_plan", {})

    if not plan.get("actions"):
        return {
            "approval_result": {
                "decision": "not_required",
                "approved": False,
                "message": "No recovery actions available."
            }
        }

    print("\nWaiting for human approval...")

    approval = interrupt({
        "type": "recovery_plan_approval",
        "message": "Review the recovery plan and approve or reject it.",
        "overall_strategy": plan.get("overall_strategy"),
        "overall_risk": plan.get("overall_risk"),
        "actions": plan.get("actions", [])
    })

    if not isinstance(approval, dict):
        raise ValueError("Invalid approval response.")

    decision = approval.get("decision")

    if decision not in ("approve", "reject"):
        raise ValueError("Decision must be approve or reject.")

    return {
        "approval_result": {
            "decision": decision,
            "approved": decision == "approve",
            "reviewer_note": approval.get("note", "")
        }
    }


def recovery_execution_node(state: DataOpsState):

    print("\nExecuting approved recovery actions...")

    results = execute_recovery(
        dataset_path=state["dataset_path"],
        recovery_plan=state["recovery_plan"],
        approval_result=state.get("approval_result", {})
    )

    return {
        "recovery_results": results
    }


def post_recovery_validation_node(state: DataOpsState):

    recovery_results = state.get("recovery_results", {})
    recovered_path = recovery_results.get("recovered_file")
    if not recovered_path:
        raise ValueError("Recovery output path is unavailable.")

    recovered_df, _ = load_dataset(recovered_path)
    schema_results = validate_schema(
        recovered_df,
        state["expected_schema"]
    )
    source_columns = [
        column for column in state["expected_schema"]
        if column in recovered_df.columns
    ]
    quality_results = analyze_data_quality(
        recovered_df[source_columns]
    )

    manual_review_issues = []

    for column in schema_results["missing_columns"]:
        manual_review_issues.append(f"Missing expected column: {column}")
    recovery_metadata_columns = [
        column for column in schema_results["unexpected_columns"]
        if (
            column.startswith("recovery_")
            or column.endswith("_original")
            or column.endswith("_quarantined")
        )
    ]
    for column in schema_results["unexpected_columns"]:
        if column in recovery_metadata_columns:
            continue
        manual_review_issues.append(
            f"Unexpected column requires schema review: {column}"
        )
    for mismatch in schema_results["type_mismatches"]:
        manual_review_issues.append(
            f"Type mismatch in {mismatch['column']}: "
            f"expected {mismatch['expected']}, found {mismatch['actual']}"
        )
    for column, count in quality_results["missing_values"].items():
        manual_review_issues.append(
            f"{count} missing values remain in {column}."
        )
    if quality_results["duplicate_records"]:
        manual_review_issues.append(
            f"{quality_results['duplicate_records']} duplicate records remain."
        )
    if quality_results["duplicate_transaction_ids"]:
        manual_review_issues.append(
            f"{quality_results['duplicate_transaction_ids']} duplicate "
            "transaction IDs remain."
        )
    for issue, count in quality_results["invalid_values"].items():
        if count:
            manual_review_issues.append(f"{count} {issue.replace('_', ' ')} remain.")
    for issue, count in quality_results["range_violations"].items():
        if count:
            manual_review_issues.append(f"{count} {issue.replace('_', ' ')} remain.")

    return {
        "post_recovery_validation": {
            "schema_results": schema_results,
            "quality_results": quality_results,
            "recovery_metadata_columns": recovery_metadata_columns,
            "manual_review_issues": manual_review_issues
        }
    }