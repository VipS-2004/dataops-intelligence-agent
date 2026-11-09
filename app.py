
from graph.workflow import build_workflow
import json
import uuid

from langgraph.types import Command
from tools.ingestion import load_dataset


# Define workflow inputs
initial_state = {
    "dataset_path": "data/current.csv",
    "baseline_path": "data/baseline.csv",

    "expected_schema": {
        "customer_id": "string",
        "age": "integer",
        "income": "float",
        "purchase_amount": "float",
        "transaction_id": "string",
        "region": "string",
        "transaction_date": "string"
    },

    "incidents": []
}


# Build and compile graph
# Build workflow
workflow = build_workflow()

# Use the same thread ID when resuming
config = {
    "configurable": {
        "thread_id": str(uuid.uuid4())
    }
}

print("Starting DataOps Intelligence Agent...\n")

_, original_metadata = load_dataset(initial_state["dataset_path"])

# First execution
result = workflow.invoke(initial_state, config=config)

# Check whether graph is waiting for approval
interrupts = result.get("__interrupt__", [])

if interrupts:

    approval_data = interrupts[0].value

    print("\n========== HUMAN APPROVAL REQUIRED ==========")
    print(json.dumps(approval_data, indent=4))

    while True:

        answer = input(
            "\nApprove the complete recovery plan? (yes/no): "
        ).strip().lower()

        if answer in ("yes", "y"):
            decision = "approve"
            break

        elif answer in ("no", "n"):
            decision = "reject"
            break

        print("Please enter yes or no.")

    # Resume the paused workflow
    result = workflow.invoke(
        Command(
            resume={
                "decision": decision
            }
        ),
        config=config
    )


def print_report_section(title, value):
    print(f"\n========== {title} ==========")
    print(json.dumps(value, indent=2, default=str))


recovery_results = result.get("recovery_results", {})
validation = result.get("post_recovery_validation", {})
manual_reviews = list(recovery_results.get("manual_reviews", []))
manual_reviews.extend(validation.get("manual_review_issues", []))

print_report_section("FINAL DATAOPS REPORT", {
    "original_dataset": {
        "path": initial_state["dataset_path"],
        **original_metadata
    },
    "detected_incidents": result.get("incidents", []),
    "root_cause_investigation": result.get("root_cause_results", {}),
    "proposed_recovery_plan": result.get("recovery_plan", {}),
    "human_approval_decision": result.get("approval_result", {}),
    "recovery_execution_status": recovery_results.get(
        "status",
        "not executed"
    ),
    "recovered_dataset_path": recovery_results.get("recovered_file"),
    "corrected_values": recovery_results.get("corrected_values", {}),
    "flagged_values": recovery_results.get("flagged_values", {}),
    "post_recovery_schema_results": validation.get("schema_results"),
    "recovery_metadata_columns": validation.get(
        "recovery_metadata_columns",
        []
    ),
    "post_recovery_quality_results": validation.get("quality_results"),
    "issues_requiring_manual_review": manual_reviews
})