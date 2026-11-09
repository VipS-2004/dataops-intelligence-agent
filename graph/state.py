
from typing import Annotated, TypedDict
import operator


class DataOpsState(TypedDict, total=False):

    # Input information
    dataset_path: str
    baseline_path: str
    expected_schema: dict[str, str]

    # Analysis results
    schema_results: dict
    quality_results: dict
    drift_results: dict
    anomaly_results: dict

    # Combined findings from parallel nodes
    incidents: Annotated[list[dict], operator.add]

    incident_summary: dict
    has_incidents: bool

    root_cause_results: dict

    recovery_plan: dict

    approval_result: dict
    recovery_results: dict
    post_recovery_validation: dict