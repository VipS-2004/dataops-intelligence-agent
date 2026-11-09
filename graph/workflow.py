
from langgraph.graph import StateGraph, START, END

from graph.state import DataOpsState
from langgraph.checkpoint.memory import InMemorySaver
from graph.routing import route_after_aggregation

from graph.nodes import (
    schema_analysis_node,
    quality_analysis_node,
    drift_analysis_node,
    anomaly_analysis_node,
    aggregate_findings_node,
    root_cause_node,
    healthy_report_node,
    recovery_planning_node,
    human_approval_node,
    recovery_execution_node,
    post_recovery_validation_node
)

from graph.routing import (
    route_after_aggregation,
    route_after_approval
)

def build_workflow():

    builder = StateGraph(DataOpsState)

    # Register nodes
    builder.add_node("schema_analysis", schema_analysis_node)
    builder.add_node("quality_analysis", quality_analysis_node)
    builder.add_node("drift_analysis", drift_analysis_node)
    builder.add_node("anomaly_analysis", anomaly_analysis_node)
    builder.add_node("aggregate_findings", aggregate_findings_node)
    builder.add_node("root_cause", root_cause_node)
    builder.add_node("healthy_report", healthy_report_node)
    builder.add_node("recovery_planning", recovery_planning_node)
    builder.add_node("human_approval", human_approval_node)
    builder.add_node("recovery_execution", recovery_execution_node)
    builder.add_node(
        "post_recovery_validation",
        post_recovery_validation_node
    )

    # Entry point: start all four analysis branches
    builder.add_edge(START, "schema_analysis")
    builder.add_edge(START, "quality_analysis")
    builder.add_edge(START, "drift_analysis")
    builder.add_edge(START, "anomaly_analysis")

    # Synchronize all four branches
    builder.add_edge(
        [
            "schema_analysis",
            "quality_analysis",
            "drift_analysis",
            "anomaly_analysis"
        ],
        "aggregate_findings"
    )

    # Finish after aggregation
    builder.add_conditional_edges(
    "aggregate_findings",
    route_after_aggregation,
    {
        "investigate": "root_cause",
        "healthy": "healthy_report"
    }
    )



    builder.add_edge("root_cause", "recovery_planning")
    builder.add_edge("recovery_planning", "human_approval")
    builder.add_conditional_edges(
    "human_approval",
    route_after_approval,
    {
        "execute": "recovery_execution",
        "stop": END
    }
)

    builder.add_edge("recovery_execution", "post_recovery_validation")
    builder.add_edge("post_recovery_validation", END)

    return builder.compile(
    checkpointer=InMemorySaver()
)
