
from typing import Literal

from graph.state import DataOpsState


def route_after_aggregation(
    state: DataOpsState
) -> Literal["investigate", "healthy"]:

    if state.get("has_incidents", False):
        return "investigate"

    return "healthy"

def route_after_approval(state: DataOpsState):

    approval = state.get("approval_result", {})

    if (
        approval.get("decision") == "approve"
        and approval.get("approved") is True
    ):
        return "execute"

    return "stop"