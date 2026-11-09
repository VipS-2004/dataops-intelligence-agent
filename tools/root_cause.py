import json
from typing import Literal

from pydantic import BaseModel, Field

from langchain_core.messages import SystemMessage, HumanMessage
from tools.llm_client import get_llm


class RootCauseHypothesis(BaseModel):
    related_issues: list[str] = Field(
        description="Incident types related to this hypothesis"
    )

    likely_cause: str = Field(
        description="A plausible explanation for the observed issues"
    )

    supporting_evidence: list[str] = Field(
        description="Specific evidence from the supplied incidents"
    )

    confidence: Literal["low", "medium", "high"] = Field(
        description="Confidence based on the available evidence"
    )

    severity: Literal["low", "medium", "high", "critical"] = Field(
        description="Potential impact on data reliability"
    )

    recommended_checks: list[str] = Field(
        description="Checks an engineer should perform to verify this hypothesis"
    )


class RootCauseReport(BaseModel):
    overall_assessment: str = Field(
        description="Overall assessment of the dataset's reliability"
    )

    hypotheses: list[RootCauseHypothesis] = Field(
        description="Grouped root-cause hypotheses"
    )

    data_limitations: list[str] = Field(
        description="Limitations that affect the reliability of this investigation"
    )


def investigate_root_cause(incidents: list[dict], incident_summary: dict) -> dict:

    if not incidents:
        return {
            "overall_assessment": "No incidents detected.",
            "hypotheses": [],
            "data_limitations": []
        }

    llm = get_llm()

    structured_llm = llm.with_structured_output(
        RootCauseReport,
        method="function_calling"
    )

    system_prompt = """
    You are a senior data reliability engineer investigating data pipeline incidents.

    Analyze the supplied incident information and produce root-cause hypotheses.

    Rules:
    1. Use only the evidence supplied in the input.
    2. Never present a hypothesis as a confirmed cause.
    3. Group related incidents when they may share a cause.
    4. Do not count overlapping findings as independent problems.
    5. Explain which evidence supports each hypothesis.
    6. Consider missing values, invalid values, type mismatches,
       distribution drift, and statistical anomalies.
    7. Mention small sample sizes and data quality issues as limitations.
    8. Do not invent dataset details.
    9. Recommend investigation checks, not automatic data modifications.
    10. Use only these exact values: confidence must be low, medium, or high;
        severity must be low, medium, high, or critical. Do not combine labels.
    """

    human_prompt = f"""
    Investigate these DataOps findings.

    Incident summary:
    {json.dumps(incident_summary, indent=2)}

    Individual incidents:
    {json.dumps(incidents, indent=2)}

    Return your structured root-cause investigation.
    """

    result = structured_llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ])

    return result.model_dump()