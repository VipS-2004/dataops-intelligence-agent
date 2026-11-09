# DataOps Intelligence Agent

An LLM-powered DataOps agent built with Python and LangGraph that detects data quality issues, investigates potential root causes, proposes recovery actions, and executes approved recovery operations with human oversight.

## Overview

Traditional data validation scripts can identify problems, but they do not investigate why those problems occurred or coordinate a recovery process.

DataOps Intelligence Agent combines machine learning, statistical analysis, LLM reasoning, and LangGraph orchestration to create an intelligent data reliability workflow.

## Architecture

```text
              CSV / Excel Dataset
                      |
                Data Ingestion
                      |
          +-----------+-----------+
          |           |           |
       Schema      Quality      Drift
     Validation   Analysis    Detection
          |           |           |
          +-----------+-----------+
                      |
              Anomaly Detection
                      |
              Incident Aggregation
                      |
              Conditional Routing
                      |
            LLM Root Cause Analysis
                      |
              Recovery Planning
                      |
               Human Approval
                      |
              Recovery Execution
                      |
            Post-Recovery Validation
                      |
                Final Report
```

## Features

### 1. Data Analysis
- CSV and Excel ingestion.
- Schema validation and type mismatch detection.
- Missing values and duplicate detection.
- Invalid business-rule value detection.
- Population Stability Index (PSI).
- Empirical CDF distance for distribution comparison.
- Isolation Forest anomaly detection.

### 2. LangGraph Orchestration
- Parallel execution of independent analysis nodes.
- Shared graph state.
- Incident aggregation.
- Conditional routing based on detected issues.

### 3. LLM Root Cause Investigation
- Structured investigation using Pydantic.
- Evidence-based root-cause hypotheses.
- Confidence and severity assessments.
- Recommended investigation checks.

### 4. Recovery Planning
- LLM-generated recovery recommendations.
- Risk assessment.
- Proposed recovery steps.
- Validation criteria for proposed actions.

### 5. Human-in-the-Loop
- Approval checkpoint using LangGraph `interrupt()`.
- Workflow resumption using `Command(resume=...)`.
- Checkpointing using `InMemorySaver`.

### 6. Safe Recovery
- Preserve the original dataset.
- Convert valid numeric values while preserving originals.
- Quarantine invalid numeric values.
- Flag missing values, duplicates and anomalous records.
- Preserve business-rule violations for manual review.
- Avoid automatic deletion or imputation.

### 7. Post-Recovery Validation
- Re-run schema checks.
- Re-run data quality checks.
- Report remaining issues and manual review requirements.

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| LLM Provider | OpenRouter |
| LLM Integration | LangChain |
| Agent Orchestration | LangGraph |
| Structured Output | Pydantic |
| Environment Management | uv |

## Project Structure

```text
dataops-intelligence-agent/
├── app.py
├── data/
│   ├── current.csv
│   └── baseline.csv
├── graph/
│   ├── state.py
│   ├── nodes.py
│   ├── workflow.py
│   └── routing.py
├── tools/
│   ├── ingestion.py
│   ├── schema_validation.py
│   ├── data_quality.py
│   ├── drift_detection.py
│   ├── anomaly_detection.py
│   ├── llm_client.py
│   ├── root_cause.py
│   └── recovery_planner.py
├── tests/
├── outputs/
├── pyproject.toml
├── uv.lock
└── README.md
```

## Installation

Clone the repository:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd dataops-intelligence-agent
```

Install dependencies using uv:

```bash
uv sync
```

## Configuration

Create a `.env` file in the project root:

```env
OPENROUTER_API_KEY=your_api_key
OPENROUTER_MODEL=your_model_identifier
```

Never commit your `.env` file or expose your API key.

## Run the Project

```bash
uv run app.py
```

The application executes the analysis workflow, generates an investigation and recovery plan, and requests human approval before recovery execution.

Enter `yes` to approve or `no` to reject the recovery plan.

## Recovery Output

Approved recovery operations generate:

```text
outputs/
└── recovered_current.csv
```

The original dataset remains unchanged.

## Current Limitations

- Small datasets make statistical drift and anomaly results exploratory.
- Root-cause findings are hypotheses, not confirmed explanations.
- Some data issues require manual investigation.
- LangGraph uses in-memory checkpointing, so interrupted sessions are not persistent across application restarts.
- Post-recovery validation can still report unresolved quality issues.

## Project Goal

Demonstrate how traditional data analysis and machine learning can be combined with LLM reasoning, LangGraph orchestration, human approval, and controlled recovery to build an intelligent DataOps system.
