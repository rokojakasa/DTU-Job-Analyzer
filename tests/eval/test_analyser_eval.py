# tests/eval/test_analyser_eval.py
"""
Evaluation tests for the LLM-backed analysis pipeline.

These tests call the real CampusAI API and assert on the *shape and
quality* of the output — not exact values, since LLMs are non-deterministic.

Run with:  pytest --run-eval
Skipped by default in CI.

What we're checking:
  - The LLM returns well-formed output (no JSON parse errors, Pydantic validates)
  - Skills are categorised into valid enum values
  - Coverage is plausible given the CV and job posting
  - Clear required gaps are detected (cloud deployment)
  - Clear covered skills are detected (Python, ML, testing)
  - Bonus gaps are correctly categorised as bonus, not required
"""

from __future__ import annotations

import pytest
from app.models import AnalysisResponse, EvidenceType, SkillCategory
from app.services.analyser import run_analysis


JOB_POSTING = """
Student Assistant — Advanced Analytics
Ørsted, Copenhagen

You'll be part of Advanced Analytics where you, together with your colleagues,
will develop forecasts, analyses, and dashboards which provide our trading floor
with a strong basis for data and statistical driven decision making in the daily
operation at our Danish power stations, wind turbines, solar PV, electric boilers,
heat pumps, and P2X.

You'll play an important role in:
- Developing and implementing performance tracking
- Developing dashboards to visualise decision support and performance tracking
- Developing and implementing forecasting models using AI/ML/DL, including
  stochastic modelling and statistical data analysis
- Communicating and collaborating with internal and external stakeholders,
  i.e., team members, traders, etc.

To succeed in the role, you:
- Are currently studying data science, applied mathematics, or another relevant
  field within analytics and/or engineering
- Are a strong Python programmer
- Have experience with writing and reviewing maintainable and testable code
- Are familiar with machine-learning concepts (supervised and unsupervised learning)
  and data analytics
- Have experience with cloud platforms (Azure or AWS) for deploying data pipelines

Bonus:
- Knowledge of Git, Docker & Kubernetes
- Familiarity with electricity markets or energy systems
- Experience with time series forecasting or stochastic modelling
"""

CV = """
Mikkel Andersen
MSc student, Mathematical Modelling and Computation, DTU
mikkel.andersen@student.dtu.dk | github.com/mikkelands

EDUCATION
Technical University of Denmark (DTU), 2022–present
MSc Mathematical Modelling and Computation (expected graduation 2025)

Relevant courses:
- 02450 Introduction to Machine Learning (grade: 10)
- 02457 Non-Linear Signal Processing (grade: 12)
- 02418 Statistical Modelling (grade: 10)
- 02613 High-Performance Computing (grade: 7)
- 02807 Computational Tools for Data Science (grade: 10)

BSc Mathematics and Technology, DTU, 2019–2022 (grade average: 9.2)

EXPERIENCE

Student Assistant — Data & AI, Vestas Wind Systems (Jun 2023–present)
Working in a small analytics team supporting wind turbine performance monitoring.
- Built and maintained Python pipelines ingesting SCADA telemetry data from 200+
  turbines into an internal data warehouse.
- Developed a regression-based power curve model using scikit-learn to flag
  underperforming turbines; reduced manual inspection workload by ~30%.
- Wrote unit tests with pytest; all code reviewed via pull requests on GitHub.
- Delivered a Jupyter-based dashboard prototype to operations team using Plotly.

Project: Probabilistic Load Forecasting (02450 course project, 2023)
- Trained LSTM and gradient boosting models on hourly Danish electricity consumption
  data from Energinet.
- Compared point forecasts vs. quantile regression for uncertainty estimation.
- Implemented cross-validation pipeline in Python (pandas, scikit-learn, PyTorch).

Project: Monte Carlo Methods for Option Pricing (02457, 2022)
- Implemented stochastic differential equation solvers (Euler-Maruyama) in Python
  and NumPy to price European and Asian options.
- Benchmarked convergence rates across variance reduction techniques.

SKILLS
Languages: Python (primary), R (familiar), MATLAB (coursework)
Libraries: scikit-learn, PyTorch, pandas, NumPy, Plotly, pytest
Tools: Git, Linux, SQL (PostgreSQL)
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def covered_labels(result: AnalysisResponse) -> list[str]:
    return [s.label.lower() for s in result.covered_skills]

def gap_labels(result: AnalysisResponse) -> list[str]:
    return [g.label.lower() for g in result.skill_gaps]

def required_gap_labels(result: AnalysisResponse) -> list[str]:
    return [
        g.label.lower() for g in result.skill_gaps
        if g.category == SkillCategory.REQUIRED
    ]

def bonus_gap_labels(result: AnalysisResponse) -> list[str]:
    return [
        g.label.lower() for g in result.skill_gaps
        if g.category == SkillCategory.BONUS
    ]


# ---------------------------------------------------------------------------
# Shape and type tests — always pass if the pipeline is working at all
# ---------------------------------------------------------------------------

@pytest.mark.eval
async def test_returns_analysis_response_type():
    """run_analysis() returns a fully validated AnalysisResponse."""
    result = await run_analysis(CV, JOB_POSTING)
    assert isinstance(result, AnalysisResponse)


@pytest.mark.eval
async def test_job_title_is_non_empty():
    result = await run_analysis(CV, JOB_POSTING)
    assert result.job_title_inferred.strip() != ""


@pytest.mark.eval
async def test_covered_skills_and_gaps_are_non_empty():
    """The CV clearly covers some skills and clearly misses others."""
    result = await run_analysis(CV, JOB_POSTING)
    assert len(result.covered_skills) > 0
    assert len(result.skill_gaps) > 0


@pytest.mark.eval
async def test_all_evidence_types_are_valid():
    result = await run_analysis(CV, JOB_POSTING)
    valid = {e.value for e in EvidenceType}
    for skill in result.covered_skills:
        assert skill.evidence_type.value in valid, (
            f"Invalid evidence_type {skill.evidence_type!r} for skill {skill.label!r}"
        )


@pytest.mark.eval
async def test_all_categories_are_valid():
    result = await run_analysis(CV, JOB_POSTING)
    valid = {c.value for c in SkillCategory}
    for skill in result.covered_skills:
        assert skill.category.value in valid
    for gap in result.skill_gaps:
        assert gap.category.value in valid


@pytest.mark.eval
async def test_summary_is_non_empty():
    result = await run_analysis(CV, JOB_POSTING)
    assert result.summary.strip() != ""


# ---------------------------------------------------------------------------
# Coverage quality — does the LLM correctly identify clear signals?
# ---------------------------------------------------------------------------

@pytest.mark.eval
async def test_python_is_covered():
    """Python is explicit throughout the CV — must be covered."""
    result = await run_analysis(CV, JOB_POSTING)
    assert any("python" in label for label in covered_labels(result)), (
        f"Expected Python to be covered. Covered: {covered_labels(result)}"
    )


@pytest.mark.eval
async def test_machine_learning_is_covered():
    """
    ML is covered by 02450, scikit-learn at Vestas, and the forecasting
    project — the LLM should find this even without an exact label match.
    """
    result = await run_analysis(CV, JOB_POSTING)
    ml_keywords = {"machine learning", "supervised", "ml", "forecasting model"}
    assert any(
        any(kw in label for kw in ml_keywords)
        for label in covered_labels(result)
    ), f"Expected ML to be covered. Covered: {covered_labels(result)}"


@pytest.mark.eval
async def test_testable_code_is_covered():
    """
    CV explicitly mentions pytest and pull request reviews — the LLM should
    map this to 'maintainable and testable code'.
    """
    result = await run_analysis(CV, JOB_POSTING)
    testing_keywords = {"test", "maintainable", "code quality", "review"}
    assert any(
        any(kw in label for kw in testing_keywords)
        for label in covered_labels(result)
    ), f"Expected testable code to be covered. Covered: {covered_labels(result)}"


@pytest.mark.eval
async def test_cloud_is_a_required_gap():
    """
    Cloud deployment is required and absent from the CV entirely.
    This is the clearest gap — the LLM must not hallucinate coverage.
    """
    result = await run_analysis(CV, JOB_POSTING)
    cloud_keywords = {"cloud", "azure", "aws", "deploy"}
    assert any(
        any(kw in label for kw in cloud_keywords)
        for label in required_gap_labels(result)
    ), (
        f"Expected cloud deployment to be a required gap.\n"
        f"Required gaps: {required_gap_labels(result)}\n"
        f"Covered: {covered_labels(result)}"
    )


@pytest.mark.eval
async def test_cloud_is_not_hallucinated_as_covered():
    """
    Mirror of the above — cloud must not appear in covered_skills.
    Guards against the LLM confusing HPC course with cloud deployment.
    """
    result = await run_analysis(CV, JOB_POSTING)
    cloud_keywords = {"cloud", "azure", "aws"}
    assert not any(
        any(kw in label for kw in cloud_keywords)
        for label in covered_labels(result)
    ), (
        f"Cloud should not be covered — HPC course is not cloud deployment.\n"
        f"Covered: {covered_labels(result)}"
    )


@pytest.mark.eval
async def test_docker_kubernetes_is_a_bonus_gap():
    """
    Docker/Kubernetes is listed as bonus and absent from the CV.
    Must appear as a gap with category=bonus, not required.
    """
    result = await run_analysis(CV, JOB_POSTING)
    container_keywords = {"docker", "kubernetes", "container"}
    assert any(
        any(kw in label for kw in container_keywords)
        for label in bonus_gap_labels(result)
    ), (
        f"Expected Docker/Kubernetes as a bonus gap.\n"
        f"Bonus gaps: {bonus_gap_labels(result)}"
    )


@pytest.mark.eval
async def test_no_hallucinated_skills():
    """
    Every covered skill must reference something that exists in the CV.
    We check this loosely — evidence_from_cv must be non-empty.
    """
    result = await run_analysis(CV, JOB_POSTING)
    for skill in result.covered_skills:
        assert skill.evidence_from_cv.strip() != "", (
            f"Skill {skill.label!r} has empty evidence_from_cv — possible hallucination."
        )