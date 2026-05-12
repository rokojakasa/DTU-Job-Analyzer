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
  - Coverage is plausible (not 0%, not 100% — the fixture is designed so
    the CV clearly covers some skills and clearly misses others)
  - The LLM doesn't hallucinate skills that aren't in the job posting
"""

from __future__ import annotations

import pytest
from app.models import AnalysisResponse, EvidenceType, SkillCategory
from app.services.analyser import run_analysis


# A job posting with clear required and bonus sections so we can reason
# about what a correct analysis should look like.
JOB_POSTING = """
We are looking for a Backend Engineer.

Required skills:
- Python (3.10+)
- PostgreSQL or another relational database
- REST API design
- Docker and containerisation
- Git version control

Bonus skills:
- Kubernetes
- Redis or another caching layer
- Experience with FastAPI
"""

# A CV that covers Python, PostgreSQL, REST API, and Docker — but NOT Git
# explicitly, and NOT the bonus skills. Gives us predictable gap assertions.
CV = """
Alice Smith — Backend Engineer

Experience:
- 2 years at Startup X: Built REST APIs in Python using FastAPI and SQLAlchemy.
  Deployed services in Docker containers on AWS.
  Managed a PostgreSQL database with ~50 tables; wrote migrations with Alembic.

Skills: Python, FastAPI, Docker, PostgreSQL, SQLAlchemy, REST API design

Education: BSc Computer Science, DTU 2022
"""

@pytest.mark.eval
async def test_run_analysis_returns_without_error():
    """The full two-pass pipeline completes without raising."""
    result = await run_analysis(CV, JOB_POSTING)
    assert result is not None


@pytest.mark.eval
async def test_returns_analysis_response_type():
    """run_analysis() should return a fully validated AnalysisResponse."""
    result = await run_analysis(CV, JOB_POSTING)
    assert isinstance(result, AnalysisResponse)


@pytest.mark.eval
async def test_job_title_is_non_empty_string():
    result = await run_analysis(CV, JOB_POSTING)
    assert isinstance(result.job_title_inferred, str)
    assert len(result.job_title_inferred) > 0


@pytest.mark.eval
async def test_covered_skills_and_gaps_are_non_empty():
    """With this CV and posting, both lists should be non-empty."""
    result = await run_analysis(CV, JOB_POSTING)
    assert len(result.covered_skills) > 0, "Expected at least one covered skill"
    assert len(result.skill_gaps) > 0, "Expected at least one skill gap"


@pytest.mark.eval
async def test_covered_skills_have_valid_evidence_types():
    result = await run_analysis(CV, JOB_POSTING)
    for skill in result.covered_skills:
        assert skill.evidence_type in EvidenceType, (
            f"Invalid evidence_type: {skill.evidence_type!r} for skill {skill.label!r}"
        )


@pytest.mark.eval
async def test_all_skills_have_valid_categories():
    result = await run_analysis(CV, JOB_POSTING)
    for skill in result.covered_skills:
        assert skill.category in SkillCategory, (
            f"Invalid category: {skill.category!r} for skill {skill.label!r}"
        )
    for gap in result.skill_gaps:
        assert gap.category in SkillCategory, (
            f"Invalid category: {gap.category!r} for gap {gap.label!r}"
        )


@pytest.mark.eval
async def test_covered_skills_have_non_empty_evidence():
    result = await run_analysis(CV, JOB_POSTING)
    for skill in result.covered_skills:
        assert skill.evidence_from_cv, (
            f"Covered skill {skill.label!r} has empty evidence_from_cv"
        )


@pytest.mark.eval
async def test_all_skills_have_non_empty_labels():
    result = await run_analysis(CV, JOB_POSTING)
    for skill in result.covered_skills:
        assert skill.label, "covered skill has empty label"
        assert skill.job_posting_label, "covered skill has empty job_posting_label"
    for gap in result.skill_gaps:
        assert gap.label, "skill gap has empty label"
        assert gap.job_posting_label, "skill gap has empty job_posting_label"


@pytest.mark.eval
async def test_python_is_covered():
    """Python is explicitly in the CV and posting — must be covered."""
    result = await run_analysis(CV, JOB_POSTING)
    covered_labels = [skill.label.lower() for skill in result.covered_skills]
    assert any("python" in label for label in covered_labels), (
        f"Expected Python to be a covered skill. Got: {covered_labels}"
    )


@pytest.mark.eval
async def test_kubernetes_is_not_a_required_gap():
    """Kubernetes is bonus-only — it must not appear as a required gap."""
    result = await run_analysis(CV, JOB_POSTING)
    required_gap_labels = [
        gap.label.lower()
        for gap in result.skill_gaps
        if gap.category == SkillCategory.REQUIRED
    ]
    assert not any("kubernetes" in label for label in required_gap_labels), (
        f"Kubernetes is bonus, not required. Required gaps: {required_gap_labels}"
    )


@pytest.mark.eval
async def test_no_hallucinated_skills():
    """
    Every returned skill label should relate to something actually in the
    job posting. We check this loosely: at least 3 of the posting's key
    terms must appear somewhere across all skill labels.

    This is a smoke test — it catches gross failures like the LLM inventing
    an entirely unrelated skill set, not subtle misclassifications.
    """
    posting_keywords = {
        "python", "postgresql", "rest", "api", "docker", "git",
        "kubernetes", "redis", "fastapi", "database", "container",
    }
    result = await run_analysis(CV, JOB_POSTING)
    all_labels = " ".join(
        skill.label.lower() for skill in result.covered_skills + result.skill_gaps
    )
    matches = [kw for kw in posting_keywords if kw in all_labels]
    assert len(matches) >= 3, (
        f"Too few posting keywords found in skill labels. "
        f"Matches: {matches}. All labels: {all_labels}"
    )


@pytest.mark.eval
async def test_summary_is_non_empty_string():
    result = await run_analysis(CV, JOB_POSTING)
    assert isinstance(result.summary, str)
    assert len(result.summary) > 20, "Summary seems too short to be meaningful"