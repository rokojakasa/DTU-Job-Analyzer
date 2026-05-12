# tests/eval/test_questions_eval.py
"""
Evaluation tests for the question generator.

We inject a fixed AnalysisResponse so the only variable is the LLM's
question generation — not the analysis itself.
"""

from __future__ import annotations

import pytest
from app.models import AnalysisResponse, CoveredSkill, SkillGap, EvidenceType, SkillCategory
from app.services.question_generator import generate_questions


CV = """
Alice Smith — Backend Engineer

Experience:
- 2 years at Startup X: Built REST APIs in Python using FastAPI and SQLAlchemy.
  Deployed services in Docker containers on AWS.
  Managed a PostgreSQL database with ~50 tables; wrote migrations with Alembic.

Skills: Python, FastAPI, Docker, PostgreSQL, SQLAlchemy, REST API design
"""

# Fixed analysis so we control the input precisely.
# Using typed Pydantic models rather than raw dicts — if a field name or
# enum value changes, this fixture breaks at construction time, not inside a test.
FIXED_ANALYSIS = AnalysisResponse(
    job_title_inferred="Backend Engineer",
    covered_skills=[
        CoveredSkill(
            label="develop REST APIs",
            job_posting_label="REST API design",
            evidence_from_cv="Built REST APIs in Python using FastAPI.",
            evidence_type=EvidenceType.EXPLICIT,
            category=SkillCategory.REQUIRED,
        ),
        CoveredSkill(
            label="manage relational databases",
            job_posting_label="PostgreSQL",
            evidence_from_cv="Managed a PostgreSQL database with ~50 tables.",
            evidence_type=EvidenceType.EXPLICIT,
            category=SkillCategory.REQUIRED,
        ),
        CoveredSkill(
            label="containerise applications",
            job_posting_label="Docker",
            evidence_from_cv="Deployed services in Docker containers.",
            evidence_type=EvidenceType.EXPLICIT,
            category=SkillCategory.REQUIRED,
        ),
    ],
    skill_gaps=[
        SkillGap(
            label="use version control systems",
            job_posting_label="Git version control",
            category=SkillCategory.REQUIRED,
        ),
        SkillGap(
            label="orchestrate containers",
            job_posting_label="Kubernetes",
            category=SkillCategory.BONUS,
        ),
    ],
    summary="Strong backend profile, minor gap in Git evidence.",
)


@pytest.mark.eval
async def test_questions_returns_list():
    result = await generate_questions(CV, FIXED_ANALYSIS)
    assert isinstance(result, list)


@pytest.mark.eval
async def test_question_count_in_range():
    """Prompt asks for 4–6 covered + 1–2 gap questions = 5–8 total."""
    result = await generate_questions(CV, FIXED_ANALYSIS)
    assert 3 <= len(result) <= 10, (
        f"Expected 3–10 questions, got {len(result)}"
    )


@pytest.mark.eval
async def test_each_question_has_required_fields():
    result = await generate_questions(CV, FIXED_ANALYSIS)
    for q in result:
        assert q.skill, "skill field should be non-empty"
        assert q.question, "question field should be non-empty"
        assert q.follow_up, "follow_up field should be non-empty"
        # cv_anchor is Optional — allowed to be None for gap questions


@pytest.mark.eval
async def test_questions_are_not_too_long():
    """
    A question over 400 chars usually means the LLM ignored the
    'one sentence' instruction and blew up the prompt.
    """
    result = await generate_questions(CV, FIXED_ANALYSIS)
    for q in result:
        assert len(q.question) <= 400, (
            f"Question too long ({len(q.question)} chars): {q.question}"
        )


@pytest.mark.eval
async def test_gap_questions_have_null_cv_anchor():
    """
    Gap questions should not hallucinate CV anchors — the prompt
    instructs cv_anchor = null for gaps.
    """
    result = await generate_questions(CV, FIXED_ANALYSIS)
    gap_labels = {g.label for g in FIXED_ANALYSIS.skill_gaps}

    for q in result:
        if q.skill in gap_labels:
            assert q.cv_anchor is None, (
                f"Gap question for '{q.skill}' should have cv_anchor=null, "
                f"got: {q.cv_anchor!r}"
            )


@pytest.mark.eval
async def test_covered_skill_questions_have_cv_anchor():
    """
    Questions about skills the candidate demonstrably has should be
    anchored to the CV — helps the student prepare.
    """
    result = await generate_questions(CV, FIXED_ANALYSIS)
    covered_labels = {s.label for s in FIXED_ANALYSIS.covered_skills}

    anchored = [q for q in result if q.skill in covered_labels and q.cv_anchor]
    assert len(anchored) >= 2, (
        "Expected at least 2 covered-skill questions to have a cv_anchor. "
        f"Got: {[(q.skill, q.cv_anchor) for q in result]}"
    )