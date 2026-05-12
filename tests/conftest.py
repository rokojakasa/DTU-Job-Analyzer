# tests/conftest.py
"""
Shared fixtures for the DTU Job Analyser test suite.

Design decisions:
- The FastAPI lifespan (build_index) is overridden in tests so no real
  embedding API calls happen during unit tests.
- LLM calls in unit tests are patched at the OpenAI client level via AsyncMock.
- Eval tests use the real app with real credentials — they are gated behind
  the --run-eval CLI flag and the @pytest.mark.eval marker.
"""

from __future__ import annotations

import json
import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock
from contextlib import asynccontextmanager

from httpx import AsyncClient, ASGITransport
from fastapi import FastAPI

from app.store import Store
from app.services.courses.index import ObjectiveChunk


# ---------------------------------------------------------------------------
# CLI option
# ---------------------------------------------------------------------------

def pytest_addoption(parser):
    parser.addoption(
        "--run-eval",
        action="store_true",
        default=False,
        help="Run LLM evaluation tests (requires real CampusAI credentials).",
    )


def pytest_collection_modifyitems(config, items):
    """Skip @pytest.mark.eval tests unless --run-eval was passed."""
    if config.getoption("--run-eval"):
        return  # run everything
    skip_eval = pytest.mark.skip(reason="Eval test — pass --run-eval to include.")
    for item in items:
        if item.get_closest_marker("eval"):
            item.add_marker(skip_eval)


# ---------------------------------------------------------------------------
# Sample fixtures — realistic but short enough to read at a glance
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_cv() -> str:
    return """
    Jane Doe — Software Engineer
    
    Skills: Python, SQL, scikit-learn, Git, Docker
    
    Experience:
    - Implemented a random forest classifier for churn prediction at Acme Corp.
      Used pandas and scikit-learn; deployed via Docker on AWS EC2.
    - Built a REST API with FastAPI and PostgreSQL for internal tooling.
    
    Education:
    - BSc Software Engineering, DTU, 2023
    - Course 02450: Introduction to Machine Learning
    - Course 02170: Database Systems
    """.strip()


@pytest.fixture
def sample_job_posting() -> str:
    return """
    We are looking for a Machine Learning Engineer.
    
    Required skills:
    - Python programming
    - Experience with machine learning frameworks (scikit-learn, PyTorch)
    - SQL and relational databases
    - Docker and containerisation
    - REST API development
    
    Bonus:
    - Experience with cloud platforms (AWS, GCP)
    - Knowledge of MLflow or similar experiment tracking
    """.strip()


@pytest.fixture
def sample_analysis_result():
    """A pre-built AnalysisResult so tests don't need to call the LLM."""
    from app.store import AnalysisResult
    return AnalysisResult(
        job_title_inferred="Machine Learning Engineer",
        covered_skills=[
            {
                "label": "apply machine learning",
                "job_posting_label": "machine learning frameworks",
                "evidence_from_cv": "Random forest classifier at Acme Corp.",
                "evidence_type": "explicit",
                "category": "required",
            },
            {
                "label": "manage relational databases",
                "job_posting_label": "SQL and relational databases",
                "evidence_from_cv": "PostgreSQL in FastAPI project.",
                "evidence_type": "explicit",
                "category": "required",
            },
        ],
        skill_gaps=[
            {
                "label": "track machine learning experiments",
                "job_posting_label": "MLflow or similar experiment tracking",
                "category": "bonus",
            }
        ],
        summary="CV covers 4 of 5 required skills. Main gap is experiment tracking.",
    )


# ---------------------------------------------------------------------------
# Minimal ObjectiveChunk fixtures for retriever / aggregator tests
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_chunks() -> list[ObjectiveChunk]:
    return [
        ObjectiveChunk(
            course_code="02170",
            title="Database Systems",
            ects=5,
            objective="Database Systems: design and implement relational database schemas",
        ),
        ObjectiveChunk(
            course_code="02170",
            title="Database Systems",
            ects=5,
            objective="Database Systems: write complex SQL queries",
        ),
        ObjectiveChunk(
            course_code="02450",
            title="Introduction to Machine Learning",
            ects=5,
            objective="Introduction to Machine Learning: apply supervised learning algorithms",
        ),
        ObjectiveChunk(
            course_code="02613",
            title="High-Performance Computing",
            ects=5,
            objective="High-Performance Computing: deploy applications on cloud infrastructure",
        ),
    ]


@pytest.fixture
def sample_matrix(sample_chunks) -> np.ndarray:
    """
    Random unit-normalised embeddings — just enough shape for cosine
    similarity to run without crashing. Not meaningful for quality tests.
    """
    rng = np.random.default_rng(42)
    raw = rng.random((len(sample_chunks), 64), dtype=np.float32)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    return raw / norms


# ---------------------------------------------------------------------------
# App fixture — overrides lifespan so no real API calls at startup
# ---------------------------------------------------------------------------

@pytest.fixture
def app_with_state(sample_chunks, sample_matrix) -> FastAPI:
    """
    A FastAPI app with a lightweight test lifespan injected at construction
    time. This avoids calling build_index() (which hits the embedding API)
    and instead populates app.state with pre-built fixtures.

    We pass the lifespan into create_app() rather than replacing it after
    construction — FastAPI bakes the lifespan in at __init__ time, so
    post-hoc replacement via router.lifespan_context is not reliable.
    """
    from app.main import create_app

    @asynccontextmanager
    async def test_lifespan(app: FastAPI):
        app.state.store = Store()
        app.state.course_chunks = sample_chunks
        app.state.course_matrix = sample_matrix
        yield

    return create_app(lifespan=test_lifespan)


@pytest.fixture
async def client(app_with_state) -> AsyncClient:
    """Async HTTP client wired to the test app. Used in all route tests."""
    async with AsyncClient(
        transport=ASGITransport(app=app_with_state),
        base_url="http://test",
    ) as ac:
        yield ac


# ---------------------------------------------------------------------------
# LLM mock helpers
# ---------------------------------------------------------------------------

def make_chat_response(content: str) -> MagicMock:
    """
    Build a minimal mock that looks like an OpenAI ChatCompletion response.
    Usage: patch "app.services.analyser.client.chat.completions.create"
           with AsyncMock(return_value=make_chat_response(json_string))
    """
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    response = MagicMock()
    response.choices = [choice]
    return response


# Canned LLM responses for analyser tests — valid JSON the prompts expect.
FAKE_REQUIREMENTS_JSON = json.dumps({
    "job_title_inferred": "Machine Learning Engineer",
    "required_skills": ["Python programming", "SQL"],
    "bonus_skills": ["MLflow"],
})

FAKE_ANALYSIS_JSON = json.dumps({
    "job_title_inferred": "Machine Learning Engineer",
    "covered_skills": [
        {
            "label": "apply machine learning",
            "job_posting_label": "Python programming",
            "evidence_from_cv": "Random forest project.",
            "evidence_type": "explicit",
            "category": "required",
        }
    ],
    "skill_gaps": [
        {
            "label": "track machine learning experiments",
            "job_posting_label": "MLflow",
            "category": "bonus",
        }
    ],
    "summary": "Good coverage of required skills.",
})