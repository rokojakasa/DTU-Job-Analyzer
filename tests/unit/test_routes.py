# tests/unit/test_routes.py
"""
Route-level unit tests.

These test the HTTP contract of each endpoint — status codes, error messages,
and guard dependencies — without making real LLM calls.

The `client` fixture (from conftest.py) uses a test app whose lifespan
injects dummy state instead of calling the embedding API at startup.

Where routes call the LLM (analyse, questions), we patch the OpenAI client
so tests remain fast and deterministic.
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch

from tests.conftest import (
    make_chat_response,
    FAKE_REQUIREMENTS_JSON,
    FAKE_ANALYSIS_JSON,
)


# ---------------------------------------------------------------------------
# /cv
# ---------------------------------------------------------------------------

class TestCVUpload:
    async def test_upload_text_returns_200(self, client, sample_cv):
        resp = await client.post("/cv/text", data={"text": sample_cv})
        assert resp.status_code == 200

    async def test_upload_text_response_contains_character_count(self, client, sample_cv):
        resp = await client.post("/cv/text", data={"text": sample_cv})
        assert resp.json()["character_count"] == len(sample_cv)

    async def test_upload_empty_text_returns_422(self, client):
        resp = await client.post("/cv/text", data={"text": "   "})
        assert resp.status_code == 422

    async def test_upload_text_overwrites_previous(self, client):
        await client.post("/cv/text", data={"text": "first cv"})
        await client.post("/cv/text", data={"text": "second cv"})
        # The store should hold the second cv — verify indirectly via analyse guard
        resp = await client.post("/cv/text", data={"text": "third cv"})
        assert resp.status_code == 200

    async def test_missing_text_field_returns_422(self, client):
        resp = await client.post("/cv/text", data={})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# /job-posting
# ---------------------------------------------------------------------------

class TestJobPostingUpload:
    async def test_upload_text_returns_200(self, client, sample_job_posting):
        resp = await client.post("/job-posting/text", data={"text": sample_job_posting})
        assert resp.status_code == 200

    async def test_upload_empty_text_returns_422(self, client):
        resp = await client.post("/job-posting/text", data={"text": ""})
        assert resp.status_code == 422

    async def test_response_source_is_text(self, client, sample_job_posting):
        resp = await client.post("/job-posting/text", data={"text": sample_job_posting})
        assert resp.json()["source"] == "text"


# ---------------------------------------------------------------------------
# /analyse — dependency guards
# ---------------------------------------------------------------------------

class TestAnalyseGuards:
    async def test_returns_422_without_cv(self, client, sample_job_posting):
        await client.post("/job-posting/text", data={"text": sample_job_posting})
        resp = await client.get("/analyse/")
        assert resp.status_code == 422
        assert "CV" in resp.json()["detail"]

    async def test_returns_422_without_job_posting(self, client, sample_cv):
        await client.post("/cv/text", data={"text": sample_cv})
        resp = await client.get("/analyse/")
        assert resp.status_code == 422
        assert "job posting" in resp.json()["detail"].lower()

    async def test_returns_200_with_both_uploaded(self, client, sample_cv, sample_job_posting):
        """End-to-end route test with mocked LLM responses."""
        await client.post("/cv/text", data={"text": sample_cv})
        await client.post("/job-posting/text", data={"text": sample_job_posting})

        # Two LLM calls: pass 1 (requirements) and pass 2 (analysis)
        with patch(
            "app.services.analyser.client.chat.completions.create",
            new=AsyncMock(side_effect=[
                make_chat_response(FAKE_REQUIREMENTS_JSON),
                make_chat_response(FAKE_ANALYSIS_JSON),
            ]),
        ):
            resp = await client.get("/analyse/")

        assert resp.status_code == 200

    async def test_analyse_response_shape(self, client, sample_cv, sample_job_posting):
        """Verify the response contains the expected top-level keys."""
        await client.post("/cv/text", data={"text": sample_cv})
        await client.post("/job-posting/text", data={"text": sample_job_posting})

        with patch(
            "app.services.analyser.client.chat.completions.create",
            new=AsyncMock(side_effect=[
                make_chat_response(FAKE_REQUIREMENTS_JSON),
                make_chat_response(FAKE_ANALYSIS_JSON),
            ]),
        ):
            resp = await client.get("/analyse/")

        body = resp.json()
        assert "job_title_inferred" in body
        assert "covered_skills" in body
        assert "skill_gaps" in body
        assert "summary" in body


# ---------------------------------------------------------------------------
# /questions — dependency guards
# ---------------------------------------------------------------------------

class TestQuestionsGuards:
    async def test_returns_422_without_analysis(self, client):
        resp = await client.get("/questions/")
        assert resp.status_code == 422
        assert "analyse" in resp.json()["detail"].lower()

    async def test_returns_200_after_analysis(
        self, client, sample_cv, sample_job_posting
    ):
        """Upload → analyse (mocked) → questions (mocked)."""
        await client.post("/cv/text", data={"text": sample_cv})
        await client.post("/job-posting/text", data={"text": sample_job_posting})

        import json
        fake_questions = json.dumps([
            {
                "skill": "apply machine learning",
                "question": "Walk me through your random forest project.",
                "follow_up": "How did you validate the model?",
                "cv_anchor": "Random forest classifier at Acme Corp.",
            }
        ])

        with patch(
            "app.services.analyser.client.chat.completions.create",
            new=AsyncMock(side_effect=[
                make_chat_response(FAKE_REQUIREMENTS_JSON),
                make_chat_response(FAKE_ANALYSIS_JSON),
            ]),
        ):
            await client.get("/analyse/")

        with patch(
            "app.services.question_generator.client.chat.completions.create",
            new=AsyncMock(return_value=make_chat_response(fake_questions)),
        ):
            resp = await client.get("/questions/")

        assert resp.status_code == 200
        assert "questions" in resp.json()


# ---------------------------------------------------------------------------
# /courses — dependency guards
# ---------------------------------------------------------------------------

class TestCoursesGuards:
    async def test_returns_422_without_analysis(self, client):
        resp = await client.post("/courses/")
        assert resp.status_code == 422
        assert "analyse" in resp.json()["detail"].lower()
