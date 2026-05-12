# tests/unit/test_store.py
"""
Tests for the Store state machine.

These are pure unit tests — no HTTP, no LLM, no fixtures beyond Store itself.
We're verifying the invariants the rest of the app relies on:
  - uploading invalidates stale analysis
  - readiness flags reflect actual state
"""

from app.store import Store
from app.models import AnalysisResponse


def _make_analysis() -> AnalysisResponse:
    return AnalysisResponse(
        job_title_inferred="Engineer",
        covered_skills=[],
        skill_gaps=[],
        summary="Test summary.",
    )


class TestInitialState:
    def test_cv_is_none(self):
        assert Store().cv_text is None

    def test_job_posting_is_none(self):
        assert Store().job_posting_text is None

    def test_last_analysis_is_none(self):
        assert Store().last_analysis is None

    def test_has_cv_false(self):
        assert not Store().has_cv()

    def test_has_job_posting_false(self):
        assert not Store().has_job_posting()

    def test_is_ready_false(self):
        assert not Store().is_ready_for_analysis()

    def test_has_analysis_false(self):
        assert not Store().has_analysis()


class TestSetCV:
    def test_stores_text(self):
        s = Store()
        s.set_cv("my cv")
        assert s.cv_text == "my cv"

    def test_has_cv_becomes_true(self):
        s = Store()
        s.set_cv("my cv")
        assert s.has_cv()

    def test_uploading_cv_invalidates_analysis(self):
        """Core invariant: stale analysis must never survive a new upload."""
        s = Store()
        s.last_analysis = _make_analysis()
        s.set_cv("new cv")
        assert s.last_analysis is None

    def test_overwrite_cv(self):
        s = Store()
        s.set_cv("old cv")
        s.set_cv("new cv")
        assert s.cv_text == "new cv"


class TestSetJobPosting:
    def test_stores_text(self):
        s = Store()
        s.set_job_posting("job posting")
        assert s.job_posting_text == "job posting"

    def test_has_job_posting_becomes_true(self):
        s = Store()
        s.set_job_posting("job posting")
        assert s.has_job_posting()

    def test_uploading_job_posting_invalidates_analysis(self):
        s = Store()
        s.last_analysis = _make_analysis()
        s.set_job_posting("new posting")
        assert s.last_analysis is None


class TestReadiness:
    def test_ready_when_both_uploaded(self):
        s = Store()
        s.set_cv("cv")
        s.set_job_posting("job")
        assert s.is_ready_for_analysis()

    def test_not_ready_with_only_cv(self):
        s = Store()
        s.set_cv("cv")
        assert not s.is_ready_for_analysis()

    def test_not_ready_with_only_job_posting(self):
        s = Store()
        s.set_job_posting("job")
        assert not s.is_ready_for_analysis()

    def test_has_analysis_true_after_setting(self):
        s = Store()
        s.last_analysis = _make_analysis()
        assert s.has_analysis()
