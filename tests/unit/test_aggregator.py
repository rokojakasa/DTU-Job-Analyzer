# tests/unit/test_aggregator.py
"""
Tests for app.services.courses.aggregator.

aggregate() and aggregate_all() are pure functions — they take retriever
hits and collapse them into course recommendations. No I/O, no LLM.

Key behaviours we verify:
  - Max pooling: a course's score is the best score among its objectives.
  - Deduplication: multiple objectives from the same course → one recommendation.
  - top_k: never returns more courses than requested.
  - Ordering: highest-scoring course comes first.
  - Edge cases: empty hits, single hit, all hits from the same course.
"""

import pytest
from app.services.courses.aggregator import aggregate, aggregate_all
from app.services.courses.index import ObjectiveChunk


def _chunk(code: str, title: str = "Course", ects: int = 5) -> ObjectiveChunk:
    return ObjectiveChunk(
        course_code=code,
        title=title,
        ects=ects,
        objective=f"{title}: some objective",
    )


class TestAggregate:
    def test_empty_hits_returns_empty_courses(self):
        result = aggregate("some skill", hits=[])
        assert result.courses == []

    def test_single_hit(self):
        hits = [(_chunk("02170", "Database Systems"), 0.9)]
        result = aggregate("design database schema", hits)
        assert len(result.courses) == 1
        assert result.courses[0].course_number == "02170"

    def test_skill_gap_label_preserved(self):
        hits = [(_chunk("02170"), 0.9)]
        result = aggregate("my skill gap", hits)
        assert result.skill_gap == "my skill gap"

    def test_deduplication_keeps_one_course_per_code(self):
        """Two objectives from the same course → one recommendation."""
        hits = [
            (_chunk("02170", "DB"), 0.85),
            (_chunk("02170", "DB"), 0.70),  # same course, lower score
        ]
        result = aggregate("design database schema", hits, top_k=5)
        assert len(result.courses) == 1
        assert result.courses[0].course_number == "02170"

    def test_max_pooling_uses_best_score_for_ranking(self):
        """
        Course A has two objectives scoring 0.60 and 0.50.
        Course B has one objective scoring 0.55.
        Course A should rank first (best score 0.60 > 0.55), not Course B.
        """
        hits = [
            (_chunk("A"), 0.60),
            (_chunk("B"), 0.55),
            (_chunk("A"), 0.50),
        ]
        result = aggregate("skill", hits, top_k=2)
        assert result.courses[0].course_number == "A"
        assert result.courses[1].course_number == "B"

    def test_top_k_respected(self):
        hits = [
            (_chunk("A"), 0.9),
            (_chunk("B"), 0.8),
            (_chunk("C"), 0.7),
            (_chunk("D"), 0.6),
        ]
        result = aggregate("skill", hits, top_k=2)
        assert len(result.courses) == 2

    def test_top_k_larger_than_hits_returns_all(self):
        hits = [(_chunk("A"), 0.9)]
        result = aggregate("skill", hits, top_k=5)
        assert len(result.courses) == 1

    def test_courses_sorted_descending_by_score(self):
        hits = [
            (_chunk("LOW"), 0.4),
            (_chunk("HIGH"), 0.9),
            (_chunk("MID"), 0.7),
        ]
        result = aggregate("skill", hits, top_k=3)
        codes = [c.course_number for c in result.courses]
        assert codes == ["HIGH", "MID", "LOW"]

    def test_course_metadata_preserved(self):
        chunk = ObjectiveChunk(
            course_code="02170",
            title="Database Systems",
            ects=5,
            objective="DB: some objective",
        )
        result = aggregate("skill", [(chunk, 0.9)])
        rec = result.courses[0]
        assert rec.course_number == "02170"
        assert rec.title == "Database Systems"
        assert rec.ects == 5


class TestAggregateAll:
    def test_output_length_matches_input(self):
        gaps = ["gap A", "gap B", "gap C"]
        hits = [
            [(_chunk("02170"), 0.9)],
            [(_chunk("02450"), 0.8)],
            [],
        ]
        results = aggregate_all(gaps, hits)
        assert len(results) == 3

    def test_raises_on_length_mismatch(self):
        with pytest.raises(ValueError, match="same length"):
            aggregate_all(["gap A", "gap B"], [[]])

    def test_empty_hits_for_one_gap_does_not_affect_others(self):
        gaps = ["gap A", "gap B"]
        hits = [
            [(_chunk("02170"), 0.9)],
            [],
        ]
        results = aggregate_all(gaps, hits)
        assert len(results[0].courses) == 1
        assert len(results[1].courses) == 0

    def test_skill_gap_labels_are_correctly_assigned(self):
        gaps = ["design database schema", "apply machine learning"]
        hits = [
            [(_chunk("02170"), 0.9)],
            [(_chunk("02450"), 0.8)],
        ]
        results = aggregate_all(gaps, hits)
        assert results[0].skill_gap == "design database schema"
        assert results[1].skill_gap == "apply machine learning"
