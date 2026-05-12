# app/services/courses/aggregator.py
from __future__ import annotations
 
import logging
from collections import defaultdict
 
from app.models import CourseRecommendation, SkillCourses
from app.services.courses.index import ObjectiveChunk
 
logger = logging.getLogger(__name__)
 
# How many courses to return per skill gap.
# Matches COURSES_TOP_K in config.py — imported here to keep aggregator
# self-contained; routes can override by passing top_k explicitly.
_DEFAULT_TOP_K = 3

def aggregate(
    skill_gap: str,
    hits: list[tuple[ObjectiveChunk, float]],
    top_k: int = _DEFAULT_TOP_K,
) -> SkillCourses:
    """
    Collapse objective-level retriever hits into a ranked list of unique
    course recommendations for a single skill gap.
 
    Scoring uses max pooling — a course's score is the highest similarity
    score among all its matching objectives. This avoids over-rewarding
    courses with many objectives.
 
    Args:
        skill_gap:          The skill gap label, e.g. "design database schema"
        hits:               (ObjectiveChunk, score) pairs from the retriever,
                            sorted descending by score.
        top_k:              Maximum number of courses to return.
        completed_courses:  Set of course codes parsed from a grade transcript.
                            If provided, each recommendation includes an action
                            field indicating whether the student should add the
                            course to their CV or consider taking it.
 
    Returns:
        SkillCourses with the top-k deduplicated, ranked course recommendations.
    """
    # --- Step 1: group hits by course_code, keep the best score per course ---
    # defaultdict(float) initialises to 0.0, so the max() comparison is safe.
    best_score: dict[str, float] = defaultdict(float)
 
    # We also need to recover course metadata (title, ects) from the chunk.
    # Since all chunks for the same course share identical metadata, we just
    # keep the first chunk seen per course_code as the metadata carrier.
    representative: dict[str, ObjectiveChunk] = {}
 
    for chunk, score in hits:
        code = chunk.course_code
        if score > best_score[code]:
            best_score[code] = score
        if code not in representative:  # always capture metadata on first sight
            representative[code] = chunk
 
    # --- Step 2: rank courses by their best score ---
    ranked_codes = sorted(best_score, key=lambda c: best_score[c], reverse=True)
    top_codes = ranked_codes[:top_k]
 
    # --- Step 3: build CourseRecommendation objects ---
    recommendations: list[CourseRecommendation] = []
 
    for code in top_codes:
        chunk = representative[code]
 
        recommendations.append(CourseRecommendation(
            course_number=code,
            title=chunk.title,
            ects=chunk.ects,
        ))
 
        logger.debug(
            "  [%s] %s — score %.4f%s",
            code,
            chunk.title,
            best_score[code],
        )
 
    return SkillCourses(skill_gap=skill_gap, courses=recommendations)

def aggregate_all(
    skill_gaps: list[str],
    hits_per_gap: list[list[tuple[ObjectiveChunk, float]]],
    top_k: int = _DEFAULT_TOP_K,
) -> list[SkillCourses]:
    """
    Aggregate retriever results for all skill gaps in one call.
 
    Args:
        skill_gaps:      Ordered list of skill gap labels.
        hits_per_gap:    Parallel list of retriever hits, one per skill gap.
        top_k:           Maximum courses to return per skill gap.
        completed_courses: Optional set of completed course codes from transcript.
 
    Returns:
        List of SkillCourses, one per skill gap, in the same order.
    """
    if len(skill_gaps) != len(hits_per_gap):
        raise ValueError(
            f"skill_gaps and hits_per_gap must have the same length, "
            f"got {len(skill_gaps)} and {len(hits_per_gap)}."
        )
 
    results = []
    for skill_gap, hits in zip(skill_gaps, hits_per_gap):
        logger.debug("Aggregating hits for skill gap: %r", skill_gap)
        result = aggregate(skill_gap, hits, top_k)
        results.append(result)
 
    return results
 
 
 
