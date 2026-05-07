# app/routes/courses.py
from __future__ import annotations

import asyncio
import logging
import numpy as np
from typing import Annotated

from fastapi import APIRouter, Depends, Query

from app.config import COURSES_TOP_K
from app.dependencies import get_course_chunks, get_course_matrix, require_analysis
from app.models import CoursesResponse
from app.services.courses.aggregator import aggregate_all
from app.services.courses.index import ObjectiveChunk
from app.services.courses.retriever import RetrievalMode, retrieve
from app.store import Store

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/", response_model=CoursesResponse, status_code=200)
async def get_courses(
    store: Store = Depends(require_analysis),
    chunks: list[ObjectiveChunk] = Depends(get_course_chunks), 
    matrix: np.ndarray = Depends(get_course_matrix),   
    retrieval_mode: Annotated[
        RetrievalMode,
        Query(
            description=(
                "Retrieval strategy to use when matching skill gaps to courses. "
                "'sparse' uses BM25 keyword matching (fast, no API calls). "
                "'dense' uses embedding similarity (handles paraphrasing). "
                "'hybrid' combines both for best results (default)."
            )
        ),
    ] = "hybrid",
) -> CoursesResponse:
    """
    For each skill gap identified by GET /analyse, suggest DTU courses
    ranked by relevance to the gap.

    Requires GET /analyse to have been called first.
    """
    analysis = store.last_analysis
    skill_gaps = analysis.skill_gaps

    if not skill_gaps:
        logger.info("No skill gaps found in analysis — returning empty recommendations.")
        return CoursesResponse(recommendations=[])

    gap_labels = [gap["label"] for gap in skill_gaps]

    logger.info(
        "Retrieving courses for %d skill gaps using %r mode.",
        len(gap_labels),
        retrieval_mode,
    )

    # Retrieve concurrently — each gap is independent.
    hits_per_gap: list = await asyncio.gather(*[
        retrieve(
            query=label,
            chunks=chunks,
            matrix=matrix,
            top_k=COURSES_TOP_K,
            mode=retrieval_mode,
        )
        for label in gap_labels
    ])

    recommendations = aggregate_all(
        skill_gaps=gap_labels,
        hits_per_gap=list(hits_per_gap),
        top_k=COURSES_TOP_K,
        completed_courses=None,  # transcript support to be added later
    )

    return CoursesResponse(recommendations=recommendations)