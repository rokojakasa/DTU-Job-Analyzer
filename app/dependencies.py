from fastapi import HTTPException, Request, status, Depends

from app.store import Store

from app.services.courses.index import ObjectiveChunk
import numpy as np

def get_course_chunks(request: Request) -> list[ObjectiveChunk]:
    return request.app.state.course_chunks

def get_course_matrix(request: Request) -> np.ndarray:
    return request.app.state.course_matrix

def get_store(request: Request) -> Store:
    return request.app.state.store


def require_cv(request: Request) -> Store:
    store = get_store(request)
    if not store.has_cv():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="No CV uploaded. POST plain text or a PDF to /cv first.",
        )
    return store

def require_job_posting(request: Request) -> Store:
    store = get_store(request)
    if not store.has_job_posting():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="No job posting uploaded. POST plain text to /job-posting first.",
        )
    return store

def require_cv_and_job_posting(
    store: Store = Depends(require_cv),
    _: Store = Depends(require_job_posting),
) -> Store:
    return store

def require_analysis(request: Request) -> Store:
    store = get_store(request)
    if not store.has_analysis():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="No analysis available. Call GET /analyse first.",
        )
    return store