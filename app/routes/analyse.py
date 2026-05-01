from app.store import AnalysisResult
from app.dependencies import require_cv_and_job_posting
from fastapi import APIRouter, Depends, HTTPException, status, Form

router = APIRouter()

@router.get("/", response_model=AnalysisResult, status_code=status.HTTP_200_OK)
async def get_analysis(
    store = Depends(require_cv_and_job_posting),
) -> AnalysisResult:
    """Perform analysis of the uploaded CV and job posting, returning skill gaps, course recommendations, and interview questions."""
    
    return store.get_analysis()