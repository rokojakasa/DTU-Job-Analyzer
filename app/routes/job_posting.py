from fastapi import APIRouter, Depends, HTTPException, status, Form

from app.dependencies import get_store
from app.models import JobPostingUploadResponse
from app.store import Store

router = APIRouter()

@router.post("/text", response_model=JobPostingUploadResponse, status_code=status.HTTP_200_OK)
async def upload_job_posting(
    text: str = Form(...),
    store: Store = Depends(get_store),
) -> JobPostingUploadResponse:
    """Upload a job posting as JSON: `{ "text": "..." }` or `{ "url": "https://..." }`"""
    if not text:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Job posting text is empty.",
        )
        
    text = text.strip()
    if not text:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Job posting text is empty after stripping whitespace.",
        )
    store.set_job_posting(text)
    return JobPostingUploadResponse(
        message=f"Job posting uploaded successfully ({len(text):,} characters).",
        source="text",
        character_count=len(text),
    )
