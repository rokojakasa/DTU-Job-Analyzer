import io
 
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status, Form
 
from app.dependencies import get_store
from app.models import CVRequest, CVUploadResponse
from app.store import Store

router = APIRouter()

@router.post("/text", response_model=CVUploadResponse, status_code=status.HTTP_200_OK)
async def upload_cv_text(
    text: str = Form(...),
    store: Store = Depends(get_store),
) -> CVUploadResponse:
    """
    Upload a CV as plain text.
 
    Send the CV as a raw `text/plain` body — paste directly, no escaping needed.
 
    ```
    POST /cv/text
    Content-Type: text/plain
 
    John Doe. Software Engineer...
    ```
    """
    text = text.strip()
    if not text:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="CV text is empty.",
        )
    store.set_cv(text)
    return CVUploadResponse(
        message=f"CV uploaded successfully ({len(text):,} characters).",
        character_count=len(text),
    )


