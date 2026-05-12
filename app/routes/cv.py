import io
 
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status, Form
 
from app.dependencies import get_store
from app.models import CVRequest, CVUploadResponse
from app.store import Store
from app.services.pdf_parser import extract_text_from_pdf

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
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="CV text is empty.",
        )
    store.set_cv(text)
    return CVUploadResponse(
        message=f"CV uploaded successfully ({len(text):,} characters).",
        character_count=len(text),
    )
    
@router.post("/pdf", response_model=CVUploadResponse, status_code=200)
async def upload_cv_pdf(
    file: UploadFile = File(...),
    store: Store = Depends(get_store),
) -> CVUploadResponse:
    contents = await file.read()
    text = extract_text_from_pdf(contents)
    if not text:
        raise HTTPException(status_code=422, detail="Could not extract text from PDF.")
    store.set_cv(text)
    return CVUploadResponse(message=f"CV uploaded successfully ({len(text):,} characters).", character_count=len(text))

