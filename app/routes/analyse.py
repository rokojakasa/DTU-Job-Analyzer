from fastapi import APIRouter, Depends
from app.dependencies import require_cv_and_job_posting
from app.models import AnalysisResponse
from app.services.analyser import run_analysis
from app.store import Store

router = APIRouter()

@router.get("/", response_model=AnalysisResponse, status_code=200)
async def get_analysis(store: Store = Depends(require_cv_and_job_posting)) -> AnalysisResponse:

    result = await run_analysis(store.cv_text, store.job_posting_text)
    store.last_analysis = result
    return result