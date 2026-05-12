from fastapi import APIRouter, Depends

from app.dependencies import require_analysis
from app.models import QuestionsResponse
from app.services.question_generator import generate_questions
from app.store import Store

router = APIRouter()

@router.get("/", response_model=QuestionsResponse, status_code=200)
async def get_questions(
    store: Store = Depends(require_analysis),
) -> QuestionsResponse:
    """
    Generate tailored interview questions based on the last analysis.
    Requires GET /analyse to have been called first.
    """
    questions = await generate_questions(
        cv=store.cv_text,
        analysis=store.last_analysis,
    )
    return QuestionsResponse(questions=questions)