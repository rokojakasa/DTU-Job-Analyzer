import json
import logging
from openai import AsyncOpenAI

from app.config import CAMPUSAI_API_KEY, CAMPUSAI_URL, CHAT_MODEL
from app.models import InterviewQuestion
from app.prompts import build_questions_prompt
from app.store import AnalysisResult

logger = logging.getLogger(__name__)

client = AsyncOpenAI(
    api_key=CAMPUSAI_API_KEY,
    base_url=CAMPUSAI_URL,
)

async def generate_questions(cv: str, analysis: AnalysisResult) -> list[InterviewQuestion]:
    prompt = build_questions_prompt(
        cv=cv,
        covered_skills=[s if isinstance(s, dict) else s.__dict__ for s in analysis.covered_skills],
        skill_gaps=[g if isinstance(g, dict) else g.__dict__ for g in analysis.skill_gaps],
        job_title=analysis.job_title_inferred,
    )

    response = await client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown fences if present — reuse same pattern as analyser
    if raw.startswith("```"):
        raw = raw.split("```")[1]
    if raw.startswith("json"):
        raw = raw[4:]

    try:
        data = json.loads(raw.strip())
    except json.JSONDecodeError as e:
        raise ValueError(f"Question generator returned invalid JSON: {e}\n\nRaw output:\n{raw}")

    return [InterviewQuestion(**item) for item in data]