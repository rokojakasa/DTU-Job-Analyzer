import json
import logging
from openai import AsyncOpenAI

from app.config import CAMPUSAI_API_KEY, CAMPUSAI_URL, CHAT_MODEL
from app.models import JobRequirements
from app.prompts import build_intent_prompt, build_analyse_prompt
from app.store import AnalysisResult

logger = logging.getLogger(__name__)

client = AsyncOpenAI(
    api_key=CAMPUSAI_API_KEY,
    base_url=CAMPUSAI_URL,
)


def _parse_json(raw: str, label: str) -> dict:
    """Strip markdown fences and parse JSON. Raises ValueError on failure."""
    if raw.startswith("```"):
        raw = raw.split("```")[1]
    if raw.startswith("json"):
        raw = raw[4:]
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError as e:
        raise ValueError(f"{label} returned invalid JSON: {e}\n\nRaw output:\n{raw}")


async def _interpret_job_posting(job_posting: str) -> JobRequirements:
    """Pass 1 — extract structured requirements from the raw job posting."""
    prompt = build_intent_prompt(job_posting)
    response = await client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    raw = response.choices[0].message.content.strip()
    data = _parse_json(raw, "Pass 1 (requirements)")
    requirements = JobRequirements(**data)
    logger.debug("Job requirements:\n%s", requirements.model_dump_json(indent=2))
    return requirements


async def _analyse_cv(cv: str, requirements: JobRequirements) -> AnalysisResult:
    """Pass 2 — match CV against structured requirements, produce gap analysis."""
    prompt = build_analyse_prompt(cv, requirements.model_dump_json(indent=2))
    response = await client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    raw = response.choices[0].message.content.strip()
    data = _parse_json(raw, "Pass 2 (analysis)")
    return AnalysisResult(
        job_title_inferred=data["job_title_inferred"],
        covered_skills=data["covered_skills"],
        skill_gaps=data["skill_gaps"],
        summary=data["summary"],
    )


async def run_analysis(cv: str, job_posting: str) -> AnalysisResult:
    requirements = await _interpret_job_posting(job_posting)
    return await _analyse_cv(cv, requirements)