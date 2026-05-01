import json
from openai import AsyncOpenAI

from app.config import CAMPUSAI_API_KEY, CAMPUSAI_URL, CHAT_MODEL
from app.prompts import build_analyse_prompt
from app.store import AnalysisResult

client = AsyncOpenAI(
    api_key=CAMPUSAI_API_KEY,
    base_url=CAMPUSAI_URL,
)

async def run_analysis(cv: str, job_posting: str) -> AnalysisResult:
    prompt = build_analyse_prompt(cv, job_posting)

    response = await client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,  # deterministic — you want consistent structured output
    )

    raw = response.choices[0].message.content.strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1]
    if raw.startswith("json"):
        raw = raw[4:]
        
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM returned invalid JSON: {e}\n\nRaw output:\n{raw}")

    return AnalysisResult(
        job_title_inferred=data["job_title_inferred"],
        covered_skills=data["covered_skills"],
        skill_gaps=data["skill_gaps"],
        coverage_score=data["coverage_score"],
        summary=data["summary"],
    )