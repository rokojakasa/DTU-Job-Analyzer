from enum import Enum
from typing import Optional

from pydantic import BaseModel, HttpUrl, model_validator


# ── CV ────────────────────────────────────────────────────────────────────────

class CVRequest(BaseModel):
    text: str


class CVUploadResponse(BaseModel):
    message: str
    character_count: int


# ── Job Posting ───────────────────────────────────────────────────────────────

class JobPostingUploadResponse(BaseModel):
    message: str
    source: str
    character_count: int


# ── Job Intent (internal, Pass 1 output) ─────────────────────────────────────

class JobRequirements(BaseModel):
    job_title_inferred: str
    required_skills: list[str]
    bonus_skills: list[str]


# ── Analysis ──────────────────────────────────────────────────────────────────

class EvidenceType(str, Enum):
    EXPLICIT = "explicit"
    IMPLICIT = "implicit"
    BARE_CLAIM = "bare_claim"


class SkillCategory(str, Enum):
    REQUIRED = "required"
    BONUS = "bonus"


class CoveredSkill(BaseModel):
    label: str
    job_posting_label: str
    evidence_from_cv: str
    evidence_type: EvidenceType
    category: SkillCategory  


class SkillGap(BaseModel):
    label: str
    job_posting_label: str
    category: SkillCategory

class AnalysisResponse(BaseModel):
    job_title_inferred: str
    covered_skills: list[CoveredSkill]
    skill_gaps: list[SkillGap]
    summary: str


# ── Courses ───────────────────────────────────────────────────────────────────

class CourseRecommendation(BaseModel):
    course_number: str
    title: str
    ects: int


class SkillCourses(BaseModel):
    skill_gap: str
    courses: list[CourseRecommendation]


class CoursesResponse(BaseModel):
    recommendations: list[SkillCourses]


# ── Questions ─────────────────────────────────────────────────────────────────

class InterviewQuestion(BaseModel):
    skill: str
    question: str
    follow_up: str
    cv_anchor: Optional[str] = None


class QuestionsResponse(BaseModel):
    questions: list[InterviewQuestion]