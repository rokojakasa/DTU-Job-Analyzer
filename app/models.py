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

class JobPostingRequest(BaseModel):
    text: Optional[str] = None
    url: Optional[HttpUrl] = None

    @model_validator(mode="after")
    def exactly_one_source(self) -> "JobPostingRequest":
        if not self.text and not self.url:
            raise ValueError("Provide either 'text' or 'url'.")
        if self.text and self.url:
            raise ValueError("Provide either 'text' or 'url', not both.")
        return self


class JobPostingUploadResponse(BaseModel):
    message: str
    source: str
    character_count: int


# ── Analysis ──────────────────────────────────────────────────────────────────

class CoveredSkill(BaseModel):
    label: str
    evidence_from_cv: str


class SkillGap(BaseModel):
    label: str


class AnalysisResponse(BaseModel):
    job_title_inferred: str
    covered_skills: list[CoveredSkill]
    skill_gaps: list[SkillGap]
    coverage_score: float
    summary: str


# ── Courses ───────────────────────────────────────────────────────────────────

class CourseAction(str, Enum):
    ADD_TO_CV = "add_to_cv"
    CONSIDER_TAKING = "consider_taking"


class CourseRecommendation(BaseModel):
    course_number: str
    title: str
    ects: int
    action: Optional[CourseAction] = None


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
    cv_anchor: str


class QuestionsResponse(BaseModel):
    questions: list[InterviewQuestion]