from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class AnalysisResult:
    job_title_inferred: str
    covered_skills: list[dict]
    skill_gaps: list[dict]
    coverage_score: float
    summary: str

@dataclass
class Store:
    cv_text: Optional[str] = field(default=None)
    job_posting_text: Optional[str] = field(default=None)
    last_analysis: Optional[AnalysisResult] = field(default=None)
    
    def set_cv(self, text: str):
        self.cv_text = text
        self.last_analysis = None
    

    def set_job_posting(self, text: str) -> None:
        self.job_posting_text = text
        self.last_analysis = None  # invalidate stale analysis
 
    def has_cv(self) -> bool:
        return self.cv_text is not None
 
    def has_job_posting(self) -> bool:
        return self.job_posting_text is not None
 
    def is_ready_for_analysis(self) -> bool:
        return self.has_cv() and self.has_job_posting()
 
    def has_analysis(self) -> bool:
        return self.last_analysis is not None
