def build_intent_prompt(job_posting: str) -> str:
    return f"""You are an expert technical recruiter.

Analyse the following job posting and extract the employer's actual requirements.

JOB POSTING:
{job_posting}

Follow these steps:

STEP 1 — Infer the job title.

STEP 2 — Identify required skills.
These are technologies, tools, or competencies the employer expects the candidate to have.
Use the exact phrasing from the posting.

Exclude the following — they are not skills:
- Academic degrees (bachelor's, master's, PhD)
- Years of experience requirements
- Soft traits (e.g. "curious mind", "willingness to learn")
- Language requirements (e.g. "fluent in English")
- Anything framed as "tools you'll meet", "you'll learn", "get strong in",
  or similar — these are things the company will teach, not screen for.

STEP 3 — Identify bonus skills.
These are explicitly marked as optional, "a plus", "nice to have", "bonus",
or "not required but welcome". Extract each technology separately.
Do not include soft traits here either.

Respond ONLY with a JSON object — no markdown, no explanation:

{{
  "job_title_inferred": "<string>",
  "required_skills": ["<as written in posting>"],
  "bonus_skills": ["<as written in posting>"]
}}"""


def build_analyse_prompt(cv: str, job_requirements: dict) -> str:
    return f"""You are an expert technical recruiter and skills analyst.

Given a candidate's CV and a structured summary of a job's requirements,
produce a skills gap analysis.

CV:
{cv}

JOB REQUIREMENTS:
{job_requirements}

Follow these steps:

STEP 1 — For each skill in required_skills and bonus_skills, assess whether
the CV demonstrates it. Place each skill in exactly one bucket:

  covered_skills — any evidence exists, even indirect. Classify the evidence type:
    · "explicit"    — skill is named and tied to a project, role, or course
    · "implicit"    — skill is unnamed but expected given the context
                      (e.g. TypeScript implied by React work)
    · "bare_claim"  — skill appears only in a flat skills list, no supporting context

  skill_gaps — no evidence of this skill anywhere in the CV.
    · Set category: "required" for skills from required_skills
    · Set category: "bonus" for skills from bonus_skills

STEP 2 — Apply domain knowledge when assessing coverage.
If the CV demonstrates a technology that is functionally equivalent to a
required skill (e.g. React instead of Angular, PostgreSQL instead of MS SQL),
mark that skill as covered with evidence_type "implicit".

STEP 3 — Canonical labels.
For each skill produce:
  - "label": canonical verb-noun form ("manage relational databases", not "MS SQL")
  - "job_posting_label": the skill exactly as listed in required_skills or bonus_skills

STEP 4 — Write a 1-2 sentence summary for the candidate.
  If any skills are bare_claim, name them and suggest adding supporting context.
  If any gaps are bonus rather than required, note they are non-critical.

Respond ONLY with a JSON object — no markdown, no explanation:

{{
  "job_title_inferred": "<string>",
  "covered_skills": [
    {{
      "label": "<canonical verb-noun>",
      "job_posting_label": "<from required_skills or bonus_skills>",
      "evidence_from_cv": "<brief explanation>",
      "evidence_type": "explicit" | "implicit" | "bare_claim"
    }}
  ],
  "skill_gaps": [
    {{
      "label": "<canonical verb-noun>",
      "job_posting_label": "<from required_skills or bonus_skills>",
      "category": "required" | "bonus"
    }}
  ],
  "summary": "<string>"
}}"""