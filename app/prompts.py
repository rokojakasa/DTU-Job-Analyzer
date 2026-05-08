def build_analyse_prompt(cv: str, job_posting: str) -> str:
    return f"""You are an expert technical recruiter and skills analyst.

Given a candidate's CV and a job posting, produce a structured skills analysis.

---
CV:
{cv}

---
JOB POSTING:
{job_posting}

---

Follow these steps in order:

STEP 1 — Infer the job title from the posting.

STEP 2 — Extract all skills required by the job posting.
For each skill produce two labels:
  - "job_posting_label": the skill exactly as phrased in the job posting
  - "label": a canonical form for matching against academic course descriptions
      · Use verb-noun form ("manage relational databases", "apply machine learning")
      · Avoid brand names ("manage relational databases" not "use MS SQL")
  - A canonical label may appear in at most one bucket. If the same canonical skill appears in both
  covered and skill_gaps (e.g. because the job lists two related technologies), 
  place it in covered_skills if any evidence exists, and collapse the gap entry.

STEP 3 — For each required skill, assess whether the CV demonstrates it.
Place each skill in exactly one of two buckets:

  covered_skills — any evidence exists, even indirect. Classify the evidence type:
    · "explicit"    — skill is named and tied to a project, role, or course
    · "implicit"    — skill is unnamed but expected given the context
                      (e.g. TypeScript implied by React work)
    · "bare_claim"  — skill appears only in a flat skills list, no supporting context
  Include a brief evidence note for all three types.

  skill_gaps — no evidence of this skill anywhere in the CV

STEP 4 — Write a 1-2 sentence summary for the candidate.
  If any skills are classified as bare_claim, name them and suggest adding
  supporting context such as a project, course, or work experience.

---

Respond ONLY with a JSON object matching this schema — no markdown, no explanation:

{{
  "job_title_inferred": "<string>",
  "covered_skills": [
    {{
      "job_posting_label": "<as written in job posting>",
      "label": "<canonical verb-noun form>",
      "evidence_from_cv": "<brief explanation>",
      "evidence_type": "explicit" | "implicit" | "bare_claim"
    }}
  ],
  "skill_gaps": [
    {{
      "job_posting_label": "<as written in job posting>",
      "label": "<canonical verb-noun form>"
    }}
  ],
  "summary": "<string>"
}}"""