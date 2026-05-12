import json

def build_intent_prompt(job_posting: str) -> str:
    return f"""You are an expert technical recruiter.

Analyse the following job posting and extract the employer's actual requirements.
The job posting may be in any language. Regardless of the input language,
all output fields must be in English.

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
produce a skills gap analysis. The job documents may be in any language. Regardless of the input language,
all output fields must be in English.

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
    Set category to 'required' or 'bonus' based on which list the skill came from."
    

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
      "evidence_type": "explicit" | "implicit" | "bare_claim",
      "category": "required" | "bonus"
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

def build_questions_prompt(
    cv: str,
    covered_skills: list[dict],
    skill_gaps: list[dict],
    job_title: str,
) -> str:
    skills_json = json.dumps(covered_skills, indent=2)
    required_gaps = [g for g in skill_gaps if g.get("category") == "required"]
    gaps_json = json.dumps(required_gaps, indent=2)

    return f"""You are an experienced technical interviewer preparing a candidate for a {job_title} interview.

Below is the candidate's CV, a list of skills they demonstrably have that are relevant to the job,
and a list of required skills missing from their CV.

CV:
{cv}

COVERED SKILLS:
{skills_json}

REQUIRED SKILL GAPS:
{gaps_json}

Follow these steps:

STEP 1 — Select 4–6 questions from covered skills.
Prioritise skills where category is "required" over "bonus".
Within the same category, prefer richer evidence (explicit > implicit > bare_claim).

STEP 2 — Select 1–2 questions from required skill gaps.
Only include gaps that are central to the role — an interviewer would not probe every gap,
only the ones they cannot overlook.

STEP 3 — For each selected covered skill, generate one question and one follow-up.
Calibrate to evidence_type:

  · explicit    — behavioural question anchored to the specific project or role mentioned.
                  Example style: "Tell me about a time you..."
  · implicit    — confirming question that surfaces implied knowledge without assuming mastery.
                  Example style: "Your experience with X suggests familiarity with Y —
                  how would you describe your comfort level with...?"
  · bare_claim  — foundational question probing basic understanding.
                  The follow_up should coach: what concrete project or context would
                  strengthen this entry on their CV?

STEP 4 — For each selected skill gap, generate one open, non-threatening question.
Acknowledge the skill is not in the CV without putting the candidate on the spot.
  Example style: "We use X fairly heavily — what's your exposure to...?"
                 "Have you had a chance to work with Y in any capacity?"
The follow_up should invite the candidate to draw parallels to what they do know.
Set cv_anchor to null for gap questions.

STEP 5 — For covered skill questions, write the cv_anchor.
Quote or closely paraphrase the specific passage from the CV this question is reacting to.
Keep it to one sentence. Do not invent passages not in the CV.

Respond ONLY with a JSON array — no markdown, no explanation:

[
  {{
    "skill": "<canonical skill label>",
    "question": "<interview question>",
    "follow_up": "<follow-up question or coaching note>",
    "cv_anchor": "<specific CV passage, or null for gap questions>"
  }}
]"""