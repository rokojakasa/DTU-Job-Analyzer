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
produce a skills gap analysis. All output fields must be in English regardless
of the input language.

CV:
{cv}

JOB REQUIREMENTS:
{job_requirements}

---

STEP 1 — For each skill in required_skills and bonus_skills, read the entire CV
before making a classification. Place each skill in exactly one bucket:

  covered_skills — any evidence exists, even indirect.
  skill_gaps     — no evidence of this skill anywhere in the CV.

STEP 2 — For each covered skill, assign an evidence_type.

  Before assigning bare_claim, re-read the full CV. A skill listed only in a
  skills section is only bare_claim if the rest of the CV — projects, roles,
  coursework — offers no context that would naturally involve it. If supporting
  context exists anywhere, classify as implicit instead.

  · "explicit"   — skill is named and tied to a specific project, role, or course.
  · "implicit"   — skill is not named, but the candidate's project work, coursework,
                   or job history makes it implausible they listed it without genuine
                   exposure. Apply domain knowledge: ask yourself whether someone doing
                   what this candidate did would inevitably have used this skill.
  · "bare_claim" — skill appears in a skills list and nothing else in the CV
                   supports or implies it.

STEP 3 — Apply equivalence when assessing coverage.
If the CV demonstrates a technology that is functionally equivalent to a required
skill, mark that skill as covered with evidence_type "implicit". Use your domain
knowledge to judge equivalence — do not require an exact name match.

STEP 4 — Assign category to each skill.
  · "required" for skills from required_skills
  · "bonus"    for skills from bonus_skills

STEP 5 — Produce canonical labels.
  · "label": verb-noun form ("manage relational databases", not "MS SQL")
  · "job_posting_label": the skill exactly as listed in required_skills or bonus_skills

STEP 6 — Write a 1–2 sentence summary for the candidate.
  · Name any bare_claim skills and suggest adding supporting context.
  · Note if remaining gaps are bonus rather than required.

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

    return f"""You are an experienced technical interviewer preparing questions for a {job_title} candidate.

CV:
{cv}

COVERED SKILLS (skills the candidate demonstrates):
{skills_json}

REQUIRED SKILL GAPS (required skills absent from the CV):
{gaps_json}

---

STEP 1 — Filter gaps before selecting.
Discard any gap that is a certification, training programme, curriculum requirement,
or formal qualification rather than a demonstrable technical skill.
Examples of what to discard: "complete X programme", "hold Y certificate", "follow Z grundforløb".
Only proceed with gaps that describe something a candidate could demonstrate in a conversation.

STEP 2 — Select 4–6 questions from covered skills.
Prioritise required over bonus. Within the same category, prefer richer evidence
(explicit > implicit > bare_claim).

STEP 3 — Select 1–2 questions from the filtered gaps.
Only include gaps central to the role — pick the ones an interviewer cannot overlook.
If no gaps survive the Step 1 filter, return only covered-skill questions.

STEP 4 — Write each question.
Keep every question to one sentence. Write as you would actually say it in a room,
not as you would write it in a document. Do not repeat or paraphrase the CV back
in the question itself — the cv_anchor field carries that context separately.

Calibrate to evidence_type for covered skills:
  · explicit    — behavioural, anchored to a project or role.
                  Style: "Walk me through...", "Tell me about a time..."
  · implicit    — confirming, surfaces implied knowledge without assuming mastery.
                  Style: "How comfortable are you with X given your work on Y?"
  · bare_claim  — foundational, probes basic understanding.
                  The follow_up should coach: what project would strengthen this CV entry?

For gap questions, keep the tone open and non-threatening.
Style: "What's your exposure to X?", "Have you had a chance to work with Y?"
The follow_up should invite the candidate to draw on what they do know.
Set cv_anchor to null for gap questions.

STEP 5 — Write cv_anchor for covered-skill questions only.
One sentence. Quote or closely paraphrase the specific CV passage this question reacts to.
Do not invent passages not in the CV.

Respond ONLY with a JSON array — no markdown, no explanation:

[
  {{
    "skill": "<canonical skill label>",
    "question": "<one sentence, conversational>",
    "follow_up": "<one sentence>",
    "cv_anchor": "<specific CV passage, or null>"
  }}
]"""