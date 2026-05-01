# Job Analyzer for DTU Students

## Overview

A locally-run web service that helps DTU students prepare for job applications. Upload your CV and a job posting, then use the analysis endpoints to identify skill gaps, get DTU course recommendations, and generate tailored interview questions.

The service holds a single CV and job posting in memory at a time. Uploading a new CV or job posting overwrites the previous one.

Skill extraction uses ESCO as a shared vocabulary for normalising skills from both the CV and the job posting. This will be evaluated during development — if plain embedding similarity proves equally effective without it, the ESCO integration may be simplified or dropped.

If a grade transcript is also provided, course recommendations are split into:
- Courses already completed but missing from the CV → **"Add this to your CV"**
- Courses not yet taken → **"Consider taking this"**

---


## Tools and Technologies

| Layer | Tool |
|---|---|
| Web framework | FastAPI (Python) |
| LLM integration | CampusAI API |
| PDF parsing | `pdfplumber` or `PyMuPDF` |
| DTU course matching | Nomic Embed and/or BM25 |
| Vector similarity | `numpy` |
| ESCO skills | Local CSV or SPARQL server |
| Containerisation | Docker + `docker-compose` |
| Testing | `pytest` + `httpx` |

---

## Data Sources

- **CV** — uploaded as plain text or PDF; stored in memory until overwritten
- **Job posting** — uploaded as plain text, or optionally fetched from a URL; overwrites previous on each upload
- **Grade transcript** — optional PDF, provided directly to `/courses` when needed
- **DTU course catalogue** — cached locally as JSON
- **ESCO skills** — local CSV or self-hosted SPARQL server

---

## API Endpoints

### `POST /cv`
Upload your CV. Overwrites any previously uploaded CV.

**Request:** Plain text body, or multipart form with a PDF file.

---

### `POST /job-posting`
Upload a job posting. Overwrites the previous one.

**Request (plain text):**
```json
{ "text": "We are looking for a machine learning engineer..." }
```

**Request (URL fetch):**
```json
{ "url": "https://example.com/jobs/ml-engineer" }
```

> **Note:** LinkedIn job posts often require login and block automated fetching. Use plain text upload for those.

---

### `GET /analyse`
Compare the uploaded CV against the uploaded job posting. Extracts skills from both, identifies what is covered and what is missing, and returns a plain-language summary.

Both a CV and a job posting must be uploaded before calling this endpoint. Results are stored in memory and used as input by `/courses` and `/questions`.

**Response:**
```json
{
  "job_title_inferred": "Machine Learning Engineer",
  "covered_skills": [
    {
      "label": "apply machine learning",
      "evidence_from_cv": "Implemented a random forest classifier for..."
    }
  ],
  "skill_gaps": [
    {
      "label": "design database schema"
    }
  ],
  "coverage_score": 0.61,
  "summary": "Your CV covers 8 of 13 required skills. Main gaps are in database design and cloud deployment."
}
```

---

### `POST /courses`
For each skill gap identified by `/analyse`, suggest DTU courses that cover it, ranked by embedding similarity. Requires `/analyse` to have been called first.

Optionally accepts a grade transcript PDF. If provided, each course recommendation includes an `action` field indicating whether the student should add the course to their CV or consider taking it.

**Request:** Optionally, a multipart form with `transcript_file` (PDF). If omitted, no `action` field is returned.

**Response (without transcript):**
```json
{
  "recommendations": [
    {
      "skill_gap": "design database schema",
      "courses": [
        {
          "course_number": "02170",
          "title": "Database Systems",
          "ects": 5
        }
      ]
    }
  ]
}
```

**Response (with transcript):**
```json
{
  "recommendations": [
    {
      "skill_gap": "design database schema",
      "courses": [
        {
          "course_number": "02170",
          "title": "Database Systems",
          "ects": 5,
          "action": "add_to_cv"
        }
      ]
    },
    {
      "skill_gap": "apply cloud infrastructure",
      "courses": [
        {
          "course_number": "02613",
          "title": "High-Performance Computing",
          "ects": 5,
          "action": "consider_taking"
        }
      ]
    }
  ]
}
```

---

### `GET /questions`
Generate tailored interview questions based on the intersection of the student's CV and the job posting — only for skills the student already demonstrates *and* that are relevant to the job. Skills present in the CV but unrelated to the job are excluded.

Requires `/analyse` to have been called first. Each question is anchored to a specific passage in the CV so the student knows what the interviewer is likely reacting to.

**Response:**
```json
{
  "questions": [
    {
      "skill": "apply machine learning",
      "question": "Walk me through a project where you selected and tuned a model. What drove your choice?",
      "follow_up": "How did you validate that it generalised beyond your training set?",
      "cv_anchor": "Random forest classifier project mentioned in CV"
    }
  ]
}
```
