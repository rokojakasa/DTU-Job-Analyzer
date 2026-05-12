# Job Analyzer for DTU Students

## Overview

A locally-run web service that helps DTU students prepare for job applications. Upload your CV and a job posting, and the service will identify skill gaps, recommend relevant DTU courses to address them, and generate tailored interview questions.

The service holds a single CV and job posting in memory at a time. Uploading a new CV or job posting overwrites the previous one and invalidates any cached analysis.

**Important note**: The service was evaluated on jobs and CVs for someone studying at DTU Compute department, as those were the ones I was familiar with and could properly evaluate.

---

## System Architecture

The pipeline has five stages:

```
User uploads CV + Job Posting
        │
        ▼
┌─────────────────────┐
│  PDF Parser         │  pdfplumber extracts text from uploaded PDFs
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Analyser (LLM)     │  Two-pass LLM pipeline via CampusAI
│                     │
│  Pass 1: Intent     │  Extracts structured requirements from job posting
│  Pass 2: Gap        │  Matches CV against requirements, classifies evidence
└────────┬────────────┘
         │
         ├──────────────────────────────────┐
         ▼                                  ▼
┌─────────────────────┐        ┌────────────────────────┐
│  Course Retriever   │        │  Question Generator    │
│                     │        │                        │
│  Query expansion    │        │  LLM generates         │
│  (LLM rephrases     │        │  interview questions   │
│  skill gaps into    │        │  anchored to CV        │
│  academic terms)    │        │  passages              │
│                     │        └────────────────────────┘
│  Hybrid retrieval   │
│  BM25 + embeddings  │
│  over DTU course    │
│  objectives         │
│                     │
│  Aggregator         │
│  (max pooling per   │
│  course code)       │
└─────────────────────┘
```

**Key design decisions:**

- The analyser uses a two-pass LLM approach: pass 1 extracts structured requirements from the raw job posting (isolating required vs. bonus skills, filtering out soft traits and degree requirements); pass 2 performs the actual gap analysis against the CV. Separating these concerns significantly improves analysis quality.
- Course retrieval uses hybrid retrieval (BM25 + embedding similarity) over per-objective chunks of the DTU course catalogue. Each learning objective is embedded separately so that a course with one highly relevant objective ranks above a course with many mediocre ones.
- Query expansion rephrases industry-specific skill gap labels into academic vocabulary before retrieval, bridging the gap between job posting language ("build LLM pipelines") and course catalogue language ("natural language processing and neural networks").

---

## Tools and Technologies

| Layer | Tool |
|---|---|
| Web framework | FastAPI (Python) |
| LLM integration | CampusAI API (OpenAI-compatible) |
| PDF parsing | `pdfplumber` |
| DTU course matching | Nomic Embed + BM25 (hybrid) |
| Vector similarity | `numpy` |
| Containerisation | Docker + `docker-compose` |
| Testing | `pytest` + `httpx` |

---

## Dataset

**DTU course catalogue** — stored at `data/dtu_courses.jsonl`. Each line is a JSON object representing one course with the following fields:

```json
{
  "course_code": "02450",
  "title": "Introduction to Machine Learning",
  "fields": { "Point( ECTS )": 5 },
  "learning_objectives": [
    "apply supervised learning algorithms to classification tasks",
    "..."
  ]
}
```

At startup, the service explodes each course into one chunk per learning objective, embeds all chunks via the CampusAI embeddings API, and saves the result to `data/index.npz`. On subsequent startups, the cache is loaded directly (under one second, no API calls) unless the source file is newer.

The course catalogue is included in the repository at `data/dtu_courses.jsonl`. However, the `index.npz` needs to be built on startup, because it is ~43 MB. 

---

## API Endpoints

### `POST /cv/text`
Upload a CV as plain text via form data.

**Request:** `multipart/form-data` with field `text`.

**Response:**
```json
{ "message": "CV uploaded successfully (1,842 characters).", "character_count": 1842 }
```

---

### `POST /cv/pdf`
Upload a CV as a PDF file. Text is extracted automatically.

**Request:** `multipart/form-data` with field `file` (PDF).

**Response:** Same as `/cv/text`.

---

### `POST /job-posting/text`
Upload a job posting as plain text via form data.

**Request:** `multipart/form-data` with field `text`.

**Response:**
```json
{
  "message": "Job posting uploaded successfully (743 characters).",
  "source": "text",
  "character_count": 743
}
```

> **Note:** LinkedIn job posts often require login and block automated fetching. Paste the text directly.

---

### `GET /analyse`
Run the two-pass LLM analysis. Extracts structured requirements from the job posting, then classifies each requirement against the CV as covered or a gap. Results are stored in memory and used by `/courses` and `/questions`.

Requires both a CV and job posting to have been uploaded.

**Response:**
```json
{
  "job_title_inferred": "Machine Learning Engineer",
  "covered_skills": [
    {
      "label": "apply machine learning",
      "job_posting_label": "machine learning frameworks (scikit-learn, PyTorch)",
      "evidence_from_cv": "Random forest classifier project at Acme Corp using scikit-learn.",
      "evidence_type": "explicit",
      "category": "required"
    }
  ],
  "skill_gaps": [
    {
      "label": "deploy applications on cloud infrastructure",
      "job_posting_label": "cloud platforms (AWS, GCP)",
      "category": "required"
    }
  ],
  "summary": "Your CV covers 4 of 5 required skills. Main gap is cloud deployment."
}
```

`evidence_type` is one of:
- `explicit` — skill named and tied to a specific project, role, or course
- `implicit` — not named, but project work or coursework implies genuine exposure
- `bare_claim` — listed in a skills section with no supporting context elsewhere in the CV

---

### `POST /courses`
For each skill gap from `/analyse`, recommend DTU courses ranked by relevance. Uses hybrid BM25 + embedding retrieval over course learning objectives, with LLM-based query expansion to bridge industry and academic vocabulary.

Requires `/analyse` to have been called first.

**Query parameters:**
- `retrieval_mode`: `hybrid` (default), `sparse` (BM25 only), or `dense` (embeddings only)

**Response:**
```json
{
  "recommendations": [
    {
      "skill_gap": "deploy applications on cloud infrastructure",
      "courses": [
        { "course_number": "02613", "title": "High-Performance Computing", "ects": 5 },
        { "course_number": "02807", "title": "Computational Tools for Data Science", "ects": 5 }
      ]
    }
  ]
}
```

---

### `GET /questions`
Generate tailored interview questions based on the analysis. Questions cover skills the candidate demonstrates that are relevant to the job (anchored to specific CV passages), plus a small number of probing questions on the most important gaps.

Requires `/analyse` to have been called first.

**Response:**
```json
{
  "questions": [
    {
      "skill": "apply machine learning",
      "question": "Walk me through the random forest project — how did you decide on that model?",
      "follow_up": "How did you validate that it generalised beyond your training set?",
      "cv_anchor": "Random forest classifier for churn prediction at Acme Corp."
    },
    {
      "skill": "deploy applications on cloud infrastructure",
      "question": "What's your exposure to cloud platforms like AWS or GCP?",
      "follow_up": "Have you had a chance to deploy anything beyond a local environment?",
      "cv_anchor": null
    }
  ]
}
```

---

## How to Run

### Prerequisites

- Docker and `docker-compose`
- A CampusAI API key
- The DTU course catalogue at `data/dtu_courses.jsonl`

### Configuration

Create a `.env` file in your home directory (outside the repository):

```bash
# ~/.env
CAMPUSAI_API_KEY=your_key_here
CAMPUSAI_API_URL=https://api.campusai.example.com/v1
CAMPUSAI_MODEL=google/gemma-4-26b-a4b
CAMPUSAI_EMBED_MODEL=Nomic Embed Text
```

The application loads this file via `python-dotenv` on startup.

### Running with Docker

```bash
# Build and start the service
docker-compose up --build

# The API is available at http://localhost:8000
# Interactive docs at http://localhost:8000/docs
```

### Running locally (development)

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### First run

On first startup, the service will embed all DTU course objectives via the CampusAI API and save the index to `data/index.npz`. This takes a few minutes. Subsequent startups load from the cache and are fast.

To force a full rebuild of the index:

```python
# In the Python REPL or a startup script
from app.services.courses.index import build_index
import asyncio
asyncio.run(build_index(force_rebuild=True))
```

### Frontend

`index.html` in project root can be used as simple visualization for user interaction once FastAPI service is up and running

---


## Tests

```bash
# Unit tests (no API calls, runs in CI)
pytest

# Evaluation tests (calls real CampusAI API — requires credentials)
pytest --run-eval
```

### Test structure

| File | Type | What it covers |
|---|---|---|
| `test_store.py` | Unit | In-memory store state machine — upload invalidation, readiness flags |
| `test_aggregator.py` | Unit | Course aggregation and deduplication — max pooling, top-k, ordering |
| `test_analyser.py` | Unit | `_parse_json()` helper — markdown fence stripping, error handling |
| `test_routes.py` | Integration | HTTP contract — status codes, guard dependencies, mocked LLM responses |
| `test_analyser_eval.py` | Eval | LLM analysis quality — skill detection, hallucination guards |
| `test_questions_eval.py` | Eval | LLM question generation — count, anchoring, field completeness |

---

## Evaluation

### Analyser quality

The eval suite (`test_analyser_eval.py`) runs a fixed CV (MSc student, 2 years at Vestas, strong Python/ML profile) against a real job posting (Ørsted student assistant in analytics). It asserts:

| Test | What it checks |
|---|---|
| `test_python_is_covered` | Python must appear in covered skills — it is explicit throughout the CV |
| `test_machine_learning_is_covered` | ML must be covered — evidenced by 02450, scikit-learn work, and forecasting project |
| `test_testable_code_is_covered` | pytest + PR reviews must map to "maintainable and testable code" |
| `test_cloud_is_a_required_gap` | Cloud deployment is required and absent — must appear as a required gap |
| `test_cloud_is_not_hallucinated_as_covered` | HPC course must not be misidentified as cloud deployment coverage |
| `test_docker_kubernetes_is_a_bonus_gap` | Docker/Kubernetes is bonus and absent — must appear as bonus gap |
| `test_no_hallucinated_skills` | Every covered skill must have non-empty evidence — guards against hallucination |

All tests pass consistently. The two-pass LLM design (separate requirement extraction and gap analysis) was introduced specifically after observing that single-pass prompts frequently misclassified bonus skills as required and struggled to distinguish implicit from bare-claim evidence.

### Question generator quality

The eval suite (`test_questions_eval.py`) injects a fixed `AnalysisResponse` (so the variable is only the question generation, not the analysis) and asserts:

| Test | What it checks |
|---|---|
| `test_question_count_in_range` | 3–10 total questions (prompt asks for 4–6 covered + 1–2 gap) |
| `test_each_question_has_required_fields` | All fields present; `cv_anchor` allowed null for gap questions |
| `test_questions_are_not_too_long` | No question exceeds 400 characters (enforces "one sentence" instruction) |
| `test_gap_questions_have_null_cv_anchor` | Gap questions must not hallucinate CV anchors |
| `test_covered_skill_questions_have_cv_anchor` | At least 2 covered-skill questions must be anchored to the CV |

The test suite validates structure and basic behavioural constraints, but a full qualitative evaluation is difficult without a large sample of real interviews for comparison. From limited observation, real technical interviews tend to be more conversational — one topic leads naturally to another — rather than systematically working through a CV against a job posting. The current design generates questions anchored to specific CV passages, which is useful preparation material but may not reflect how an interview actually unfolds.

### Retrieval quality (qualitative)

Hybrid retrieval (BM25 + embeddings) with LLM query expansion outperforms either method alone on the skill gaps most likely to mismatch in vocabulary between job postings and course catalogues. For example:

- "build LLM pipelines" → query expansion produces "large language model pipelines", "natural language processing", "machine learning and neural networks" → correctly retrieves 02456 Deep Learning and 02807 Computational Tools for Data Science
- "manage MS SQL" → expansion to "relational database management" → correctly retrieves 02170 Database Systems

Without query expansion, BM25 fails on these cases entirely (no token overlap) and dense retrieval alone sometimes drifts toward loosely related courses.

The main ceiling on retrieval quality is the dataset itself. DTU's course catalogue does not explicitly cover certain industry or very niche tools, so no retrieval method can recommend a course that doesn't exist in the data. 