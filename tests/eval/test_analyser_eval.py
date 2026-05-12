# tests/eval/test_analyser_eval.py
"""
Evaluation tests for the LLM-backed analysis pipeline.

These tests call the real CampusAI API and assert on the *shape and
quality* of the output — not exact values, since LLMs are non-deterministic.

Run with:  pytest --run-eval
Skipped by default in CI.

Scenarios
---------
1. Ørsted Analytics (Student Assistant)       — Python/ML, stochastic modelling
2. Lunar (Backend Software Engineer)          — Java/Spring Boot, REST APIs
3. Novo Nordisk (Data Analyst)                — SQL, Power BI, Python
4. DTU Bioinformatics (Research Assistant)    — Python, RNA-seq, NGS pipelines

Each scenario runs run_analysis() exactly once via a session-scoped fixture.
All tests in that scenario share the cached result — no redundant API calls.
"""

from __future__ import annotations

import pytest
from app.models import AnalysisResponse, EvidenceType, SkillCategory
from app.services.analyser import run_analysis


# ===========================================================================
# Scenario 1 — Ørsted Analytics (Student Assistant)
# Candidate: Python/ML, stochastic modelling, pytest; no cloud deployment
# Job: requires Python, ML, testable code, cloud (Azure/AWS); bonus Docker/K8s
# ===========================================================================

_ORSTED_JOB_POSTING = """
Student Assistant — Advanced Analytics
Ørsted, Copenhagen

You'll be part of Advanced Analytics where you, together with your colleagues,
will develop forecasts, analyses, and dashboards which provide our trading floor
with a strong basis for data and statistical driven decision making in the daily
operation at our Danish power stations, wind turbines, solar PV, electric boilers,
heat pumps, and P2X.

You'll play an important role in:
- Developing and implementing performance tracking
- Developing dashboards to visualise decision support and performance tracking
- Developing and implementing forecasting models using AI/ML/DL, including
  stochastic modelling and statistical data analysis
- Communicating and collaborating with internal and external stakeholders,
  i.e., team members, traders, etc.

To succeed in the role, you:
- Are currently studying data science, applied mathematics, or another relevant
  field within analytics and/or engineering
- Are a strong Python programmer
- Have experience with writing and reviewing maintainable and testable code
- Are familiar with machine-learning concepts (supervised and unsupervised learning)
  and data analytics
- Have experience with cloud platforms (Azure or AWS) for deploying data pipelines

Bonus:
- Knowledge of Git, Docker & Kubernetes
- Familiarity with electricity markets or energy systems
- Experience with time series forecasting or stochastic modelling
"""

_ORSTED_CV = """
Mikkel Andersen
MSc student, Mathematical Modelling and Computation, DTU
mikkel.andersen@student.dtu.dk | github.com/mikkelands

EDUCATION
Technical University of Denmark (DTU), 2022–present
MSc Mathematical Modelling and Computation (expected graduation 2025)

Relevant courses:
- 02450 Introduction to Machine Learning (grade: 10)
- 02457 Non-Linear Signal Processing (grade: 12)
- 02418 Statistical Modelling (grade: 10)
- 02613 High-Performance Computing (grade: 7)
- 02807 Computational Tools for Data Science (grade: 10)

BSc Mathematics and Technology, DTU, 2019–2022 (grade average: 9.2)

EXPERIENCE

Student Assistant — Data & AI, Vestas Wind Systems (Jun 2023–present)
Working in a small analytics team supporting wind turbine performance monitoring.
- Built and maintained Python pipelines ingesting SCADA telemetry data from 200+
  turbines into an internal data warehouse.
- Developed a regression-based power curve model using scikit-learn to flag
  underperforming turbines; reduced manual inspection workload by ~30%.
- Wrote unit tests with pytest; all code reviewed via pull requests on GitHub.
- Delivered a Jupyter-based dashboard prototype to operations team using Plotly.

Project: Probabilistic Load Forecasting (02450 course project, 2023)
- Trained LSTM and gradient boosting models on hourly Danish electricity consumption
  data from Energinet.
- Compared point forecasts vs. quantile regression for uncertainty estimation.
- Implemented cross-validation pipeline in Python (pandas, scikit-learn, PyTorch).

Project: Monte Carlo Methods for Option Pricing (02457, 2022)
- Implemented stochastic differential equation solvers (Euler-Maruyama) in Python
  and NumPy to price European and Asian options.
- Benchmarked convergence rates across variance reduction techniques.

SKILLS
Languages: Python (primary), R (familiar), MATLAB (coursework)
Libraries: scikit-learn, PyTorch, pandas, NumPy, Plotly, pytest
Tools: Git, Linux, SQL (PostgreSQL)
"""


@pytest.fixture(scope="session")
async def orsted_result() -> AnalysisResponse:
    return await run_analysis(_ORSTED_CV, _ORSTED_JOB_POSTING)


def _orsted_covered(result: AnalysisResponse) -> list[str]:
    return [s.label.lower() for s in result.covered_skills]

def _orsted_required_gaps(result: AnalysisResponse) -> list[str]:
    return [g.label.lower() for g in result.skill_gaps if g.category == SkillCategory.REQUIRED]

def _orsted_bonus_gaps(result: AnalysisResponse) -> list[str]:
    return [g.label.lower() for g in result.skill_gaps if g.category == SkillCategory.BONUS]


# Shape and type tests

@pytest.mark.eval
async def test_orsted_returns_analysis_response_type(orsted_result):
    assert isinstance(orsted_result, AnalysisResponse)

@pytest.mark.eval
async def test_orsted_job_title_is_non_empty(orsted_result):
    assert orsted_result.job_title_inferred.strip() != ""

@pytest.mark.eval
async def test_orsted_covered_skills_and_gaps_are_non_empty(orsted_result):
    assert len(orsted_result.covered_skills) > 0
    assert len(orsted_result.skill_gaps) > 0

@pytest.mark.eval
async def test_orsted_all_evidence_types_are_valid(orsted_result):
    valid = {e.value for e in EvidenceType}
    for skill in orsted_result.covered_skills:
        assert skill.evidence_type.value in valid, (
            f"Invalid evidence_type {skill.evidence_type!r} for skill {skill.label!r}"
        )

@pytest.mark.eval
async def test_orsted_all_categories_are_valid(orsted_result):
    valid = {c.value for c in SkillCategory}
    for skill in orsted_result.covered_skills:
        assert skill.category.value in valid
    for gap in orsted_result.skill_gaps:
        assert gap.category.value in valid

@pytest.mark.eval
async def test_orsted_summary_is_non_empty(orsted_result):
    assert orsted_result.summary.strip() != ""


# Coverage quality

@pytest.mark.eval
async def test_orsted_python_is_covered(orsted_result):
    """Python is explicit throughout the CV — must be covered."""
    assert any("python" in l for l in _orsted_covered(orsted_result)), (
        f"Expected Python to be covered. Covered: {_orsted_covered(orsted_result)}"
    )

@pytest.mark.eval
async def test_orsted_machine_learning_is_covered(orsted_result):
    """ML is covered by 02450, scikit-learn at Vestas, and the forecasting project."""
    ml_keywords = {"machine learning", "supervised", "ml", "forecasting model"}
    assert any(
        any(kw in l for kw in ml_keywords)
        for l in _orsted_covered(orsted_result)
    ), f"Expected ML to be covered. Covered: {_orsted_covered(orsted_result)}"

@pytest.mark.eval
async def test_orsted_testable_code_is_covered(orsted_result):
    """CV explicitly mentions pytest and pull request reviews."""
    testing_keywords = {"test", "maintainable", "code quality", "review"}
    assert any(
        any(kw in l for kw in testing_keywords)
        for l in _orsted_covered(orsted_result)
    ), f"Expected testable code to be covered. Covered: {_orsted_covered(orsted_result)}"

@pytest.mark.eval
async def test_orsted_cloud_is_a_required_gap(orsted_result):
    """Cloud deployment is required and absent from the CV entirely."""
    cloud_keywords = {"cloud", "azure", "aws", "deploy"}
    assert any(
        any(kw in l for kw in cloud_keywords)
        for l in _orsted_required_gaps(orsted_result)
    ), (
        f"Expected cloud deployment to be a required gap.\n"
        f"Required gaps: {_orsted_required_gaps(orsted_result)}\n"
        f"Covered: {_orsted_covered(orsted_result)}"
    )

@pytest.mark.eval
async def test_orsted_cloud_is_not_hallucinated_as_covered(orsted_result):
    """Guards against the LLM confusing the HPC course with cloud deployment."""
    cloud_keywords = {"cloud", "azure", "aws"}
    assert not any(
        any(kw in l for kw in cloud_keywords)
        for l in _orsted_covered(orsted_result)
    ), (
        f"Cloud should not be covered — HPC course is not cloud deployment.\n"
        f"Covered: {_orsted_covered(orsted_result)}"
    )

@pytest.mark.eval
async def test_orsted_docker_kubernetes_is_a_bonus_gap(orsted_result):
    """Docker/Kubernetes is listed as bonus and absent from the CV."""
    container_keywords = {"docker", "kubernetes", "container"}
    assert any(
        any(kw in l for kw in container_keywords)
        for l in _orsted_bonus_gaps(orsted_result)
    ), (
        f"Expected Docker/Kubernetes as a bonus gap.\n"
        f"Bonus gaps: {_orsted_bonus_gaps(orsted_result)}"
    )

@pytest.mark.eval
async def test_orsted_no_hallucinated_skills(orsted_result):
    """Every covered skill must have non-empty evidence_from_cv."""
    for skill in orsted_result.covered_skills:
        assert skill.evidence_from_cv.strip() != "", (
            f"Skill {skill.label!r} has empty evidence_from_cv — possible hallucination."
        )


# ===========================================================================
# Scenario 2 — Lunar (Backend Software Engineer)
# Candidate: Java/Spring Boot, PostgreSQL, REST APIs, JUnit; no Go, no K8s
# Job: requires Java, SQL, REST APIs, testing; bonus Go and Kubernetes
# ===========================================================================

_SE_JOB_POSTING = """
Backend Software Engineer — Platform Team
Lunar, Copenhagen

We're looking for a backend engineer to join our platform team, building the
services that power Lunar's banking app.

You will:
- Design and implement scalable REST APIs consumed by mobile clients
- Own microservices from design through deployment
- Write and maintain SQL schemas and data migrations
- Participate in code reviews and help set technical standards for the team

We expect you to:
- Have solid experience with Java and the Spring Boot framework
- Be comfortable writing SQL against relational databases (PostgreSQL or MySQL)
- Understand REST API design principles (versioning, error handling, pagination)
- Have experience with unit and integration testing (JUnit or equivalent)

Nice to have:
- Experience with Go
- Familiarity with Kubernetes for container orchestration
- Experience with event-driven architectures (Kafka or RabbitMQ)
"""

_SE_CV = """
Lars Petersen
BSc Software Engineering, DTU, 2021
lars.petersen@gmail.com | github.com/larsp

EXPERIENCE

Backend Developer, Netcompany (2021–present)
- Developed and maintained REST APIs in Java 17 with Spring Boot for a large
  Danish public-sector client; APIs serve ~500k requests/day.
- Designed PostgreSQL schemas and wrote Flyway migrations for a document
  management system with 40+ tables.
- Wrote unit tests with JUnit 5 and integration tests with Spring Boot Test;
  maintained >85% branch coverage on core modules.
- Reviewed pull requests daily; co-authored team coding standards document.

Side project: Personal finance tracker (2023)
- Built a small Spring Boot REST API backed by SQLite; deployed on a VPS
  using Docker Compose.
- Wrote a basic OpenAPI spec and versioned the API with URI prefixes (/v1/).

SKILLS
Languages: Java (primary), Python (scripting), SQL
Frameworks: Spring Boot, Spring Data JPA, Hibernate
Tools: PostgreSQL, Git, Docker, Maven, IntelliJ
Testing: JUnit 5, Mockito, Spring Boot Test
"""


@pytest.fixture(scope="session")
async def se_result() -> AnalysisResponse:
    return await run_analysis(_SE_CV, _SE_JOB_POSTING)


def _se_covered(result: AnalysisResponse) -> list[str]:
    return [s.label.lower() for s in result.covered_skills]

def _se_bonus_gaps(result: AnalysisResponse) -> list[str]:
    return [g.label.lower() for g in result.skill_gaps if g.category == SkillCategory.BONUS]


@pytest.mark.eval
async def test_se_java_spring_is_covered(se_result):
    """Java/Spring Boot is explicit throughout the CV — must be covered."""
    assert any("java" in l or "spring" in l for l in _se_covered(se_result)), (
        f"Expected Java/Spring Boot to be covered. Covered: {_se_covered(se_result)}"
    )

@pytest.mark.eval
async def test_se_sql_is_covered(se_result):
    """PostgreSQL schemas and Flyway migrations are explicit evidence."""
    assert any("sql" in l or "database" in l or "postgresql" in l for l in _se_covered(se_result)), (
        f"Expected SQL/database work to be covered. Covered: {_se_covered(se_result)}"
    )

@pytest.mark.eval
async def test_se_rest_api_is_covered(se_result):
    """REST APIs built at Netcompany and in the side project — clearly covered."""
    assert any("rest" in l or "api" in l for l in _se_covered(se_result)), (
        f"Expected REST API design to be covered. Covered: {_se_covered(se_result)}"
    )

@pytest.mark.eval
async def test_se_testing_is_covered(se_result):
    """JUnit 5 + integration tests with >85% coverage — should not be a gap."""
    assert any("test" in l for l in _se_covered(se_result)), (
        f"Expected testing to be covered. Covered: {_se_covered(se_result)}"
    )

@pytest.mark.eval
async def test_se_go_is_a_bonus_gap(se_result):
    """Go is listed as nice-to-have and absent from the CV — must be a bonus gap."""
    assert any("go" in l for l in _se_bonus_gaps(se_result)), (
        f"Expected Go to be a bonus gap. Bonus gaps: {_se_bonus_gaps(se_result)}"
    )

@pytest.mark.eval
async def test_se_kubernetes_is_a_bonus_gap(se_result):
    """Kubernetes is explicitly nice-to-have — must not be labelled required."""
    assert any("kubernetes" in l or "orchestrat" in l for l in _se_bonus_gaps(se_result)), (
        f"Expected Kubernetes as bonus gap. Bonus gaps: {_se_bonus_gaps(se_result)}"
    )

@pytest.mark.eval
async def test_se_go_not_hallucinated_as_covered(se_result):
    """
    Go does not appear anywhere in the CV.
    Guards against the LLM inferring Go from general backend/Java experience.
    """
    assert not any("go" in l.split() for l in _se_covered(se_result)), (
        f"Go should not be covered — it is absent from the CV. Covered: {_se_covered(se_result)}"
    )


# ===========================================================================
# Scenario 3 — Novo Nordisk (Data Analyst)
# Candidate: SQL/Snowflake, Power BI, Python/pandas; no dbt, no Spark
# Job: requires SQL, BI tooling, Python; bonus dbt and Spark
# ===========================================================================

_DA_JOB_POSTING = """
Data Analyst — Commercial Intelligence
Novo Nordisk, Søborg

You will join our Commercial Intelligence team, providing data-driven insights
to sales and marketing leadership across European markets.

Your responsibilities:
- Write and maintain complex SQL queries against our enterprise data warehouse
- Build and maintain dashboards and reports in Power BI or Tableau
- Conduct ad-hoc analysis in Python (pandas, matplotlib) to support business decisions
- Collaborate with data engineers to define data requirements and ensure data quality
- Present findings to non-technical stakeholders

Requirements:
- Strong SQL skills — able to write window functions, CTEs, and optimise queries
- Experience with a BI tool (Power BI or Tableau)
- Python proficiency for data wrangling and visualisation (pandas, matplotlib or seaborn)
- Ability to translate business questions into analytical tasks

Bonus:
- Experience with dbt for data transformation
- Familiarity with Spark or large-scale distributed data processing
- Experience working with CRM data (Salesforce or similar)
"""

_DA_CV = """
Sofie Madsen
MSc Business Analytics, CBS, 2022
sofie.madsen@outlook.com

EXPERIENCE

Data Analyst, Coloplast (Jul 2022–present)
- Write SQL queries daily against a Snowflake data warehouse; regularly use
  window functions (ROW_NUMBER, LAG, LEAD) and CTEs for cohort analysis.
- Built and maintain 12 Power BI dashboards consumed by the commercial team;
  introduced row-level security to segment views by sales region.
- Developed a Python script (pandas + seaborn) that automates a weekly churn
  report, saving ~4 hours of manual Excel work per week.
- Presented quarterly analysis to VP of Sales; distilled findings into
  executive-level slide decks.

Student project: Customer Segmentation (CBS, 2022)
- Ran k-means clustering on 50k customer records using scikit-learn.
- Visualised segment profiles with matplotlib and delivered recommendations
  to a fictional retail client.

SKILLS
SQL: Snowflake, PostgreSQL, window functions, CTEs
BI: Power BI (certified), Excel
Python: pandas, seaborn, matplotlib, scikit-learn
Other: Git, PowerPoint
"""


@pytest.fixture(scope="session")
async def da_result() -> AnalysisResponse:
    return await run_analysis(_DA_CV, _DA_JOB_POSTING)


def _da_covered(result: AnalysisResponse) -> list[str]:
    return [s.label.lower() for s in result.covered_skills]

def _da_bonus_gaps(result: AnalysisResponse) -> list[str]:
    return [g.label.lower() for g in result.skill_gaps if g.category == SkillCategory.BONUS]


@pytest.mark.eval
async def test_da_sql_is_covered(da_result):
    """Daily SQL with window functions and CTEs — unambiguously covered."""
    assert any("sql" in l or "query" in l or "database" in l for l in _da_covered(da_result)), (
        f"Expected SQL to be covered. Covered: {_da_covered(da_result)}"
    )

@pytest.mark.eval
async def test_da_power_bi_is_covered(da_result):
    """Power BI dashboards built and maintained at Coloplast — clearly covered."""
    assert any(
        "power bi" in l or "bi" in l or "dashboard" in l or "visualis" in l
        for l in _da_covered(da_result)
    ), (
        f"Expected Power BI / BI tooling to be covered. Covered: {_da_covered(da_result)}"
    )

@pytest.mark.eval
async def test_da_python_is_covered(da_result):
    """pandas + seaborn automation script is explicit Python evidence."""
    assert any("python" in l for l in _da_covered(da_result)), (
        f"Expected Python to be covered. Covered: {_da_covered(da_result)}"
    )

@pytest.mark.eval
async def test_da_dbt_is_a_bonus_gap(da_result):
    """dbt is bonus in the posting and absent from the CV."""
    assert any("dbt" in l or "transform" in l for l in _da_bonus_gaps(da_result)), (
        f"Expected dbt to be a bonus gap. Bonus gaps: {_da_bonus_gaps(da_result)}"
    )

@pytest.mark.eval
async def test_da_spark_is_a_bonus_gap(da_result):
    """Spark / distributed processing is bonus and absent from the CV."""
    assert any(
        "spark" in l or "distributed" in l or "large-scale" in l
        for l in _da_bonus_gaps(da_result)
    ), (
        f"Expected Spark to be a bonus gap. Bonus gaps: {_da_bonus_gaps(da_result)}"
    )

@pytest.mark.eval
async def test_da_dbt_not_hallucinated_as_covered(da_result):
    """
    dbt appears nowhere in the CV.
    Guards against the LLM inferring dbt from Snowflake / data warehouse experience.
    """
    assert not any("dbt" in l for l in _da_covered(da_result)), (
        f"dbt should not be covered. Covered: {_da_covered(da_result)}"
    )


# ===========================================================================
# Scenario 4 — DTU Bioinformatics (Research Assistant)
# Candidate: Python, RNA-seq/DESeq2, NGS tools, Linux; no Nextflow, no cloud HPC
# Job: requires Python, RNA-seq, NGS tools, Linux; bonus Nextflow and cloud HPC
# ===========================================================================

_BIO_JOB_POSTING = """
Bioinformatics Research Assistant
DTU Bioinformatics, Lyngby

We are looking for a research assistant to support our computational biology
group working on cancer genomics. You will process and analyse large-scale
sequencing datasets and contribute to reproducible research pipelines.

You will:
- Develop and maintain Python scripts and pipelines for processing NGS data
- Perform RNA-seq differential expression analysis using established R packages
  (DESeq2, edgeR)
- Work with standard genomics file formats (FASTQ, BAM, VCF, BED)
- Collaborate with wet-lab researchers to design experiments and interpret results
- Ensure reproducibility through version control and documentation

We require:
- Proficiency in Python for data processing and scripting
- Experience with RNA-seq analysis and familiarity with DESeq2 or edgeR in R
- Practical knowledge of standard NGS file formats and command-line bioinformatics tools
  (samtools, STAR or HISAT2)
- Comfort working in a Linux/Unix environment

Nice to have:
- Experience with workflow managers such as Nextflow or Snakemake
- Access to and experience with cloud-based HPC (AWS Batch, Google Life Sciences)
- Familiarity with single-cell RNA-seq (scRNA-seq) analysis
"""

_BIO_CV = """
Emma Christoffersen
MSc Bioinformatics, DTU, 2023 (expected)
emma.christoffersen@student.dtu.dk

EDUCATION
DTU, MSc Bioinformatics, 2021–present
Relevant courses: Algorithms in Bioinformatics, Statistical Genomics,
Applied Machine Learning, High-Throughput Sequencing Data Analysis

RESEARCH EXPERIENCE

Research Assistant, DTU Bioinformatics (Jan 2023–present)
- Developed Python scripts to automate QC and trimming of FASTQ files
  using FastQC and Trimmomatic; processes ~20 samples per run.
- Ran STAR alignment and featureCounts on paired-end RNA-seq data; generated
  count matrices for downstream analysis.
- Performed differential expression analysis in R with DESeq2; identified
  312 significantly differentially expressed genes in a colorectal cancer dataset.
- Maintained analysis code in a Git repository with README documentation.

Bachelor project: Variant Calling Pipeline (2021)
- Wrote a Bash/Python pipeline to call SNVs from WGS data using GATK HaplotypeCaller.
- Processed BAM and VCF files; applied standard filtering with bcftools.
- Ran all analysis on a Linux cluster (DTU HPC).

SKILLS
Languages: Python (primary), R (DESeq2, ggplot2, dplyr), Bash
Bioinformatics tools: samtools, STAR, HISAT2, FastQC, GATK, bcftools, featureCounts
Formats: FASTQ, BAM, VCF, BED
Environment: Linux, Git, DTU HPC (SLURM)
"""


@pytest.fixture(scope="session")
async def bio_result() -> AnalysisResponse:
    return await run_analysis(_BIO_CV, _BIO_JOB_POSTING)


def _bio_covered(result: AnalysisResponse) -> list[str]:
    return [s.label.lower() for s in result.covered_skills]

def _bio_bonus_gaps(result: AnalysisResponse) -> list[str]:
    return [g.label.lower() for g in result.skill_gaps if g.category == SkillCategory.BONUS]


@pytest.mark.eval
async def test_bio_python_is_covered(bio_result):
    """Python scripting is used throughout the research role — must be covered."""
    assert any("python" in l for l in _bio_covered(bio_result)), (
        f"Expected Python to be covered. Covered: {_bio_covered(bio_result)}"
    )

@pytest.mark.eval
async def test_bio_rnaseq_deseq2_is_covered(bio_result):
    """DESeq2 differential expression on real data — unambiguous coverage."""
    assert any(
        kw in l for l in _bio_covered(bio_result)
        for kw in ("rna", "deseq", "differential", "expression", "genomic")
    ), (
        f"Expected RNA-seq/DESeq2 to be covered. Covered: {_bio_covered(bio_result)}"
    )

@pytest.mark.eval
async def test_bio_ngs_tools_are_covered(bio_result):
    """STAR, samtools, FASTQ/BAM are explicit — NGS tooling must be covered."""
    assert any(
        kw in l for l in _bio_covered(bio_result)
        for kw in ("ngs", "sequenc", "alignment", "samtools", "star", "fastq", "bam", "pipeline")
    ), (
        f"Expected NGS tools / file formats to be covered. Covered: {_bio_covered(bio_result)}"
    )

@pytest.mark.eval
async def test_bio_linux_is_covered(bio_result):
    """All analysis done on Linux cluster — Linux proficiency is clearly covered."""
    assert any("linux" in l or "unix" in l or "command" in l for l in _bio_covered(bio_result)), (
        f"Expected Linux environment to be covered. Covered: {_bio_covered(bio_result)}"
    )

@pytest.mark.eval
async def test_bio_nextflow_is_a_bonus_gap(bio_result):
    """Nextflow/Snakemake is nice-to-have and absent from the CV."""
    assert any(
        kw in l for l in _bio_bonus_gaps(bio_result)
        for kw in ("nextflow", "snakemake", "workflow")
    ), (
        f"Expected Nextflow/Snakemake as a bonus gap. Bonus gaps: {_bio_bonus_gaps(bio_result)}"
    )

@pytest.mark.eval
async def test_bio_cloud_hpc_is_a_bonus_gap(bio_result):
    """Cloud HPC (AWS Batch etc.) is nice-to-have; DTU HPC (SLURM) is not the same."""
    assert any(
        kw in l for l in _bio_bonus_gaps(bio_result)
        for kw in ("cloud", "aws", "hpc", "batch")
    ), (
        f"Expected cloud HPC to be a bonus gap. Bonus gaps: {_bio_bonus_gaps(bio_result)}"
    )

@pytest.mark.eval
async def test_bio_cloud_not_hallucinated_as_covered(bio_result):
    """
    DTU HPC (SLURM) is on-premise, not cloud. The LLM must not conflate
    SLURM experience with AWS Batch or Google Life Sciences.
    """
    assert not any(
        kw in l for l in _bio_covered(bio_result)
        for kw in ("aws", "google life", "cloud hpc", "cloud-based")
    ), (
        f"Cloud HPC should not be covered — DTU HPC (SLURM) is on-premise.\n"
        f"Covered: {_bio_covered(bio_result)}"
    )

@pytest.mark.eval
async def test_bio_nextflow_not_hallucinated_as_covered(bio_result):
    """Nextflow and Snakemake do not appear anywhere in the CV."""
    assert not any(
        kw in l for l in _bio_covered(bio_result)
        for kw in ("nextflow", "snakemake")
    ), (
        f"Nextflow/Snakemake should not be covered. Covered: {_bio_covered(bio_result)}"
    )