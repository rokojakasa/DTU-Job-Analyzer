from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.store import Store
from app.routes import cv, job_posting, analyse, courses
from app.services.courses.index import build_index

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.store = Store()
    app.state.course_index = await build_index()
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="DTU Job Analyzer",
        description="Identify skill gaps, get course recommendations, and prepare for interviews.",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.include_router(cv.router, prefix="/cv", tags=["CV"])
    app.include_router(job_posting.router, prefix="/job-posting", tags=["Job Posting"])
    app.include_router(analyse.router, prefix="/analyse", tags=["Analysis"])
    app.include_router(courses.router, prefix="/courses", tags=["Courses"])
    # app.include_router(questions.router, prefix="/questions", tags=["Questions"])

    return app


app = create_app()