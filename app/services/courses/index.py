from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import logging
import asyncio

import numpy as np
from openai import AsyncOpenAI

from app.config import CAMPUSAI_API_KEY, CAMPUSAI_URL, EMBED_MODEL, DTU_COURSES_PATH


logger = logging.getLogger(__name__)

client = AsyncOpenAI(api_key=CAMPUSAI_API_KEY, base_url=CAMPUSAI_URL)

INDEX_CACHE_PATH = Path("data/index.npz")

EMBED_BATCH_SIZE = 400

BATCH_DELAY = 0.1

@dataclass
class ObjectiveChunk:
    course_code: str
    title: str
    ects: int
    objective: str
    # embedding: np.ndarray = field(default_factory=lambda: np.array([]))
    
def load_chunks(path: Path = DTU_COURSES_PATH) -> list[ObjectiveChunk]:
    """
    Load courses from JSON and explode into one ObjectiveChunk per
    learning objective. No embeddings yet — pure data reshaping.
    """
    raw = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    chunks: list[ObjectiveChunk] = []
    
    for course in raw:
        code = course.get("course_code", "")
        title = course.get("title", "")
        ects = course.get("fields", {}).get("Point( ECTS )", 0)
        objectives = course.get("learning_objectives", [])

        if not objectives:
            # Some courses may have no objectives — skip rather than
            # emit an empty chunk that would produce a meaningless embedding
            continue

        for objective in objectives:
            chunks.append(ObjectiveChunk(
                course_code=code,
                title=title,
                ects=int(float(str(ects).replace(",", "."))) if ects else 0,
                objective=f"{title}: {objective}"
            ))
    logger.info("Loaded %d objective chunks from %d courses.", len(chunks), len(raw))
    return chunks

async def embed_chunks(
    chunks: list[ObjectiveChunk],
    batch_size: int = EMBED_BATCH_SIZE,
    batch_delay: float = BATCH_DELAY,
) -> tuple[list[ObjectiveChunk], np.ndarray]:
    """
    Embed all objective chunks via the CampusAI embeddings API.
 
    Sends objectives in batches of `batch_size` to stay within API limits.
    A short delay between batches avoids triggering rate limiting.
    Embeddings are attached to each chunk in-place.
    """
    all_embeddings: list[list[float]] = []
    total = len(chunks)
    num_batches = (total + batch_size - 1) // batch_size
 
    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, total)
        batch = chunks[start:end]
 
        logger.info(
            "Embedding batch %d/%d (objectives %d–%d)...",
            batch_idx + 1,
            num_batches,
            start,
            end - 1,
        )
 
        response = await client.embeddings.create(
            model=EMBED_MODEL,
            input=[c.objective for c in batch],
        )
        all_embeddings.extend(e.embedding for e in response.data)
 
        # Avoid hammering the API between batches.
        if batch_idx < num_batches - 1:
            await asyncio.sleep(batch_delay)
 
    matrix = np.array(all_embeddings, dtype=np.float32)
 
    logger.info("Finished embedding %d chunks.", total)
    return chunks, matrix

def save_index(chunks: list[ObjectiveChunk], matrix: np.ndarray, path: Path = INDEX_CACHE_PATH) -> None:
    """
    Persist the embedded index to a compressed numpy archive.
 
    Stores embeddings as a 2-D float32 matrix (N × embedding_dim) and all
    metadata fields as string/int arrays in the same order.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
 
    np.savez_compressed(
        path,
        embeddings=matrix,  # (N, dim)
        course_codes=np.array([c.course_code for c in chunks]),
        titles=np.array([c.title for c in chunks]),
        ects=np.array([c.ects for c in chunks], dtype=np.int16),
        objectives=np.array([c.objective for c in chunks]),
    )
    logger.info("Saved index with %d chunks to %s.", len(chunks), path)

def load_index(path: Path = INDEX_CACHE_PATH) -> tuple[list[ObjectiveChunk], np.ndarray]:
    data = np.load(path, allow_pickle=False)
    
    # Convert all arrays to Python lists up front — one vectorized operation each
    course_codes = data["course_codes"].tolist()
    titles = data["titles"].tolist()
    ects = data["ects"].tolist()
    objectives = data["objectives"].tolist()
    matrix = data["embeddings"]

    logger.info("arrays converted, building chunks...")

    chunks = [
        ObjectiveChunk(
            course_code=course_codes[i],
            title=titles[i],
            ects=ects[i],
            objective=objectives[i],
        )
        for i in range(len(course_codes))
    ]

    logger.info("Loaded index with %d chunks from %s.", len(chunks), path)
    return chunks, matrix

def _cache_is_fresh(
    cache_path: Path = INDEX_CACHE_PATH,
    source_path: Path = DTU_COURSES_PATH,
) -> bool:
    """
    Returns True if the cache exists and is newer than chunks.json.
    If chunks.json has been updated since the last index build, we rebuild.
    """
    if not cache_path.exists():
        return False
    return cache_path.stat().st_mtime > source_path.stat().st_mtime


async def build_index(force_rebuild: bool = False) -> list[ObjectiveChunk]:
    """
    Build or load the course objective index.
 
    On first run (or when chunks.json is newer than the cache), embeds all
    objectives via the CampusAI API and saves the result to data/index.npz.
 
    On subsequent startups, loads directly from the cache — typically under
    one second, with no API calls.
 
    Args:
        force_rebuild: Ignore the cache and re-embed everything. Useful when
                       the embedding model changes.
    """
    if not force_rebuild and _cache_is_fresh():
        logger.info("Cache is fresh — loading index from disk.")
        return load_index()
 
    reason = "force_rebuild=True" if force_rebuild else "cache missing or stale"
    logger.info("Building index from scratch (%s). This may take a while...", reason)
 
    chunks = load_chunks()
    chunks, matrix = await embed_chunks(chunks)
    save_index(chunks, matrix)
 
    return chunks, matrix
