from __future__ import annotations
 
import logging
from typing import Literal
 
import numpy as np
import json
from openai import AsyncOpenAI
from rank_bm25 import BM25Okapi
 
from app.config import CAMPUSAI_API_KEY, CAMPUSAI_URL, EMBED_MODEL, CHAT_MODEL
from app.services.courses.index import ObjectiveChunk
 
logger = logging.getLogger(__name__)
 
client = AsyncOpenAI(api_key=CAMPUSAI_API_KEY, base_url=CAMPUSAI_URL)
 
RetrievalMode = Literal["sparse", "dense", "hybrid"]

_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "for",
    "is", "be", "with", "on", "by", "as", "at", "from", "use",
}

# How many BM25 candidates to pass to the dense reranker in hybrid mode.
# Large enough to capture relevant results, small enough to keep reranking fast.
_HYBRID_CANDIDATE_POOL = 150


def _tokenise(text: str) -> list[str]:
    """
    Lowercase, split on whitespace, strip punctuation, remove stopwords.
    Kept simple — objectives are already clean, verb-noun prose.
    """
    tokens = []
    for token in text.lower().split():
        token = token.strip(".,;:()")
        if token and token not in _STOPWORDS:
            tokens.append(token)
    return tokens


def _build_bm25(chunks: list[ObjectiveChunk]) -> BM25Okapi:
    """
    Build a BM25 index over the objective texts.
    Called once per retrieval request — BM25Okapi is cheap to construct
    at this scale (~15k short documents).
    """
    corpus = [_tokenise(chunk.objective) for chunk in chunks]
    return BM25Okapi(corpus)

async def transform_query(label: str) -> str:
    prompt = f"""You are helping match industry job requirements to university course topics.

    Rephrase the following industry skill as a short academic topic that would appear 
    in a university course title or learning objective.

    Rules:
    - Use broad academic vocabulary, not industry jargon or brand names
    - Return 4-8 words maximum
    - Return ONLY the rephrased topic, nothing else

    Examples:
    "build LLM pipelines"        → "natural language processing and neural networks"
    "manage cloud infrastructure" → "distributed systems and cloud computing"
    "write REST APIs"             → "web services and network programming"
    "implement CI/CD pipelines"   → "software development and deployment automation"

    Skill: {label}
    Rephrased:"""
    
    response = await client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()

async def expand_query(label: str, job_title: str) -> list[str]:
    prompt = f"""You are helping match industry job skills to university course topics.

        The candidate is applying for position as: {job_title}
        
        Given an industry skill, return a JSON array of 3 search queries ordered from most specific to most general.
        The first query should preserve the original phrasing where useful.
        The others should use broader academic vocabulary a university course title or learning objective would use.

        Return ONLY a valid JSON array of strings, no explanation.

        Examples:
        "program in C# and .NET" → ["C# and .NET programming", "object-oriented programming", "software frameworks and design patterns"]
        "build LLM pipelines"    → ["large language model pipelines", "natural language processing", "machine learning and neural networks"]
        "manage MS SQL"          → ["MS SQL database", "relational database management", "data modelling and query languages"]

        Skill: {label}
        Queries:"""
    response = await client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    raw = response.choices[0].message.content.strip()
    try:
        queries = json.loads(raw)
        if not isinstance(queries, list) or not queries:
            raise ValueError("Expected a non-empty list")
        logger.info("Expanded skill gap label %r to queries %r for better retrieval.", label, queries)
        return queries
    except (json.JSONDecodeError, ValueError):
        logger.warning("Query expansion failed to parse, falling back to original: %r", raw)
        return [label]  # graceful degradation

async def _embed_query(query: str) -> np.ndarray:
    """
    Embed a single skill gap query via the CampusAI embeddings API.
 
    The "skill: " prefix nudges the embedding model toward a skills-matching
    context, which improves similarity scores for short queries.
    Remove the prefix if you find it hurts retrieval quality in practice.
    """
    prefixed = f"skill: {query}"
    response = await client.embeddings.create(
        model=EMBED_MODEL,
        input=[prefixed],
    )
    return np.array(response.data[0].embedding, dtype=np.float32)

def _cosine_similarity(query_vec: np.ndarray, chunk_matrix: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between a single query vector and a matrix
    of chunk embeddings.
 
    Args:
        query_vec:    shape (dim,)
        chunk_matrix: shape (N, dim)
 
    Returns:
        similarities: shape (N,), values in [-1, 1]
    """
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    chunk_norms = chunk_matrix / (np.linalg.norm(chunk_matrix, axis=1, keepdims=True) + 1e-10)
    return chunk_norms @ query_norm

async def _sparse_retrieve(
    query: str,
    chunks: list[ObjectiveChunk],
    top_k: int,
) -> list[tuple[ObjectiveChunk, float]]:
    """
    BM25 retrieval. Fast, no API calls, strong on exact keyword matches.
    Struggles with vocabulary mismatch (synonyms, paraphrasing).
    """
    bm25 = _build_bm25(chunks)
    query_tokens = _tokenise(query)
    scores = bm25.get_scores(query_tokens)  # shape (N,)
 
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(chunks[i], float(scores[i])) for i in top_indices]

async def _dense_retrieve(
    query: str,
    chunks: list[ObjectiveChunk],
    matrix: np.ndarray,
    top_k: int,
) -> list[tuple[ObjectiveChunk, float]]:
    """
    Embedding similarity retrieval. Handles paraphrasing and synonyms well.
    Costs one API call per query. Weaker on exact keyword matches.
    """
    query_vec = await _embed_query(query)
 
    # Stack all embeddings into a matrix for a single vectorised similarity op.
    scores = _cosine_similarity(query_vec, matrix)     # (N,)
 
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(chunks[i], float(scores[i])) for i in top_indices]

async def _hybrid_retrieve(
    query: str,
    chunks: list[ObjectiveChunk],
    matrix: np.ndarray,
    top_k: int,
    candidate_pool: int = _HYBRID_CANDIDATE_POOL,
) -> list[tuple[ObjectiveChunk, float]]:
    """
    BM25 narrows to a candidate pool, dense reranks the shortlist.
 
    BM25 and dense fail in complementary ways:
    - BM25 misses synonyms and paraphrasing
    - Dense misses exact keyword matches and can be fooled by semantic drift
 
    Combining them covers both failure modes. At 14k chunks this is not
    needed for performance, but often improves precision.
    """
    # Step 1 — BM25 retrieves a broad candidate pool.
    bm25 = _build_bm25(chunks)
    query_tokens = _tokenise(query)
    bm25_scores = bm25.get_scores(query_tokens)
    candidate_indices = np.argsort(bm25_scores)[::-1][:candidate_pool]
    candidates = [chunks[i] for i in candidate_indices]
 
    # Step 2 — Dense reranks the shortlist.
    query_vec = await _embed_query(query)
    candidate_matrix = matrix[candidate_indices]  # (pool, dim)
    dense_scores = _cosine_similarity(query_vec, candidate_matrix)  # (pool,)
 
    top_indices = np.argsort(dense_scores)[::-1][:top_k]
    return [(candidates[i], float(dense_scores[i])) for i in top_indices]

async def retrieve(
    query: str,
    chunks: list[ObjectiveChunk],
    matrix: np.ndarray,
    top_k: int,
    mode: RetrievalMode = "hybrid",
) -> list[tuple[ObjectiveChunk, float]]:
    """
    Retrieve the top-k most relevant ObjectiveChunks for a skill gap query.
 
    Args:
        query:  Skill gap label, e.g. "design database schema"
        chunks: The full indexed list from build_index()
        top_k:  Number of results to return
        mode:   "sparse" (BM25), "dense" (embeddings), or "hybrid" (both)
 
    Returns:
        List of (ObjectiveChunk, score) pairs, sorted by score descending.
    """
    logger.debug("Retrieving top-%d chunks for %r using %s mode.", top_k, query, mode)
 
    if mode == "sparse":
        return await _sparse_retrieve(query, chunks, top_k)
    elif mode == "dense":
        return await _dense_retrieve(query, chunks, matrix, top_k)
    elif mode == "hybrid":
        return await _hybrid_retrieve(query, chunks, matrix, top_k)
    else:
        raise ValueError(f"Unknown retrieval mode: {mode!r}. Choose sparse, dense, or hybrid.")

async def retrieve_expanded(
    queries: list[str],
    chunks: list[ObjectiveChunk],
    matrix: np.ndarray,
    top_k: int,
    mode: RetrievalMode = "hybrid",
) -> list[tuple[ObjectiveChunk, float]]:
    # Retrieve for each expanded query concurrently
    results_per_query = []
    for q in queries:
        hits = await retrieve(query=q, chunks=chunks, matrix=matrix, top_k=top_k, mode=mode)
        results_per_query.append(hits)

    # Merge: max score per course code across all queries
    best: dict[str, tuple[ObjectiveChunk, float]] = {}
    for hits in results_per_query:
        for chunk, score in hits:
            code = chunk.course_code
            if code not in best or score > best[code][1]:
                best[code] = (chunk, score)

    # Re-sort and return top_k
    merged = sorted(best.values(), key=lambda x: x[1], reverse=True)
    return merged[:top_k]