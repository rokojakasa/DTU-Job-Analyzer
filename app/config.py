from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv(os.path.expanduser("~/.env"))
CAMPUSAI_API_KEY = os.getenv("CAMPUSAI_API_KEY")
CAMPUSAI_URL = os.getenv("CAMPUSAI_API_URL")
EMBED_MODEL = os.getenv("CAMPUSAI_EMBED_MODEL", "Nomic Embed Text")
CHAT_MODEL = os.getenv("CAMPUSAI_MODEL", "google/gemma-4-26b-a4b")

ESCO_PATH = Path("docs/Nielsen2025Natural-2026-03-20.pdf.tei.xml")
DTU_COURSES_PATH = Path("data/dtu_courses.jsonl")
COURSES_TOP_K: int = 3

def configure_dspy():
    import dspy
    """Configure DSPy's global LM to use the CampusAI OpenAI-compatible endpoint.
    Must be called once at startup before any dspy.Predict, dspy.ChainOfThought,
    or dspy.ReAct is used — regardless of which mode (rag/react) is active."""
    lm = dspy.LM(
        model=f"openai/{CHAT_MODEL}",
        api_base=CAMPUSAI_URL,
        api_key=CAMPUSAI_API_KEY,
        cache=False,
    )
    dspy.configure(lm=lm)