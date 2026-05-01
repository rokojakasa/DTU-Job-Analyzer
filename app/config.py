from pathlib import Path
from dotenv import load_dotenv
import os

import dspy

load_dotenv(os.path.expanduser("~/.env"))
CAMPUSAI_API_KEY = os.getenv("CAMPUSAI_API_KEY")
CAMPUSAI_URL = "https://api.campusai.compute.dtu.dk/"
EMBED_MODEL = "Nomic Embed Text"
CHAT_MODEL = "Gemma 4"

ESCO_PATH = Path("docs/Nielsen2025Natural-2026-03-20.pdf.tei.xml")
DTU_COURSES_PATH = Path("data/chunks.json")
COURSES_TOP_K: int = 5

def configure_dspy():
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