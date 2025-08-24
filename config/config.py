import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

MODEL_NAME = "gpt-4"
EMBEDDING_MODEL = "text-embedding-ada-002"
TEMPERATURE = 0.2
MAX_TOKENS = 4096

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
PROMPT_DIR = os.path.join(os.path.dirname(__file__), "..", "prompts")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..") 