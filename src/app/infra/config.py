from os import getenv

from dotenv import load_dotenv
from qdrant_client import models

load_dotenv()

DB_NAME = getenv("DB_NAME")
DB_PASSWORD = getenv("DB_PASSWORD")
DB_HOST = getenv("DB_HOST")
DB_PORT = getenv("DB_PORT")
DB_USER = getenv("DB_USER")

MODEL_NAME = getenv("OPEN_ROUTER_MODEL", "nvidia/nemotron-3.5-lightning:free")
EMBEDDING_MODEL = getenv("EMBEDDING_MODEL", "liquid/lfm-2.5-embedding-350m:free")
MODEL_KEY = getenv("OPEN_ROUTER_API_KEY")

LOG_LEVEL = getenv("LOG_LEVEL", "INFO")
EMBED_TOKENIZER = getenv("EMBED_TOKENIZER", "Qwen/Qwen3-Embedding-0.6B")
MAX_TOKENS = int(getenv("MAX_TOKENS", "512"))

QDRANT_URL = "http://localhost:6333"
COLLECTION = "chunks"

# BM25 is computed by the Qdrant server: we send raw text, it tokenizes, stems and
# weights terms. k1 / b / avg_len are knobs (Chapter 2); avg_len is in tokens.
BM25_MODEL = "Qdrant/bm25"
BM25_OPTS = models.Bm25Config(k=1.2, b=0.75, avg_len=300)

REWRITE_MODEL = getenv("REWRITE_MODEL", "openai/gpt-4o-mini")
RERANK_MODEL = getenv("RERANK_MODEL", "cohere/rerank-v3.5")
