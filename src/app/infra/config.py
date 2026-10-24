from os import getenv

from dotenv import load_dotenv

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
