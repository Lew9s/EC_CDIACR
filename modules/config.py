import os

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    return float(raw)


class Settings:
    neo4j_username: str = os.getenv("NEO4J_USERNAME", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "12345678")
    neo4j_url: str = os.getenv("NEO4J_URL", "bolt://localhost:7687")

    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen3:30b-a3b-Instruct")
    ollama_embedding_model: str = os.getenv("OLLAMA_EMBEDDING_MODEL", "bge-m3:latest")
    ollama_request_timeout: int = _env_int("OLLAMA_REQUEST_TIMEOUT", 3600)

    data_path: str = os.getenv("DATA_PATH", "./data")
    cypher_query: str = os.getenv("CYPHER_QUERY", "")

    entities_list: str = os.getenv("ENTITIES_LIST", "")
    relations_list: str = os.getenv("RELATIONS_LIST", "")
    validation_schema: str = os.getenv("VALIDATION_SCHEMA", "")
    extraction_prompt: str = os.getenv("EXTRACTION_PROMPT", "")

    consensus_threshold: float = _env_float("CONSENSUS_THRESHOLD", 0.8)
    max_consensus_rounds: int = _env_int("MAX_CONSENSUS_ROUNDS", 3)
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = _env_int("API_PORT", 8000)


settings = Settings()
