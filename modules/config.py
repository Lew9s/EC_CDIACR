import os
from typing import Optional
from pydantic import BaseSettings


class Settings(BaseSettings):
    neo4j_username: str = os.getenv("NEO4J_USERNAME")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD")
    neo4j_url: str = os.getenv("NEO4J_URL")
    
    ollama_model: str = os.getenv("OLLAMA_MODEL")
    ollama_embedding_model: str = os.getenv("OLLAMA_EMBEDDING_MODEL")
    ollama_request_timeout: int = int(os.getenv("OLLAMA_REQUEST_TIMEOUT"))
    
    data_path: str = os.getenv("DATA_PATH")
    
    cypher_query: str = os.getenv("CYPHER_QUERY")

    # Align naming with environmental variables expected by the project
    entities_list: str = os.getenv("ENTITIES_LIST")
    relations_list: str = os.getenv("RELATIONS_LIST")
    validation_schema: str = os.getenv("VALIDATION_SCHEMA")
    extraction_prompt: str = os.getenv("EXTRACTION_PROMPT")
    
    class Config:
        env_file = ".env"


settings = Settings()
