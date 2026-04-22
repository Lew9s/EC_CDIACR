import json
from typing import List

import nest_asyncio
from llama_index.core import Document, PropertyGraphIndex
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

try:
    from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
except ImportError:
    from llama_index.graph_stores.neo4j import Neo4jPGStore as Neo4jPropertyGraphStore

from config import settings


def parse_literal_from_string(literal_string: str) -> List[str]:
    if not literal_string:
        return []
    return [item.strip() for item in literal_string.split(",") if item.strip()]


def create_graph_store():
    return Neo4jPropertyGraphStore(
        username=settings.neo4j_username,
        password=settings.neo4j_password,
        url=settings.neo4j_url,
    )


def create_llm(json_mode: bool = False):
    return Ollama(
        model=settings.ollama_model,
        json_mode=json_mode,
        request_timeout=settings.ollama_request_timeout,
    )


def create_embed_model():
    return OllamaEmbedding(model_name=settings.ollama_embedding_model)


def build_property_graph_index(documents: List[Document]) -> PropertyGraphIndex:
    nest_asyncio.apply()

    validation_schema = (
        json.loads(settings.validation_schema) if settings.validation_schema else {}
    )
    kg_extractor = SchemaLLMPathExtractor(
        llm=create_llm(json_mode=True),
        possible_entities=parse_literal_from_string(settings.entities_list),
        possible_relations=parse_literal_from_string(settings.relations_list),
        kg_validation_schema=validation_schema,
        num_workers=1,
        max_triplets_per_chunk=20,
        extract_prompt=settings.extraction_prompt or None,
        strict=True,
    )

    return PropertyGraphIndex.from_documents(
        documents,
        kg_extractors=[kg_extractor],
        embed_model=create_embed_model(),
        property_graph_store=create_graph_store(),
        show_progress=True,
    )


def load_existing_property_graph_index() -> PropertyGraphIndex:
    return PropertyGraphIndex.from_existing(
        property_graph_store=create_graph_store(),
        llm=create_llm(),
        embed_model=create_embed_model(),
        kg_extractors=[],
        embed_kg_nodes=True,
    )
