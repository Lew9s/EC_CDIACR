from llama_index.core import SimpleDirectoryReader, Document
from typing import Literal
from llama_index.llms.ollama import Ollama
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor
from llama_index.graph_stores.neo4j import Neo4jPGStore
from llama_index.core import PropertyGraphIndex
from llama_index.embeddings.ollama import OllamaEmbedding
from typing import List
import re
import nest_asyncio
import json
from config import settings

documents = SimpleDirectoryReader(settings.data_path).load_data()

def split_change_orders(documents: List[Document]) -> List[Document]:
    split_docs = []
    separator = r"!@#\$%\^&\*"
    for doc in documents:
        # Split the text by delimiter
        raw_text = doc.text.strip()
        raw_segments = re.split(separator, raw_text)
        for i, segment in enumerate(raw_segments):
            segment = segment.strip()
            if not segment:
                continue
            metadata = {
                "source_file": doc.metadata.get("file_name", "unknown"),
                "change_order_index": i + 1,
                # You can try to extract the order number from the text as the ID.
                **doc.metadata
            }
            split_doc = Document(
                text=segment,
                metadata=metadata,
                id_=f"{metadata['source_file']}_CO_{i+1}"
            )
            split_docs.append(split_doc)
    return split_docs

split_documents = split_change_orders(documents)

nest_asyncio.apply()

# Define schema
def parse_literal_from_string(literal_string: str) -> List[str]:
    if not literal_string:
        return []
    items = [item.strip() for item in literal_string.split(',') if item.strip()]
    return items

# Load entity/relation type lists from environment-aligned names
entities = parse_literal_from_string(settings.entities_list)
relations = parse_literal_from_string(settings.relations_list)
# Validation schema may be optional; guard against None
validation_schema = json.loads(settings.validation_schema) if settings.validation_schema else {}

prompt = settings.extraction_prompt

kg_extractor = SchemaLLMPathExtractor(
    llm=Ollama(model=settings.ollama_model, json_mode=True, request_timeout=settings.ollama_request_timeout),
    possible_entities=entities,
    possible_relations=relations,
    kg_validation_schema=validation_schema,
    num_workers=1,
    max_triplets_per_chunk=20,
    extract_prompt=prompt,
    strict=True,
)

graph_store = Neo4jPGStore(
    username=settings.neo4j_username or "neo4j",
    password=settings.neo4j_password or "12345678",
    url=settings.neo4j_url or "bolt://localhost:7687",
)

index = PropertyGraphIndex.from_documents(
    split_documents,
    kg_extractors=[kg_extractor],
    embed_model=OllamaEmbedding(model_name="bge-m3:latest"),
    property_graph_store=graph_store,
    show_progress=True,
)
