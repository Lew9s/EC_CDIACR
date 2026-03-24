from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from llama_index.core import PropertyGraphIndex
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.graph_stores.neo4j import Neo4jPGStore
from llama_index.core.indices.property_graph import (
    LLMSynonymRetriever,
    VectorContextRetriever,
    CypherTemplateRetriever,
)
from pydantic import BaseModel as PydanticBaseModel, Field
import uvicorn
import logging
import os
from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="GraphRAG", version="1.0")

# 初始化
graph_store = Neo4jPGStore(
    username=settings.neo4j_username,
    password=settings.neo4j_password,
    url=settings.neo4j_url,
)

llm = Ollama(model=settings.ollama_model, request_timeout=settings.ollama_request_timeout)
embed_model = OllamaEmbedding(model_name=settings.ollama_embedding_model)

# 加载图谱
index = PropertyGraphIndex.from_existing(
    property_graph_store=graph_store,
    llm=llm,
    embed_model=embed_model,
    kg_extractors=[],
    embed_kg_nodes=True,
)

# 子检索器
llm_synonym = LLMSynonymRetriever(
    index.property_graph_store,
    llm=llm,
    include_text=True,
)

vector_context = VectorContextRetriever(
    index.property_graph_store,
    embed_model=embed_model,
    include_text=True,
)

cypher = settings.cypher_query
class ComponentChangeQuery(PydanticBaseModel):
    component_name: str = Field(description="Component name for querying change history")

Cypher_retriever = CypherTemplateRetriever(
    index.property_graph_store,
    ComponentChangeQuery,
    cypher,
    llm
)

# 查询引擎
retriever = index.as_retriever(
    sub_retrievers=[
        llm_synonym,
        # vector_context,
        Cypher_retriever
    ],
    include_text=True,
    similarity_top_k=10
)

class QueryRequest(BaseModel):
    question:str

@app.post("/query",response_model=List[str])
async def query(request:QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="The question cannot be left blank.")

    try:
        nodes = await retriever.aretrieve(request.question.strip())
        return [node.text.strip() for node in nodes if node.text.strip()]
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail="Knowledge graph retrieval failed")

@app.get("/")
def health():
    return {"status":200}

if __name__ == "__main__":
    uvicorn.run("app:app",host="0.0.0.0",port=8000,reload=False)