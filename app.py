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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="GraphRAG", version="1.0")

# 初始化
graph_store = Neo4jPGStore(
    username="neo4j",
    password="12345678",
    url="bolt://localhost:8687",
)

llm = Ollama(model="qwen3:30b-a3b-Instruct", request_timeout=3600)
embed_model = OllamaEmbedding(model_name="bge-m3:latest")

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

cypher = """
MATCH (comp:COMPONENT {name: $component_name})

MATCH (co:CHANGE_ORDER)-[:MODIFIES]->(comp)

WITH DISTINCT co.group_key AS target_group

MATCH (related_co:CHANGE_ORDER)
WHERE related_co.group_key = target_group

MATCH (related_co)-[:HAS_REASON]->(reason:REASON)
MATCH (related_co)-[:OCCURS_AT]->(time:TIME_POINT)
MATCH (related_co)-[:SIGNED_BY]->(dept:DEPARTMENT)

ORDER BY related_co.name
RETURN
  related_co.name AS change_order,
  related_co.group_key AS change_group,
  reason.name AS reason,
  time.name AS time_point,
  COLLECT(DISTINCT dept.name) AS departments,
  [(related_co)-[:MODIFIES]->(c:COMPONENT) | c.name] AS modified_components
"""
class ComponentChangeQuery(PydanticBaseModel):
    component_name: str = Field(description="要查询变更历史的组件名称")

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
        raise HTTPException(status_code=400, detail="问题不能为空")

    try:
        nodes = await retriever.aretrieve(request.question.strip())
        return [node.text.strip() for node in nodes if node.text.strip()]
    except Exception as e:
        logger.error(f"检索失败: {e}")
        raise HTTPException(status_code=500, detail="知识图谱检索失败")

@app.get("/")
def health():
    return {"status":200}

if __name__ == "__main__":
    uvicorn.run("app:app",host="0.0.0.0",port=8000,reload=False)
