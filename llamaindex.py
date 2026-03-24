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

documents = SimpleDirectoryReader("/home/justjg/code/data").load_data()

def split_change_orders(documents: List[Document]) -> List[Document]:
    split_docs = []
    separator = r"!@#\$%\^&\*"
    for doc in documents:
        # 按分隔符切分文本
        raw_text = doc.text.strip()
        raw_segments = re.split(separator, raw_text)
        for i, segment in enumerate(raw_segments):
            segment = segment.strip()
            if not segment:
                continue
            metadata = {
                "source_file": doc.metadata.get("file_name", "unknown"),
                "change_order_index": i + 1,
                # 可尝试从文本中提取单号作为 id
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

# 定义 schema
entities = Literal["CHANGE_ORDER", "COMPONENT", "DEPARTMENT", "REASON", "TIME_POINT"]
relations = Literal["MODIFIES", "SIGNED_BY", "HAS_REASON", "OCCURS_AT", "PART_OF"]
validation_schema = {
    "CHANGE_ORDER": ["MODIFIES", "SIGNED_BY", "HAS_REASON", "OCCURS_AT"],
    "COMPONENT": ["MODIFIES", "PART_OF"],
    "DEPARTMENT": ["SIGNED_BY"],
    "REASON": ["HAS_REASON"],
    "TIME_POINT": ["OCCURS_AT"]
}

prompt=(
    """
你是一个专业的知识图谱信息提取助手。请从以下变更单文本中，严格按照指定的 schema 提取知识图谱三元组。

### 【Schema 定义】
- 实体类型（entities）：
  - CHANGE_ORDER：变更单号，如 H-01、H-02-1
  - COMPONENT：变更对象，如“缆绳”、“污水井”
  - DEPARTMENT：签收部门，如“舾冷车间”
  - REASON：变更原因，如“规格书要求”
  - TIME_POINT：变更时间点，如“订货前修改”

- 关系类型（relations）：
  - MODIFIES：变更单修改了某个部件
  - SIGNED_BY：变更单由某个部门签收
  - HAS_REASON：变更单有某个原因
  - OCCURS_AT：变更单发生在某个时间点
  - PART_OF：部件属于某个系统（可选）

- 合法关系约束（仅允许以下组合）：
  - CHANGE_ORDER → MODIFIES → COMPONENT
  - CHANGE_ORDER → SIGNED_BY → DEPARTMENT
  - CHANGE_ORDER → HAS_REASON → REASON
  - CHANGE_ORDER → OCCURS_AT → TIME_POINT
  - COMPONENT → PART_OF → COMPONENT（可选层级）

### 【提取规则】
1. 每个变更单（单号）作为一个独立的 `CHANGE_ORDER` 节点。
2. “变更对象” → `COMPONENT`
3. 若“变更内容”中明确提及被修改、替换、新增或移除的具体部件，即使未在“变更对象”中列出，也应作为 `COMPONENT` 实体提取，并建立 CHANGE_ORDER → MODIFIES → COMPONENT 关系
4. 若“变更内容”中表明某部件因另一部件的位置、设计或状态变化而被修改（如“A修改导致B补孔”），且二者存在空间或结构依存关系，应补充 COMPONENT → PART_OF → COMPONENT 三元组，方向为“子部件 → PART_OF → 父结构”或“附属部件 → PART_OF → 主体结构”
5. “签收部门” → 多个 `DEPARTMENT`（按逗号或顿号分割）
6. “变更原因” → `REASON`
7. “变更时间点” → `TIME_POINT`
8. 请尽量输出 {max_triplets_per_chunk} 个三元组以内。
9. 不要添加解释、不要自由发挥、不要猜测未提及的内容。

### 【待处理文本】
-------
{text}
-------"""
)

kg_extractor = SchemaLLMPathExtractor(
    llm=Ollama(model="qwen3:30b-a3b-Instruct", json_mode=True, request_timeout=3600),
    possible_entities=entities,
    possible_relations=relations,
    kg_validation_schema=validation_schema,
    num_workers=1,
    max_triplets_per_chunk=20,
    extract_prompt=prompt,
    strict=True,
)

graph_store = Neo4jPGStore(
    username="neo4j",
    password="12345678",
    url="bolt://localhost:8687",
)

index = PropertyGraphIndex.from_documents(
    split_documents,
    kg_extractors=[kg_extractor],
    embed_model=OllamaEmbedding(model_name="bge-m3:latest"),
    property_graph_store=graph_store,
    show_progress=True,
)