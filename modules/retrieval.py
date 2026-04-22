import logging
from typing import List, Optional

from pydantic import BaseModel as PydanticBaseModel, Field

from config import settings

logger = logging.getLogger(__name__)


class ComponentChangeQuery(PydanticBaseModel):
    component_name: str = Field(description="Component name for querying change history")


class GraphRAGService:
    def __init__(self) -> None:
        self._retriever = None

    def _build_retriever(self):
        from llama_index.core.indices.property_graph import (
            CypherTemplateRetriever,
            LLMSynonymRetriever,
        )

        from graph_index import create_llm, load_existing_property_graph_index

        index = load_existing_property_graph_index()
        llm = create_llm()

        sub_retrievers = [
            LLMSynonymRetriever(
                index.property_graph_store,
                llm=llm,
                include_text=True,
            )
        ]

        if settings.cypher_query:
            sub_retrievers.append(
                CypherTemplateRetriever(
                    index.property_graph_store,
                    ComponentChangeQuery,
                    settings.cypher_query,
                    llm,
                )
            )

        return index.as_retriever(
            sub_retrievers=sub_retrievers,
            include_text=True,
            similarity_top_k=10,
        )

    @property
    def retriever(self):
        if self._retriever is None:
            logger.info("Initializing LlamaIndex graph retriever")
            self._retriever = self._build_retriever()
        return self._retriever

    async def retrieve(self, question: str) -> List[str]:
        nodes = await self.retriever.aretrieve(question.strip())
        return [node.text.strip() for node in nodes if getattr(node, "text", "").strip()]


class StaticRAGService:
    def __init__(self, context: Optional[List[str]] = None) -> None:
        self.context = context or []

    async def retrieve(self, question: str) -> List[str]:
        return list(self.context)
