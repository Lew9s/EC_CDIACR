import asyncio
import logging
from functools import lru_cache
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from neo4j import GraphDatabase

from config import settings
from retrieval import GraphRAGService
from schemas import (
    ProposalAnalysisRequest,
    ProposalGenerateRequest,
    ProposalGenerateResponse,
    ProposalRunRequest,
    ProposalRunResponse,
    QueryRequest,
)
from workflow import ProposalWorkflow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="GraphRAG LangGraph API", version="2.0")


@lru_cache(maxsize=1)
def get_rag_service() -> GraphRAGService:
    return GraphRAGService()


@lru_cache(maxsize=1)
def get_workflow() -> ProposalWorkflow:
    return ProposalWorkflow(get_rag_service())


def _initial_state(question: str):
    return {
        "user_query": question,
        "iteration": 0,
        "errors": [],
        "retrieval_context": [],
        "expert_opinions": [],
        "expert_votes": [],
        "consensus_rate": 0.0,
    }


@app.post("/query", response_model=List[str])
async def query(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="The question cannot be left blank.")

    try:
        return await get_rag_service().retrieve(request.question.strip())
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail="Knowledge graph retrieval failed")


@app.post("/visualize")
async def analyze_proposal(request: ProposalAnalysisRequest):
    if not request.proposal_text.strip():
        raise HTTPException(
            status_code=400, detail="The proposal text cannot be left blank."
        )

    try:
        from explainability_module import ExplainabilityModule

        driver = GraphDatabase.driver(
            settings.neo4j_url,
            auth=(settings.neo4j_username, settings.neo4j_password),
        )
        session = driver.session()

        module = ExplainabilityModule(
            llm=settings.ollama_model,
            kg_client=session,
            embedding_model=settings.ollama_embedding_model,
        )
        result = module.run(request.proposal_text, visualize=request.visualize)

        if request.visualize:
            return HTMLResponse(content=result, status_code=200)
        return result
    except Exception as e:
        logger.error(f"Proposal analysis failed: {e}")
        raise HTTPException(status_code=500, detail="Proposal analysis failed")
    finally:
        if "session" in locals():
            session.close()
        if "driver" in locals():
            driver.close()


@app.post("/proposal/generate", response_model=ProposalGenerateResponse)
async def generate_proposal(request: ProposalGenerateRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="The question cannot be left blank.")

    try:
        question = request.question.strip()
        state = await asyncio.to_thread(
            lambda: get_workflow().graph.invoke(_initial_state(question))
        )
        return ProposalGenerateResponse(
            proposal=state.get("final_proposal", ""),
            consensus_rate=state.get("consensus_rate", 0.0),
            iterations=state.get("iteration", 0),
        )
    except Exception as e:
        logger.error(f"Proposal generation failed: {e}")
        raise HTTPException(status_code=500, detail="Proposal generation failed")


@app.post("/proposal/run", response_model=ProposalRunResponse)
async def run_proposal_workflow(request: ProposalRunRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="The question cannot be left blank.")

    try:
        question = request.question.strip()
        state = await asyncio.to_thread(
            lambda: get_workflow().graph.invoke(_initial_state(question))
        )
        return ProposalRunResponse(
            proposal=state.get("final_proposal", ""),
            consensus_rate=state.get("consensus_rate", 0.0),
            iterations=state.get("iteration", 0),
            trace=state if request.include_trace else None,
        )
    except Exception as e:
        logger.error(f"Proposal workflow failed: {e}")
        raise HTTPException(status_code=500, detail="Proposal workflow failed")


@app.get("/")
@app.get("/health")
def health():
    return {"status": 200}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host=settings.api_host, port=settings.api_port, reload=False)
