from typing import Any, Dict, List, Optional, TypedDict

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str


class ProposalAnalysisRequest(BaseModel):
    proposal_text: str
    visualize: bool = True


class ProposalGenerateRequest(BaseModel):
    question: str
    visualize: bool = False


class ProposalRunRequest(BaseModel):
    question: str
    include_trace: bool = True


class ProposalGenerateResponse(BaseModel):
    proposal: str
    consensus_rate: float
    iterations: int


class ProposalRunResponse(BaseModel):
    proposal: str
    consensus_rate: float
    iterations: int
    trace: Optional[Dict[str, Any]] = None


class ChangeIntent(BaseModel):
    component: str = ""
    constraints: List[str] = Field(default_factory=list)
    objective: str = ""
    raw: Dict[str, Any] = Field(default_factory=dict)


class ExpertOpinion(BaseModel):
    expert: str
    weight: float
    opinion: str


class ExpertVote(BaseModel):
    expert: str
    weight: float
    score: int
    reason: str = ""


class ProposalState(TypedDict, total=False):
    user_query: str
    change_intent: Dict[str, Any]
    retrieval_context: List[str]
    expert_weights: Dict[str, float]
    expert_opinions: List[Dict[str, Any]]
    draft_consensus: str
    expert_votes: List[Dict[str, Any]]
    consensus_rate: float
    iteration: int
    final_proposal: str
    errors: List[str]
