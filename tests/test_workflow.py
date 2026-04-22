import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODULES = os.path.join(ROOT, "modules")
sys.path.insert(0, MODULES)

from llm_client import parse_json_object
from workflow import (
    ProposalWorkflow,
    calculate_consensus_rate,
    normalize_weights,
    should_continue_consensus,
)


class StaticRAGService:
    async def retrieve(self, question: str):
        return ["历史案例：主甲板人孔调整"]


class DeterministicLLM:
    def complete(self, prompt: str, json_mode: bool = False) -> str:
        if "字段：" in prompt and json_mode:
            return '{"component":"主甲板人孔","constraints":["DIN 83402"],"objective":"调整安装方案"}'
        if "专家调度器" in prompt:
            return '{"structure":0.4,"outfitting":0.3,"quality":0.3}'
        if "consensus_score" in prompt:
            return '{"consensus_score":1,"reason":"保留核心约束"}'
        if "最终提案起草人" in prompt:
            return "一、变更背景\n...\n二、历史案例\n...\n三、具体措施\n...\n四、注意事项\n..."
        if "整合智能体" in prompt:
            return "专家共识草案"
        return "专家意见"


def test_parse_json_object_cleans_think_tags():
    parsed = parse_json_object('<think>draft</think>{"consensus_score":1}')
    assert parsed == {"consensus_score": 1}


def test_normalize_weights_sum_to_one():
    weights = normalize_weights({"structure": 2, "outfitting": 1})
    assert set(weights) >= {"structure", "outfitting", "quality"}
    assert round(sum(weights.values()), 6) == 1.0
    assert weights["structure"] > weights["outfitting"]


def test_calculate_consensus_rate():
    votes = [
        {"weight": 0.4, "score": 1},
        {"weight": 0.3, "score": 0},
        {"weight": 0.3, "score": 1},
    ]
    assert calculate_consensus_rate(votes) == 0.7


def test_route_consensus_finishes_on_threshold():
    state = {"consensus_rate": 0.9, "iteration": 1}
    assert should_continue_consensus(state) == "generate_proposal"


def test_full_workflow_with_mock_llm_and_retriever():
    workflow = ProposalWorkflow(
        StaticRAGService(),
        llm_client=DeterministicLLM(),
    )
    result = workflow.graph.invoke(
        {
            "user_query": "调整主甲板人孔，满足 DIN 83402",
            "iteration": 0,
            "errors": [],
            "retrieval_context": [],
            "expert_opinions": [],
            "expert_votes": [],
            "consensus_rate": 0.0,
        }
    )
    assert result["consensus_rate"] == 1.0
    assert result["iteration"] == 1
    assert "变更背景" in result["final_proposal"]
