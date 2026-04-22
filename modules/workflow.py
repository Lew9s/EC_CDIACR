import asyncio
import json
from typing import Any, Dict, List

try:
    from langgraph.graph import END, START, StateGraph
except ImportError:
    END = "__end__"
    START = "__start__"
    StateGraph = None

from config import settings
from llm_client import OllamaChatClient, parse_json_object
from schemas import ProposalState


EXPERTS: Dict[str, str] = {
    "structure": "结构设计专家",
    "outfitting": "舾装工艺专家",
    "quality": "质量与规范专家",
    "electrical": "电气系统专家",
    "machinery": "轮机系统专家",
    "material": "材料与焊接专家",
}


def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    normalized = {expert: max(float(weights.get(expert, 0.0)), 0.0) for expert in EXPERTS}
    total = sum(normalized.values())
    if total <= 0:
        equal = round(1.0 / len(EXPERTS), 6)
        normalized = {expert: equal for expert in EXPERTS}
        last_key = next(reversed(normalized))
        normalized[last_key] = round(normalized[last_key] + 1.0 - sum(normalized.values()), 6)
        return normalized

    normalized = {expert: round(value / total, 6) for expert, value in normalized.items()}
    last_key = next(reversed(normalized))
    normalized[last_key] = round(normalized[last_key] + 1.0 - sum(normalized.values()), 6)
    return normalized


def calculate_consensus_rate(votes: List[Dict[str, Any]]) -> float:
    return round(
        sum(float(vote.get("weight", 0.0)) * int(vote.get("score", 0)) for vote in votes),
        4,
    )


def should_continue_consensus(state: ProposalState) -> str:
    if state.get("consensus_rate", 0.0) >= settings.consensus_threshold:
        return "generate_proposal"
    if state.get("iteration", 0) >= settings.max_consensus_rounds:
        return "generate_proposal"
    return "expert_review"


def run_async(coro):
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro)


class SequentialWorkflow:
    def __init__(self, workflow: "ProposalWorkflow") -> None:
        self.workflow = workflow

    def invoke(self, state: ProposalState) -> ProposalState:
        state = self.workflow.parse_intent(state)
        state = self.workflow.retrieve_cases(state)
        state = self.workflow.select_experts(state)
        while True:
            state = self.workflow.expert_review(state)
            state = self.workflow.merge_consensus(state)
            state = self.workflow.vote_consensus(state)
            if should_continue_consensus(state) == "generate_proposal":
                break
        return self.workflow.generate_proposal(state)


class ProposalWorkflow:
    def __init__(self, rag_service, llm_client: OllamaChatClient | None = None) -> None:
        self.rag_service = rag_service
        self.llm = llm_client or OllamaChatClient()
        self.graph = self._compile_graph()

    def _compile_graph(self):
        if StateGraph is None:
            return SequentialWorkflow(self)

        graph = StateGraph(ProposalState)
        graph.add_node("parse_intent", self.parse_intent)
        graph.add_node("retrieve_cases", self.retrieve_cases)
        graph.add_node("select_experts", self.select_experts)
        graph.add_node("expert_review", self.expert_review)
        graph.add_node("merge_consensus", self.merge_consensus)
        graph.add_node("vote_consensus", self.vote_consensus)
        graph.add_node("generate_proposal", self.generate_proposal)

        graph.add_edge(START, "parse_intent")
        graph.add_edge("parse_intent", "retrieve_cases")
        graph.add_edge("retrieve_cases", "select_experts")
        graph.add_edge("select_experts", "expert_review")
        graph.add_edge("expert_review", "merge_consensus")
        graph.add_edge("merge_consensus", "vote_consensus")
        graph.add_conditional_edges(
            "vote_consensus",
            should_continue_consensus,
            {
                "expert_review": "expert_review",
                "generate_proposal": "generate_proposal",
            },
        )
        graph.add_edge("generate_proposal", END)
        return graph.compile()

    def parse_intent(self, state: ProposalState) -> ProposalState:
        prompt = f"""
你是一名专业的工程变更需求解析专家。
请从用户输入中提取结构化变更意图，只输出 JSON。
字段：
- component: 变更主体
- constraints: 约束条件数组
- objective: 变更目标

用户输入：{state["user_query"]}
"""
        parsed = parse_json_object(self.llm.complete(prompt, json_mode=True))
        intent = {
            "component": parsed.get("component", ""),
            "constraints": parsed.get("constraints", []),
            "objective": parsed.get("objective", ""),
            "raw": parsed,
        }
        return {**state, "change_intent": intent}

    def retrieve_cases(self, state: ProposalState) -> ProposalState:
        question = state.get("change_intent", {}).get("component") or state["user_query"]
        context = run_async(self.rag_service.retrieve(question))
        return {**state, "retrieval_context": context}

    def select_experts(self, state: ProposalState) -> ProposalState:
        prompt = f"""
你是在船舶制造工程变更系统中的专家调度器。
请基于变更意图为以下专家分配权重，权重总和为 1；无关专家为 0。
专家键名：{", ".join(EXPERTS.keys())}
只输出 JSON 对象，例如 {{"structure":0.3,"outfitting":0.2}}。

变更意图：{json.dumps(state.get("change_intent", {}), ensure_ascii=False)}
用户需求：{state["user_query"]}
"""
        parsed = parse_json_object(self.llm.complete(prompt, json_mode=True))
        return {**state, "expert_weights": normalize_weights(parsed)}

    def expert_review(self, state: ProposalState) -> ProposalState:
        opinions: List[Dict[str, Any]] = []
        previous_consensus = state.get("draft_consensus", "")
        context = "\n".join(state.get("retrieval_context", []))

        for expert_key, weight in state.get("expert_weights", {}).items():
            if weight <= 0:
                continue
            expert_name = EXPERTS[expert_key]
            prompt = f"""
【Role】你是一名{expert_name}。
【Task】基于历史案例、用户需求和上一轮共识，从本专业角度评估工程变更。
【要求】指出可行性、风险、约束和建议；不要编造历史案例或标准。

用户需求：{state["user_query"]}
变更意图：{json.dumps(state.get("change_intent", {}), ensure_ascii=False)}
历史案例：{context}
上一轮共识：{previous_consensus or "无"}
"""
            opinions.append(
                {
                    "expert": expert_key,
                    "expert_name": expert_name,
                    "weight": weight,
                    "opinion": self.llm.complete(prompt),
                }
            )

        return {
            **state,
            "iteration": state.get("iteration", 0) + 1,
            "expert_opinions": opinions,
        }

    def merge_consensus(self, state: ProposalState) -> ProposalState:
        opinions = json.dumps(state.get("expert_opinions", []), ensure_ascii=False)
        prompt = f"""
你是在多智能体工程变更协同系统中的整合智能体。
请将各专家意见融合为一份连贯的共识草案，识别专业冲突、信息缺失和关键约束。
不要新增专家未提出且历史案例未支持的事实。

专家意见：{opinions}
"""
        return {**state, "draft_consensus": self.llm.complete(prompt)}

    def vote_consensus(self, state: ProposalState) -> ProposalState:
        votes: List[Dict[str, Any]] = []
        consensus = state.get("draft_consensus", "")
        opinions_by_expert = {
            opinion["expert"]: opinion for opinion in state.get("expert_opinions", [])
        }

        for expert_key, weight in state.get("expert_weights", {}).items():
            if weight <= 0:
                continue
            original = opinions_by_expert.get(expert_key, {}).get("opinion", "")
            prompt = f"""
你是在多智能体工程变更协同系统中参与共识评估的专家智能体。
请判断共识草案是否准确反映你此前的专业意见，输出 JSON：
{{"consensus_score": 0 或 1, "reason": "简短原因"}}

此前意见：{original}
共识草案：{consensus}
"""
            parsed = parse_json_object(self.llm.complete(prompt, json_mode=True))
            try:
                score = 1 if int(parsed.get("consensus_score", 0)) == 1 else 0
            except (TypeError, ValueError):
                score = 0
            votes.append(
                {
                    "expert": expert_key,
                    "weight": weight,
                    "score": score,
                    "reason": parsed.get("reason", ""),
                }
            )

        return {
            **state,
            "expert_votes": votes,
            "consensus_rate": calculate_consensus_rate(votes),
        }

    def generate_proposal(self, state: ProposalState) -> ProposalState:
        context = "\n".join(state.get("retrieval_context", []))
        prompt = f"""
你是在船舶制造工程变更系统中的最终提案起草人。
请生成工程变更提案，必须包含四章：变更背景、历史案例、具体措施、注意事项。
所有建议必须由检索上下文或专家共识支撑，不得虚构部门名、时间、人名或标准。

用户需求：{state["user_query"]}
历史案例：{context}
专家共识：{state.get("draft_consensus", "")}
共识率：{state.get("consensus_rate", 0.0)}
"""
        return {**state, "final_proposal": self.llm.complete(prompt)}
