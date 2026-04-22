import json
import re
from typing import Any, Dict, List

import ollama

from config import settings


def clean_llm_output(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL)
    text = text.strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group() if match else text


def parse_json_object(text: str) -> Dict[str, Any]:
    cleaned = clean_llm_output(text)
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


class OllamaChatClient:
    def __init__(self, model: str | None = None) -> None:
        self.model = model or settings.ollama_model

    def complete(self, prompt: str, json_mode: bool = False) -> str:
        response = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            format="json" if json_mode else None,
        )
        return response.get("message", {}).get("content", "")


class FakeChatClient:
    def __init__(self, responses: Dict[str, str] | None = None) -> None:
        self.responses = responses or {}
        self.prompts: List[str] = []

    def complete(self, prompt: str, json_mode: bool = False) -> str:
        self.prompts.append(prompt)
        for key, value in self.responses.items():
            if key in prompt:
                return value
        if json_mode:
            return "{}"
        return "测试响应"
