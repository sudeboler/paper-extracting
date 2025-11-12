from __future__ import annotations
import json
import logging
from typing import Optional, Dict, Any, List
import requests

log = logging.getLogger(__name__)

class OpenAICompatibleClient:
    """Thin client for OpenAI-compatible /v1/chat/completions endpoints.
    Works with local llama.cpp servers that expose an OpenAI-compatible API.
    Optionally supports a top-level 'grammar' field (llama.cpp feature).
    """

    def __init__(self, base_url: str, api_key: str, model: str, use_grammar: bool = False):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.use_grammar = use_grammar
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        })

    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 256,
        response_format: Optional[Dict[str, Any]] = None,
        grammar: Optional[str] = None,
    ) -> str:
        url = f"{self.base_url}/chat/completions"
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            # Many OpenAI-compatible servers support at least {"type": "json_object"}
            payload["response_format"] = response_format
        if self.use_grammar and grammar:
            # llama.cpp accepts a top-level 'grammar' string (EBNF-like)
            payload["grammar"] = grammar

        log.debug("POST %s payload=%s", url, json.dumps(payload)[:500])
        resp = self.session.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        # Standard OpenAI response shape
        content = data["choices"][0]["message"]["content"]
        return content
