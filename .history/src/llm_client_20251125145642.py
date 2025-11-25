from __future__ import annotations
import json, logging
from typing import Optional, Dict, Any, List
import requests

log = logging.getLogger(__name__)

def _with_v1(base: str) -> str:
    base = base.rstrip("/")
    return base if base.endswith("/v1") else base + "/v1"

class OpenAICompatibleClient:
    def __init__(self, base_url: str, api_key: str, model: str, use_grammar: bool = False):
        self.base_url = _with_v1(base_url)
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
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> str:
        url = f"{self.base_url}/chat/completions"
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            payload["response_format"] = response_format
        if self.use_grammar and grammar:
            payload["grammar"] = grammar
        if extra_body:
            payload.update(extra_body)

        def _post(data: Dict[str, Any]) -> requests.Response:
            r = self.session.post(url, json=data, timeout=600)
            if r.status_code >= 400:
                try:
                    log.error("LLM %s: %s", r.status_code, r.text[:2000])
                except Exception:
                    pass
            return r

        resp = _post(payload)

        if resp.status_code == 400:
            stripped = dict(payload)
            stripped.pop("response_format", None)
            stripped.pop("grammar", None)
            resp = _post(stripped)

        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
