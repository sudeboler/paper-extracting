from __future__ import annotations
import logging
import re
from typing import Optional, Dict, Any
from pypdf import PdfReader

from .llm_client import OpenAICompatibleClient
from . import prompts
from .llm_grammar import GRAMMAR_JSON_INT_OR_NULL

log = logging.getLogger(__name__)

def load_pdf_text(path: str, max_pages: Optional[int] = None) -> str:
    reader = PdfReader(path)
    pages = reader.pages[:max_pages] if max_pages else reader.pages
    texts = []
    for i, p in enumerate(pages):
        try:
            texts.append(p.extract_text() or "")
        except Exception as e:
            log.warning("Page %d could not be extracted: %s", i + 1, e)
    return "\n\n".join(texts)

def quick_regex_peek_for_inclusions(text: str) -> Optional[int]:
    """Quick heuristic to help the LLM (or serve as a fallback)."""
    patterns = [
        r"included\s*\(\s*n\s*=\s*(\d+)\s*\)",
        r"included\s+(\d+)",
        r"geïncludeerd\s*(\d+)",
        r"geïncludeerd\s*\(\s*n\s*=\s*(\d+)\)",
        r"inclusies?\s*:\s*(\d+)",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                pass
    return None

def extract_n_included(
    client: OpenAICompatibleClient,
    paper_text: str,
    *,
    use_grammar: bool,
    temperature: float,
    max_tokens: int,
    use_vllm_template: bool = False,
) -> Dict[str, Any]:
    """If use_vllm_template=True, send `extra_body.chat_template_kwargs.template` for vLLM servers.
    Otherwise we inline the template in the user message (works for llama.cpp and most servers).
    """
    regex_hint = quick_regex_peek_for_inclusions(paper_text)

    user_msg = prompts.USER_TEMPLATE_N_INCLUDED.format(paper_text=paper_text[:100000])
    messages = [
        {"role": "system", "content": prompts.SYSTEM_EXTRACTOR},
        {"role": "user", "content": user_msg},
    ]

    extra_body = None
    if use_vllm_template:
        # vLLM NuExtract chat template hook (optional). We still inline the content for portability.
        extra_body = {
            "chat_template_kwargs": {
                "template": prompts.NUEXTRACT_TEMPLATE_N_INCLUDED,
            }
        }

    raw = client.chat(
        messages,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},  # many servers honor this; harmless if ignored
        grammar=GRAMMAR_JSON_INT_OR_NULL if use_grammar else None,
        extra_body=extra_body,
    )

    # Post-parse
    try:
        data = json_safe_load(raw)
    except Exception as e:
        log.warning("JSON parse failed (%s), falling back to heuristic.", e)
        data = {"n_included": regex_hint}

    if (data.get("n_included") is None) and (regex_hint is not None):
        data["n_included"] = regex_hint

    return data

def json_safe_load(s: str):
    import json, re
    s = s.strip()
    # Some servers wrap with ```json fences; strip them
    if s.startswith("```"):
        s = re.sub(r"^```json\n|^```\n|```$", "", s, flags=re.IGNORECASE | re.MULTILINE)
    return json.loads(s)
