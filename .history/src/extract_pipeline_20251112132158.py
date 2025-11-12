from __future__ import annotations
import logging
import re
from typing import Optional, Dict, Any
from pypdf import PdfReader

# If you have llm_client.py in src/, this import should work:
from .llm_client import OpenAICompatibleClient

# Grammar is optional; we try to import it. If missing, we just skip it.
try:
    from .llm_grammar import GRAMMAR_JSON_INT_OR_NULL
except Exception:
    GRAMMAR_JSON_INT_OR_NULL = None  # type: ignore

log = logging.getLogger(__name__)

# -------- PDF utilities --------

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

# -------- Heuristic fallback (EN-only) --------

def quick_regex_peek_for_inclusions(text: str) -> Optional[int]:
    """
    Quick heuristic to help the LLM (or serve as a fallback).
    English-only patterns by request.
    """
    patterns = [
        r"included\s*\(\s*n\s*=\s*(\d+)\s*\)",  # "included (n=123)"
        r"included\s+(\d+)",                     # "included 123"
        r"n\s*=\s*(\d+)\s*included",             # "n=123 included"
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                pass
    return None

# -------- NuExtract prompt helpers (text-only) --------

def build_nuextract_prompt_n_included(paper_text: str) -> str:
    """
    Build the exact NuExtract chat text expected by the model card:
    # Template:
    {JSON template}
    # Context:
    {text}
    """
    template = '{"n_included": "integer"}'
    return (
        "# Template:\n"
        f"{template}\n"
        "# Context:\n"
        f"{paper_text}"
    )

# -------- Main extraction entrypoint --------

def extract_n_included(
    client: OpenAICompatibleClient,
    paper_text: str,
    *,
    use_grammar: bool,
    temperature: float,
    max_tokens: int,
    use_vllm_template: bool = False,
) -> Dict[str, Any]:
    """
    Extract {"n_included": int|null} using NuExtract formatting.
    - English-only prompts/content.
    - temperature close to 0 is recommended for NuExtract.
    - If use_vllm_template=True, we also pass the template via extra_body.chat_template_kwargs.template
      (useful when serving with vLLM). For llama.cpp this is not required.
    """
    regex_hint = quick_regex_peek_for_inclusions(paper_text)

    user_msg = build_nuextract_prompt_n_included(paper_text=paper_text[:100000])
    messages = [
        {"role": "system", "content": (
            "You are NuExtract: extract structured data as JSON only. "
            "Output MUST be a single JSON object, no prose, no markdown fences. "
            "If a field is not supported by the context, return null. "
            "Use only double quotes, valid UTF-8, and no trailing commas."
        )},
        {"role": "user", "content": user_msg},
    ]

    extra_body = None
    if use_vllm_template:
        extra_body = {
            "chat_template_kwargs": {
                "template": '{"n_included": "integer"}',
            }
        }

    raw = client.chat(
        messages,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},  # harmless if ignored
        grammar=(GRAMMAR_JSON_INT_OR_NULL if (use_grammar and GRAMMAR_JSON_INT_OR_NULL) else None),
        extra_body=extra_body,
    )

    try:
        data = json_safe_load(raw)
    except Exception as e:
        log.warning("JSON parse failed (%s), falling back to heuristic.", e)
        data = {"n_included": regex_hint}

    if (data.get("n_included") is None) and (regex_hint is not None):
        data["n_included"] = regex_hint

    return data

# -------- JSON helper --------

def json_safe_load(s: str):
    import json
    s = s.strip()
    # Some servers wrap with ```json fences; strip them
    if s.startswith("```"):
        s = re.sub(r"^```json\n|^```\n|```$", "", s, flags=re.IGNORECASE | re.MULTILINE)
    return json.loads(s)
