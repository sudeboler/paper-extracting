from __future__ import annotations
import logging
from typing import Optional, Dict, Any, List, Tuple
from pypdf import PdfReader

from .llm_client import OpenAICompatibleClient

try:
    from .llm_grammar import GRAMMAR_JSON_INT_OR_NULL
except Exception:
    GRAMMAR_JSON_INT_OR_NULL = None  # type: ignore

log = logging.getLogger(__name__)

# ---------- PDF ----------

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

def load_pdf_pages(path: str, max_pages: Optional[int] = None) -> List[str]:
    """Per-pagina tekst (geen regex)."""
    reader = PdfReader(path)
    pages = reader.pages[:max_pages] if max_pages else reader.pages
    out: List[str] = []
    for i, p in enumerate(pages):
        try:
            out.append(p.extract_text() or "")
        except Exception as e:
            log.warning("Page %d could not be extracted: %s", i + 1, e)
            out.append("")
    return out

# ---------- Helpers ----------

def _json_load_stripping_fences(s: str) -> Dict[str, Any]:
    import json, re
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```json\n|^```\n|```$", "", s, flags=re.IGNORECASE | re.MULTILINE)
    return json.loads(s)

def _chunk_text(txt: str, max_chars: int = 40000):
    start = 0
    L = len(txt)
    while start < L:
        end = min(L, start + max_chars)
        if end < L:
            nl = txt.rfind("\n\n", start, end)
            if nl != -1 and nl > start + 2000:
                end = nl
        yield txt[start:end]
        start = end

def _build_nuextract_prompt(template_json: str, instructions: Optional[str], paper_text: str) -> str:
    template_json = (template_json or "").strip()
    instr = (instructions or "").strip()
    blocks = ["# Template:", template_json]
    if instr:
        blocks += ["# Instructions:", instr]
    blocks += ["# Context:", paper_text]
    return "\n".join(blocks)

def _call_llm_minimal(
    client: OpenAICompatibleClient,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    grammar,
) -> str:
    return client.chat(
        messages,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format=None,
        grammar=(grammar if grammar else None),
    )

def _merge_json_results(acc: Dict[str, Any], cur: Dict[str, Any]) -> Dict[str, Any]:
    from collections import Counter
    if not acc:
        return dict(cur)
    for k, v in cur.items():
        if k not in acc:
            acc[k] = v
            continue
        a = acc[k]
        if isinstance(a, list) or isinstance(v, list):
            left = a if isinstance(a, list) else ([] if a is None else [a])
            right = v if isinstance(v, list) else ([] if v is None else [v])
            seen = set()
            merged = []
            for item in left + right:
                if item is None:
                    continue
                key = ("s", item) if isinstance(item, str) else ("n", item) if isinstance(item, (int, float)) else ("o", repr(item))
                if key not in seen:
                    seen.add(key)
                    merged.append(item)
            acc[k] = merged
            continue
        if isinstance(a, (int, float)) or isinstance(v, (int, float)):
            if a is None:
                acc[k] = v
            elif v is None:
                pass
            else:
                acc[k] = Counter([a, v]).most_common(1)[0][0]
            continue
        if isinstance(a, str) or isinstance(v, str):
            acc[k] = a if (isinstance(a, str) and a) else v
            continue
        if a is None and v is not None:
            acc[k] = v
    return acc

# ---------- Defaults (edit these as you add fields) ----------

DEFAULT_TEMPLATE = '{"n_included": "integer", "countries": ["verbatim-string"]}'

DEFAULT_INSTRUCTIONS = (
    "Extract the number of INCLUDED participants and the list of countries where the INCLUDED participants came from.\n"
    "- Include ONLY countries for the final included cohort.\n"
    "- EXCLUDE countries for screened/excluded/eligible-but-not-included participants, non-contributing sites, and author affiliations.\n"
    '- Return each country name verbatim (e.g., "United Kingdom of Great Britain and Northern Ireland (the)").\n'
    "- Deduplicate; order does not matter."
)

RETRY_INSTRUCTIONS_SUFFIX = (
    "If the context contains country names for the included cohort, you MUST return them in `countries`.\n"
    "Do not return an empty list when such country names are present.\n"
    "If none are present for the included cohort, return an empty list.\n"
)

# ---------- Single public function ----------

def extract_fields(
    client: OpenAICompatibleClient,
    paper_text: str,
    *,
    template_json: Optional[str] = None,
    instructions: Optional[str] = None,
    use_grammar: bool,
    temperature: float,
    max_tokens: int,
    chunk_chars: int = 40000,
    pdf_path_for_pages: Optional[str] = None,
    pdf_max_pages: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Generic extractor. Edit `template_json` / `instructions` to add fields.
    If `countries` is empty after full/chunk pass, we do a page-wise retry (no regex).
    """
    tpl = (template_json or DEFAULT_TEMPLATE)
    instr = (instructions or DEFAULT_INSTRUCTIONS)

    system_msg = (
        "You are NuExtract: extract structured data as JSON only. "
        "Output MUST be a single JSON object, no prose, no markdown fences. "
        "If a field is not supported by the context, return null or []. "
        "Use only double quotes, valid UTF-8, and no trailing commas."
    )

    # Pass 1: full text (capped)
    full_prompt = _build_nuextract_prompt(tpl, instr, paper_text[:200000])
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": full_prompt},
    ]

    data: Dict[str, Any] = {}
    try:
        raw = _call_llm_minimal(
            client, messages, temperature, max_tokens,
            GRAMMAR_JSON_INT_OR_NULL if use_grammar else None
        )
        data = _json_load_stripping_fences(raw)
    except Exception as e:
        log.warning("Full-text LLM call failed (%s).", e)
        data = {}

    if isinstance(data, dict) and data.get("countries"):
        return data

    # Pass 2: chunk fallback (merge)
    merged: Dict[str, Any] = data if isinstance(data, dict) else {}
    for part in _chunk_text(paper_text, max_chars=chunk_chars):
        part_prompt = _build_nuextract_prompt(tpl, instr, part)
        part_messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": part_prompt},
        ]
        try:
            raw = _call_llm_minimal(
                client, part_messages, temperature, max_tokens,
                GRAMMAR_JSON_INT_OR_NULL if use_grammar else None
            )
            cur = _json_load_stripping_fences(raw)
            if isinstance(cur, dict):
                merged = _merge_json_results(merged, cur)
        except Exception as ex:
            log.debug("Chunk failed: %s", ex)

    if merged.get("countries"):
        return merged

    # Pass 3: per-page retry (still LLM-only, no regex), if we have pdf path
    if pdf_path_for_pages:
        try:
            pages = load_pdf_pages(pdf_path_for_pages, max_pages=pdf_max_pages)
        except Exception as e:
            log.debug("Page retry skipped (load error): %s", e)
            pages = []

        for i, page_text in enumerate(pages, start=1):
            # stricter instructions for page retries
            page_instr = (instr + "\n" + RETRY_INSTRUCTIONS_SUFFIX).strip()
            page_prompt = _build_nuextract_prompt(tpl, page_instr, page_text)
            page_messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": page_prompt},
            ]
            try:
                rawp = _call_llm_minimal(
                    client, page_messages, temperature, max_tokens,
                    GRAMMAR_JSON_INT_OR_NULL if use_grammar else None
                )
                dp = _json_load_stripping_fences(rawp)
                if isinstance(dp, dict):
                    merged = _merge_json_results(merged, dp)
                if merged.get("countries"):
                    break
            except Exception as ex:
                log.debug("Page %d retry failed: %s", i, ex)

    return merged
