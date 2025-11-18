from __future__ import annotations
import logging
from typing import Optional, Dict, Any, List
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
    texts: List[str] = []
    for i, p in enumerate(pages):
        try:
            texts.append(p.extract_text() or "")
        except Exception as e:
            log.warning("Page %d could not be extracted: %s", i + 1, e)
    return "\n\n".join(texts)

# ---------- Helpers ----------

def _json_load_stripping_fences(s: str) -> Dict[str, Any]:
    import json, re
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```json\n|^```\n|```$", "", s,
                   flags=re.IGNORECASE | re.MULTILINE)
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

def _build_nuextract_prompt(template_json: str,
                            instructions: Optional[str],
                            paper_text: str) -> str:
    template_json = (template_json or "").strip()
    instr = (instructions or "").strip()
    blocks = ["# Template:", template_json]
    if instr:
        blocks += ["# Instructions:", instr]
    blocks += ["# Context:", paper_text]
    return "\n".join(blocks)

def _call_llm_minimal(client: OpenAICompatibleClient,
                      messages: List[Dict[str, str]],
                      temperature: float,
                      max_tokens: int,
                      grammar) -> str:
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

        # list merge
        if isinstance(a, list) or isinstance(v, list):
            left = a if isinstance(a, list) else ([] if a is None else [a])
            right = v if isinstance(v, list) else ([] if v is None else [v])
            seen = set()
            merged = []
            for item in left + right:
                if item is None:
                    continue
                key = ("s", item) if isinstance(item, str) else \
                      ("n", item) if isinstance(item, (int, float)) else \
                      ("o", repr(item))
                if key not in seen:
                    seen.add(key)
                    merged.append(item)
            acc[k] = merged
            continue

        # numeric majority
        if isinstance(a, (int, float)) or isinstance(v, (int, float)):
            if a is None:
                acc[k] = v
            elif v is None:
                pass
            else:
                acc[k] = Counter([a, v]).most_common(1)[0][0]
            continue

        # string prefer non-empty
        if isinstance(a, str) or isinstance(v, str):
            acc[k] = a if (isinstance(a, str) and a) else v
            continue

        # fallback prefer non-null
        if a is None and v is not None:
            acc[k] = v
    return acc

def _extract_focus_contexts(full_text: str) -> List[str]:
    lower = full_text.lower()
    cut = lower.find("introduction")
    abstract = full_text[:max(3000, min(len(full_text),
                                5000 if cut == -1 else cut))]

    lines = [ln.strip() for ln in full_text.splitlines() if ln.strip()]
    caption_like = []
    for ln in lines:
        head = ln[:12].lower()
        if head.startswith("figure") or head.startswith("table"):
            caption_like.append(ln)
        lnl = ln.lower()
        if (" uk " in f" {lnl} ") or ("united kingdom" in lnl) or \
           ("bristol" in lnl) or ("england" in lnl) or \
           ("scotland" in lnl) or ("wales" in lnl) or ("ireland" in lnl):
            caption_like.append(ln)

    windows: List[str] = []
    if abstract:
        windows.append(abstract[-4000:])
    if caption_like:
        joined = "\n".join(caption_like)
        windows.append(joined[:4000])
    return windows

# ---------- Updated Defaults ----------

DEFAULT_TEMPLATE_VERBATIM = """
{
  "pid": "string",
  "study_name": "string",
  "study_acronym": "string",
  "study_types": ["string"],
  "cohort_type": "string",
  "n_included": "integer",
  "countries": ["verbatim-string"]
}
"""

DEFAULT_TEMPLATE_NORMALIZED = """
{
  "pid": "string",
  "study_name": "string",
  "study_acronym": "string",
  "study_types": ["string"],
  "cohort_type": "string",
  "n_included": "integer",
  "countries": ["string"]
}
"""

DEFAULT_INSTRUCTIONS = (
    "Extract pid, study_name, study_acronym, study_types, cohort_type, "
    "n_included, countries. Countries must be EXACT VERBATIM names from the "
    "included participants only. Deduplicate. Exclude affiliations."
)

NORMALIZE_SUFFIX = (
    "If abbreviations like UK occur, normalize to the full country name."
)

# ---------- Public function ----------

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
) -> Dict[str, Any]:

    tpl_verbatim = (template_json or DEFAULT_TEMPLATE_VERBATIM)
    tpl_norm     = DEFAULT_TEMPLATE_NORMALIZED
    instr = (instructions or DEFAULT_INSTRUCTIONS)

    system_msg = (
        "You are NuExtract: output MUST be a single JSON object. "
        "No markdown, no text, only JSON. If unknown, use null or []."
    )

    def _run_once(text: str, tpl: str, extra_instr: str = "") -> Dict[str, Any]:
        prompt = _build_nuextract_prompt(
            tpl, (instr + ("\n" + extra_instr if extra_instr else "")), text
        )
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ]
        raw = _call_llm_minimal(
            client, messages, temperature, max_tokens,
            GRAMMAR_JSON_INT_OR_NULL if use_grammar else None,
        )
        return _json_load_stripping_fences(raw)

    # Pass 1: full
    data: Dict[str, Any] = {}
    try:
        data = _run_once(paper_text[:200000], tpl_verbatim)
    except Exception:
        data = {}

    if data.get("countries"):
        return data

    # Pass 2: windows
    for w in _extract_focus_contexts(paper_text):
        try:
            d2 = _run_once(w, tpl_verbatim)
            data = _merge_json_results(data, d2)
        except Exception:
            pass

    if data.get("countries"):
        return data

    # Pass 3: normalized
    try:
        d3 = _run_once(paper_text[:200000], tpl_norm, NORMALIZE_SUFFIX)
        data = _merge_json_results(data, d3)
    except Exception:
        pass

    if data.get("countries"):
        return data

    # Pass 4: windows normalized
    for w in _extract_focus_contexts(paper_text):
        try:
            d4 = _run_once(w, tpl_norm, NORMALIZE_SUFFIX)
            data = _merge_json_results(data, d4)
        except Exception:
            pass

    if data.get("countries"):
        return data

    # Pass 5: chunk merge
    merged = data
    for part in _chunk_text(paper_text, max_chars=chunk_chars):
        try:
            cur = _run_once(part, tpl_verbatim)
            merged = _merge_json_results(merged, cur)
        except Exception:
            pass

    if not merged.get("countries"):
        for part in _chunk_text(paper_text, max_chars=chunk_chars):
            try:
                cur = _run_once(part, tpl_norm, NORMALIZE_SUFFIX)
                merged = _merge_json_results(merged, cur)
            except Exception:
                pass

    return merged
