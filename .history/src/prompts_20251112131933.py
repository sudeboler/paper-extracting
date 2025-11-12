# NuExtract-conform prompts (text-only). 
# We mirror the model card's style: first "# Template:", then the JSON template, then "# Context:".

SYSTEM_EXTRACTOR = (
    "You are NuExtract: extract structured data as JSON only. "
    "Output MUST be a single JSON object, no prose, no markdown fences. "
    "If a field is not explicitly supported by the context, return null (or [] for arrays). "
    "Use only double quotes, valid UTF-8, and no trailing commas."
)

# Minimal template for the test task
TEMPLATE_N_INCLUDED = '{"n_included": "integer"}'

def build_user_prompt_n_included(paper_text: str) -> str:
    """
    Build the exact NuExtract chat text expected by the model card:
    # Template:
    {JSON template}
    # Context:
    {text}
    """
    return (
        "# Template:\n"
        f"{TEMPLATE_N_INCLUDED}\n"
        "# Context:\n"
        f"{paper_text}"
    )
