SYSTEM_EXTRACTOR = (
    "You are a meticulous information extractor. Respond ONLY with valid JSON that matches the schema."
)

# Test task: extract the number of included items (n_included) from the paper text.
# Do not guess. If it is not explicitly stated, return null.
USER_TEMPLATE_N_INCLUDED = (
    "You will receive the full (or summarized) text of a scientific article.\n"
    "Return the number of inclusions the study reports (n_included).\n"
    "If not explicitly stated, do NOT estimate: return null.\n\n"
    "Text:\n\n{paper_text}"
)

# JSON guidance (for humans)
JSON_GUIDANCE = {
    "n_included": "int | null"
}
