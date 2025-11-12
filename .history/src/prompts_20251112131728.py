SYSTEM_EXTRACTOR = (
    "You are a meticulous information extractor. Respond ONLY with valid JSON."
)

# NuExtract expects a JSON template that declares types.
# We'll ask for an integer field "n_included".
NUEXTRACT_TEMPLATE_N_INCLUDED = '{"n_included": "integer"}'

# Full user message: include a short header, the template, then the context text.
USER_TEMPLATE_N_INCLUDED = (
    "# Template:\n"
    f"{NUEXTRACT_TEMPLATE_N_INCLUDED}\n"
    "# Context:\n"
    "{paper_text}"
)
