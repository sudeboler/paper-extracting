# Optional EBNF-like grammar for llama.cpp 'grammar' field to force a minimal JSON object
# of the shape: {"n_included": <number|null>}

GRAMMAR_JSON_INT_OR_NULL = r'''
root        ::= object
object      ::= "{" ws pair ws "}"
pair        ::= string_n field_sep value ws
field_sep   ::= ws ":" ws
value       ::= number | null

string_n    ::= '"' "n_included" '"'
number      ::= ["-+"]? digit+
null        ::= "null"

ws          ::= (" " | "\n" | "\r" | "\t")*
digit       ::= [0-9]
'''
