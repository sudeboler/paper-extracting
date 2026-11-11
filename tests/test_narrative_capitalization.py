"""Regression checks for source-preserving capitalization auditing."""

import pandas as pd

from cohort_narrative_policy import audit_narrative_capitalization


def test_historical_oncolifes_examples_are_reviewed():
    registry = {
        "tables": {
            "Collections": {
                "fields": {
                    "description": {"column_type": "text"},
                    "other exclusion criteria": {"column_type": "text"},
                },
            },
            "Subpopulations": {
                "fields": {
                    "description": {"column_type": "text"},
                    "other inclusion criteria": {"column_type": "text"},
                },
            },
        },
    }

    frames = {
        "Collections": pd.DataFrame([{
            "description": (
                "a comprehensive repository of clinical data"
            ),
            "other exclusion criteria": (
                "without further exclusion criteria"
            ),
        }]),
        "Subpopulations": pd.DataFrame([{
            "description": (
                "patients from the head and neck oncology group"
            ),
            "other inclusion criteria": "diagnosed with cancer",
        }]),
    }

    originals = {
        sheet: frame.copy(deep=True)
        for sheet, frame in frames.items()
    }

    findings = audit_narrative_capitalization(
        registry, frames
    )

    assert len([
        item for item in findings
        if item["reason"] == "cell_start"
    ]) == 4

    for sheet, original in originals.items():
        pd.testing.assert_frame_equal(
            frames[sheet], original
        )


def test_scientific_case_is_never_silently_modified():
    examples = [
        "dna repair was assessed",
        "rna sequencing was performed",
        "mrna expression was measured",
        "hiv status was recorded",
        "eGFR was calculated",
        "pH was measured",
        "iPSC-derived cells",
        "rs123 was investigated",
        "BRCA1 mutations were investigated",
        "Prof. Dr. S. de Jong contributed.",
        "The advisory board... will review the proposal.",
    ]

    registry = {
        "tables": {
            "Collections": {
                "fields": {
                    "description": {"column_type": "text"},
                },
            },
        },
    }

    frame = pd.DataFrame({"description": examples})
    original = frame.copy(deep=True)

    findings = audit_narrative_capitalization(
        registry, {"Collections": frame}
    )

    assert isinstance(findings, list)

    pd.testing.assert_frame_equal(frame, original)


def test_non_narrative_fields_are_not_flagged():
    registry = {
        "tables": {
            "Collections": {
                "fields": {
                    "description": {"column_type": "text"},
                    "name": {"column_type": "text"},
                },
            },
            "Publications": {
                "fields": {
                    "title": {"column_type": "text"},
                },
            },
        },
    }

    frames = {
        "Collections": pd.DataFrame([{
            "description": "patients were enrolled",
            "name": "oncoLifeS",
        }]),
        "Publications": pd.DataFrame([{
            "title": "lowercase publication title",
        }]),
    }

    findings = audit_narrative_capitalization(
        registry, frames
    )

    assert len(findings) == 1
    assert findings[0]["sheet"] == "Collections"
    assert findings[0]["column"] == "description"