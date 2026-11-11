import pandas as pd

from cohort_dynamic_prompts import replace_task_sections
from cohort_narrative_policy import (
    NARRATIVE_GUIDANCE_MARKER,
    append_narrative_guidance,
    eligible_narrative_fields,
    audit_narrative_capitalization,
)


def test_guidance_preserves_templates_and_is_idempotent():
    cfg = {
        "llm": {"model": "test-model"},
        "task_overview": {
            "instructions": "Extract cohort information.",
            "template_json": '{"description": null}',
        },
    }

    assert append_narrative_guidance(cfg) == 1
    assert append_narrative_guidance(cfg) == 0

    task = cfg["task_overview"]
    assert task["instructions"].count(NARRATIVE_GUIDANCE_MARKER) == 1
    assert task["template_json"] == '{"description": null}'
    assert cfg["llm"] == {"model": "test-model"}


def test_guidance_after_dynamic_task_replacement():
    cfg = {
        "task_overview": {
            "instructions": "Original task.",
            "template_json": "{}",
        },
    }

    append_narrative_guidance(cfg)

    generated = {
        "task_overview": {
            "instructions": "Generated extraction task.",
            "template_json": '{"description": null}',
        },
    }

    replace_task_sections(cfg, generated)

    assert NARRATIVE_GUIDANCE_MARKER not in (
        cfg["task_overview"]["instructions"]
    )

    assert append_narrative_guidance(cfg) == 1
    assert NARRATIVE_GUIDANCE_MARKER in (
        cfg["task_overview"]["instructions"]
    )
    assert cfg["task_overview"]["template_json"] == (
        '{"description": null}'
    )


def test_schema_restricts_narrative_fields():
    registry = {
        "tables": {
            "Collections": {
                "extends": "Resources",
                "fields": {
                    "description": {"column_type": "text"},
                    "other inclusion criteria": {"column_type": "text"},
                    "acronym": {"column_type": "text"},
                    "type": {"column_type": "ontology_array"},
                },
            },
            "Publications": {
                "fields": {
                    "title": {"column_type": "text"},
                },
            },
            "Subpopulations": {
                "fields": {
                    "description": {"column_type": "string"},
                },
            },
        },
    }

    selected = eligible_narrative_fields(registry)

    assert selected["Collections"] == frozenset({
        "description",
        "other inclusion criteria",
    })
    assert "Publications" not in selected
    assert "Subpopulations" not in selected
    assert eligible_narrative_fields(None) == {}


def test_historical_oncolifes_lowercase_cases_are_detected():
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
        "Collections": pd.DataFrame({
            "description": [
                "a comprehensive repository of clinical data"
            ],
            "other exclusion criteria": [
                "without further exclusion criteria"
            ],
        }),
        "Subpopulations": pd.DataFrame({
            "description": (
                [""] * 3
                + ["patients from the head and neck oncology group"]
                + [""] * 4
                + ["patients diagnosed with hereditary breast-ovarian cancer"]
            ),
            "other inclusion criteria": [
                "diagnosed with cancer"
            ] * 9,
        }),
    }

    originals = {
        sheet: frame.copy(deep=True)
        for sheet, frame in frames.items()
    }

    findings = audit_narrative_capitalization(registry, frames)

    cell_starts = [
        finding for finding in findings
        if finding["reason"] == "cell_start"
    ]

    assert len(cell_starts) == 13

    for sheet, frame in frames.items():
        pd.testing.assert_frame_equal(frame, originals[sheet])


def test_scientific_text_and_ambiguous_boundaries_are_preserved():
    registry = {
        "tables": {
            "Collections": {
                "fields": {
                    "description": {"column_type": "text"},
                    "acronym": {"column_type": "text"},
                },
            },
        },
    }

    texts = [
        "Patients were enrolled. follow-up was performed.",
        "Prof. Dr. S. de Jong and Dr. A.J. van der Wekken",
        "The advisory board... will review the proposal.",
        "eGFR was calculated.",
        "pH was measured.",
        "mRNA expression was assessed.",
        "dna repair was investigated.",
    ]

    frame = pd.DataFrame({
        "description": texts,
        "acronym": ["lowercase"] * len(texts),
    })

    original = frame.copy(deep=True)

    findings = audit_narrative_capitalization(
        registry, {"Collections": frame}
    )

    assert any(
        item["reason"] == "possible_sentence_start"
        for item in findings
    )

    assert any(
        item["original"].startswith("eGFR")
        and item["reason"] == "cell_start"
        for item in findings
    )

    assert all(item["column"] == "description" for item in findings)

    pd.testing.assert_frame_equal(frame, original)


def test_missing_registry_does_not_change_metadata():
    frame = pd.DataFrame({
        "description": ["patients were enrolled"],
    })
    original = frame.copy(deep=True)

    findings = audit_narrative_capitalization(
        None, {"Collections": frame}
    )

    assert findings == []
    pd.testing.assert_frame_equal(frame, original)


def test_real_dynamic_generator_preserves_narrative_guidance():
    """Verify the real dynamic prompt construction pathway.

    Uses a synthetic local EMX2 registry; no network or LLM calls.
    """
    from copy import deepcopy

    from cohort_dynamic_prompts import (
        build_dynamic_task_sections,
        replace_task_sections,
    )
    from emx2_dynamic_runtime import (
        apply_dynamic_constraints_to_config,
    )

    registry = {
        "profile": "UMCGCohortsStaging",
        "tables": {
            "Resources": {
                "fields": {
                    "id": {"column_type": "string"},
                    "description": {"column_type": "text"},
                },
            },
            "Collections": {
                "extends": "Resources",
                "fields": {
                    "id": {"column_type": "string"},
                    "description": {"column_type": "text"},
                    "other inclusion criteria": {
                        "column_type": "text",
                    },
                },
            },
            "Subpopulations": {
                "fields": {
                    "id": {"column_type": "string"},
                    "description": {"column_type": "text"},
                },
            },
        },
        "sources": [],
    }

    cfg = {
        "llm": {"model": "test-model"},
        "task_old": {
            "instructions": "Obsolete extraction task.",
            "template_json": "{}",
        },
    }

    # Step 1: Generate tasks using the actual model code.
    generated = build_dynamic_task_sections(registry)

    assert generated
    assert "task_overview" in generated
    assert "task_subpopulations" in generated

    # Step 2: Replace outdated task configurations.
    replace_task_sections(cfg, generated)

    assert "task_old" not in cfg

    # Step 3: Run the actual EMX2 constraints mechanism.
    constraints = apply_dynamic_constraints_to_config(
        cfg, registry
    )

    assert isinstance(constraints, dict)

    # Preserve the effective task configuration for comparison.
    before_guidance = {
        name: deepcopy(section)
        for name, section in cfg.items()
        if name.startswith("task_")
    }

    # Step 4: Append the shared narrative guidance.
    updated = append_narrative_guidance(cfg)

    assert updated == len(before_guidance)

    for name, original in before_guidance.items():
        current = cfg[name]

        # Every generated task retains the narrative guidance.
        assert NARRATIVE_GUIDANCE_MARKER in (
            current["instructions"]
        )

        assert current["instructions"].count(
            NARRATIVE_GUIDANCE_MARKER
        ) == 1

        # All previous instructions remain in their original order.
        assert current["instructions"].startswith(
            original["instructions"]
        )

        # Templates and other task metadata are unchanged.
        for key, value in original.items():
            if key != "instructions":
                assert current[key] == value

    # Step 5: Applying the policy again must change nothing.
    assert append_narrative_guidance(cfg) == 0

    # Non-task configuration must also remain unchanged.
    assert cfg["llm"] == {"model": "test-model"}


def test_narrative_guidance_reaches_actual_llm_messages():
    """Verify prompt delivery without contacting an actual LLM."""
    from extract_pipeline import extract_fields

    class PromptCaptured(Exception):
        pass

    class RecordingClient:
        def __init__(self):
            self.messages = None

        def chat(self, messages, **kwargs):
            self.messages = messages
            raise PromptCaptured()

    cfg = {
        "task_overview": {
            "instructions": (
                "Extract the source-supported cohort description."
            ),
            "template_json": '{"description": null}',
        },
    }

    assert append_narrative_guidance(cfg) == 1
    task = cfg["task_overview"]

    modes = {
        "standard": None,
        "context_prefix": [
            {
                "role": "system",
                "content": "Previously supplied paper context.",
            }
        ],
    }

    for mode, prefix in modes.items():
        client = RecordingClient()

        try:
            extract_fields(
                client,
                "Patients were enrolled in a prospective cohort.",
                template_json=task["template_json"],
                instructions=task["instructions"],
                use_grammar=False,
                temperature=0.0,
                max_tokens=128,
                prefix_messages=prefix,
                chunking_enabled=False,
            )
        except PromptCaptured:
            pass

        assert client.messages is not None, mode

        final_prompt = client.messages[-1]["content"]

        assert "# Template:" in final_prompt
        assert "# Instructions:" in final_prompt
        assert NARRATIVE_GUIDANCE_MARKER in final_prompt
        assert (
            "Preserve scientifically meaningful capitalization"
            in final_prompt
        )

        assert task["template_json"] in final_prompt



def test_final_workbook_audit_is_saved_without_changing_excel(tmp_path):
    """Verify the final workbook and audit-report contract."""
    import hashlib
    import json

    from main_cohort import write_json
    from cohort_narrative_policy import audit_final_narrative_workbook

    workbook = tmp_path / "cohorts.xlsx"
    report_path = tmp_path / "cohorts.xlsx.narrative_audit.json"

    registry = {
        "profile": "UMCGCohortsStaging",
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

    with pd.ExcelWriter(workbook) as writer:
        pd.DataFrame({
            "description": [
                "patients were enrolled",
                "eGFR was calculated",
            ],
            "name": ["study-one", "study-two"],
        }).to_excel(
            writer, sheet_name="Collections", index=False
        )

        pd.DataFrame({
            "title": ["lowercase publication title"],
        }).to_excel(
            writer, sheet_name="Publications", index=False
        )

    original_hash = hashlib.sha256(
        workbook.read_bytes()
    ).hexdigest()

    report = audit_final_narrative_workbook(
        workbook, registry
    )
    write_json(report_path, report)

    saved = json.loads(
        report_path.read_text(encoding="utf-8")
    )

    assert saved["status"] == "completed"
    assert saved["finding_count"] == 2
    assert all(
        finding["column"] == "description"
        for finding in saved["findings"]
    )

    # eGFR is a review candidate, not a confirmed error.
    assert any(
        finding["original"] == "eGFR was calculated"
        for finding in saved["findings"]
    )

    assert hashlib.sha256(
        workbook.read_bytes()
    ).hexdigest() == original_hash

    assert saved["workbook_sha256"] == original_hash

    skipped = audit_final_narrative_workbook(
        workbook, None
    )
    assert skipped["status"] == "skipped_missing_registry"


def test_final_audit_reports_missing_coverage(tmp_path):
    """An unaudited workbook must not be reported as validated."""
    import pytest

    from cohort_narrative_policy import audit_final_narrative_workbook

    workbook = tmp_path / "publications_only.xlsx"

    pd.DataFrame({
        "title": ["lowercase publication title"],
    }).to_excel(
        workbook, sheet_name="Publications", index=False
    )

    registry = {
        "profile": "UMCGCohortsStaging",
        "tables": {
            "Publications": {
                "fields": {
                    "title": {"column_type": "text"},
                },
            },
        },
    }

    result = audit_final_narrative_workbook(
        workbook, registry
    )

    assert result["status"] == "no_eligible_narrative_columns"
    assert result["finding_count"] == 0
    assert result["audited_columns"] == {}

    with pytest.raises(FileNotFoundError):
        audit_final_narrative_workbook(
            tmp_path / "missing.xlsx",
            registry,
        )


def test_final_audit_distinguishes_coverage_states(tmp_path):
    """Check partial, empty, and complete audit coverage."""
    from cohort_narrative_policy import audit_final_narrative_workbook

    registry = {
        "profile": "UMCGCohortsStaging",
        "tables": {
            "Resources": {
                "fields": {
                    "description": {"column_type": "text"},
                },
            },
            "Collections": {
                "extends": "Resources",
                "fields": {
                    "description": {"column_type": "text"},
                    "other exclusion criteria": {
                        "column_type": "text"
                    },
                },
            },
            "Subpopulations": {
                "fields": {
                    "description": {"column_type": "text"},
                },
            },
        },
    }

    # Case 1: An existing worksheet has a missing narrative column.
    partial_path = tmp_path / "partial.xlsx"
    pd.DataFrame({
        "description": ["patients were recruited"],
    }).to_excel(
        partial_path, sheet_name="Collections", index=False
    )

    partial = audit_final_narrative_workbook(
        partial_path, registry
    )

    assert partial["status"] == "partial_coverage"
    assert partial["missing_columns"]["Collections"] == [
        "other exclusion criteria"
    ]
    assert "Resources" in partial["absent_eligible_sheets"]
    assert partial["missing_required_sheets"] == []
    assert partial["inspected_value_count"] == 1

    # Case 2: All eligible columns exist, but contain no values.
    empty_path = tmp_path / "empty.xlsx"
    pd.DataFrame({
        "description": [""],
        "other exclusion criteria": [""],
    }).to_excel(
        empty_path, sheet_name="Collections", index=False
    )

    empty = audit_final_narrative_workbook(
        empty_path, registry
    )

    assert empty["status"] == "no_populated_narrative_values"
    assert empty["inspected_value_count"] == 0

    # Case 3: Populated Collections; optional Resources is absent.
    complete_path = tmp_path / "complete.xlsx"
    pd.DataFrame({
        "description": ["Patients were recruited."],
        "other exclusion criteria": ["No additional exclusions."],
    }).to_excel(
        complete_path, sheet_name="Collections", index=False
    )

    complete = audit_final_narrative_workbook(
        complete_path, registry
    )

    assert complete["status"] == "completed"
    assert complete["missing_columns"] == {}
    assert complete["missing_required_sheets"] == []
    assert complete["inspected_value_count"] == 2
    assert "Resources" in complete["absent_eligible_sheets"]

    # Case 4: Collections is missing even though other data exist.
    missing_core_path = tmp_path / "missing_core.xlsx"
    pd.DataFrame({
        "description": ["Patients with cancer"],
    }).to_excel(
        missing_core_path,
        sheet_name="Subpopulations",
        index=False,
    )

    missing_core = audit_final_narrative_workbook(
        missing_core_path, registry
    )

    assert missing_core["status"] == "partial_coverage"
    assert missing_core["missing_required_sheets"] == [
        "Collections"
    ]