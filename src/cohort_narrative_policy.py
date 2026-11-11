"""Source-preserving narrative presentation guidance for cohort extraction."""

from typing import Any
import re

import hashlib
from pathlib import Path

import pandas as pd

NARRATIVE_GUIDANCE_MARKER = "NARRATIVE PRESENTATION RULES"

NARRATIVE_GUIDANCE = """
NARRATIVE PRESENTATION RULES

- For natural-language descriptions and free-text inclusion or exclusion
  criteria, use conventional capitalization at the beginning of sentences.
- Preserve the original scientific meaning and evidence from the paper.
- Preserve scientifically meaningful capitalization in gene symbols,
  proteins, measurements, abbreviations, study names, and technical terms.
- Do not rewrite or normalize scientific terminology merely to change
  capitalization.
- Preserve exact identifiers, references, DOIs, URLs, ontology labels,
  controlled vocabulary terms, and publication titles.
- Do not invent information or expand source fragments into unsupported
  statements.
- When correct capitalization is uncertain, preserve the source wording
  rather than guessing.
""".strip()


def append_narrative_guidance(cfg: dict[str, Any]) -> int:
    """Append guidance to existing extraction tasks without replacing them.

    Returns the number of task sections updated.
    """
    updated = 0

    for name, section in cfg.items():
        if not str(name).startswith("task_"):
            continue

        if not isinstance(section, dict):
            continue

        if "instructions" not in section:
            continue

        instructions = str(section.get("instructions") or "").rstrip()

        if NARRATIVE_GUIDANCE_MARKER in instructions:
            continue

        section["instructions"] = (
            f"{instructions}\n\n{NARRATIVE_GUIDANCE}".strip()
        )
        updated += 1

    return updated


# Explicitly reviewed narrative fields.
# These candidates are only eligible when the effective EMX2
# registry also declares them as text fields.

_RESOURCE_NARRATIVE_FIELDS = frozenset({
    "description",
    "design description",
    "data access conditions description",
    "release description",
    "funding statement",
    "acknowledgements",
    "provenance statement",
    "other inclusion criteria",
    "other exclusion criteria",
})

NARRATIVE_FIELDS_BY_TABLE = {
    "Resources": _RESOURCE_NARRATIVE_FIELDS,
    "Collections": _RESOURCE_NARRATIVE_FIELDS,
    "Subpopulations": frozenset({
        "description",
        "other inclusion criteria",
        "other exclusion criteria",
    }),
    "Collection events": frozenset({"description"}),
    "Documentation": frozenset({"description"}),
}


def eligible_narrative_fields(
    registry: dict[str, Any] | None,
) -> dict[str, frozenset[str]]:
    """Select reviewed narrative text fields from the EMX2 registry.

    No metadata values are changed by this function.
    """
    if not isinstance(registry, dict):
        return {}

    tables = registry.get("tables") or {}
    if not isinstance(tables, dict):
        return {}

    eligible = {}

    for table_name, candidate_fields in NARRATIVE_FIELDS_BY_TABLE.items():
        table_meta = tables.get(table_name) or {}
        fields = table_meta.get("fields") or {}

        if not isinstance(fields, dict):
            continue

        selected = frozenset(
            column_name
            for column_name in candidate_fields
            if isinstance(fields.get(column_name), dict)
            and str(
                fields[column_name].get("column_type") or ""
            ).strip() == "text"
        )

        if selected:
            eligible[table_name] = selected

    return eligible


def audit_narrative_capitalization(
    registry: dict[str, Any] | None,
    frames: dict[str, Any],
) -> list[dict[str, Any]]:
    """Identify capitalization candidates without changing metadata."""

    findings = []
    eligible = eligible_narrative_fields(registry)

    for sheet_name, allowed_columns in eligible.items():
        frame = frames.get(sheet_name)

        if frame is None or frame.empty:
            continue

        columns = sorted(allowed_columns.intersection(frame.columns))

        for column in columns:
            for excel_row, value in enumerate(frame[column], start=2):
                if not isinstance(value, str) or not value.strip():
                    continue

                # Check the beginning of the cell.
                start = len(value) - len(value.lstrip())

                if "a" <= value[start] <= "z":
                    findings.append({
                        "sheet": sheet_name,
                        "excel_row": excel_row,
                        "column": column,
                        "reason": "cell_start",
                        "position": start,
                        "original": value,
                    })

                # Check possible sentences within the same cell.
                # Periods after abbreviations may NOT be sentence boundaries.
                # Therefore, these are review candidates, not corrections.
                for match in re.finditer(
                    r"([.!?])\s+([a-z])", value
                ):
                    findings.append({
                        "sheet": sheet_name,
                        "excel_row": excel_row,
                        "column": column,
                        "reason": "possible_sentence_start",
                        "position": match.start(2),
                        "original": value,
                    })

    return findings


def audit_final_narrative_workbook(
    workbook_path: str | Path,
    registry: dict[str, Any] | None,
) -> dict[str, Any]:
    """Audit the final workbook without modifying its contents."""

    path = Path(workbook_path)

    if not path.is_file():
        raise FileNotFoundError(path)

    checksum_before = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()

    if not isinstance(registry, dict):
        return {
            "status": "skipped_missing_registry",
            "workbook_sha256": checksum_before,
            "findings": [],
            "message": "No effective EMX2 registry was available.",
        }

    eligible = eligible_narrative_fields(registry)

    with pd.ExcelFile(path) as workbook:
        available_sheets = set(workbook.sheet_names)

        frames = {
            sheet: pd.read_excel(
                workbook,
                sheet_name=sheet,
                dtype=str,
                keep_default_na=False,
            )
            for sheet in eligible
            if sheet in available_sheets
        }

    audited_columns = {
        sheet: sorted(fields.intersection(frames[sheet].columns))
        for sheet, fields in eligible.items()
        if sheet in frames
    }

        # Report missing eligible columns on worksheets that exist.
    # An absent Resources worksheet may be legitimate.
    missing_columns = {
        sheet: sorted(fields.difference(frames[sheet].columns))
        for sheet, fields in eligible.items()
        if sheet in frames
        and fields.difference(frames[sheet].columns)
    }

    # Record every eligible worksheet that is absent.
    # Absence does not necessarily mean that the worksheet is optional.
    absent_eligible_sheets = sorted(
        set(eligible).difference(frames)
    )

    # Collections is the core cohort worksheet in this pipeline.
    # Resources may legitimately be omitted when empty.
    missing_required_sheets = [
        sheet
        for sheet in ("Collections",)
        if sheet in eligible and sheet not in frames
    ]

    inspected_value_count = sum(
        1
        for sheet, columns in audited_columns.items()
        for column in columns
        for value in frames[sheet][column]
        if isinstance(value, str) and value.strip()
    )

    findings = audit_narrative_capitalization(
        registry, frames
    )

    checksum_after = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()

    if checksum_before != checksum_after:
        raise RuntimeError(
            "Workbook contents changed during narrative auditing."
        )

    if missing_required_sheets or missing_columns:
        status = "partial_coverage"
    elif not any(audited_columns.values()):
        status = "no_eligible_narrative_columns"
    elif inspected_value_count == 0:
        status = "no_populated_narrative_values"
    else:
        status = "completed"

    return {
                "status": status,
        "registry_profile": registry.get("profile"),
        "workbook_sha256": checksum_before,
        "audited_columns": audited_columns,
        "missing_columns": missing_columns,
        "absent_eligible_sheets": absent_eligible_sheets,
        "missing_required_sheets": missing_required_sheets,
        "inspected_value_count": inspected_value_count,
        "finding_count": len(findings),
        "findings": findings,
        "interpretation": (
            "Findings are review candidates, not confirmed errors. "
            "No capitalization changes were applied."
        ),
    }