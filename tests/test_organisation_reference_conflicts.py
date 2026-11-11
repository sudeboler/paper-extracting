from openpyxl import Workbook

from fix_molgenis_staging_types_callable import (
    _normalize_external_organisation_refs,
)


def workbook_with_row(primary, other, department):
    wb = Workbook()
    ws = wb.active
    ws.title = "Organisations"
    ws.append([
        "id", "type", "organisation",
        "other organisation", "department",
    ])
    ws.append([
        "unit-1", "Organisation", primary, other, department,
    ])
    return wb, ws


def test_conflicting_names_are_reported_and_free_text_preserved():
    wb, ws = workbook_with_row(
        "Example University",
        "Another Institution",
        "Department of Genetics",
    )
    index = {
        "exact": {"Example University": "Example University"},
        "norm": {"exampleuniversity": "Example University"},
    }

    findings = _normalize_external_organisation_refs(wb, index)

    assert ws["C2"].value == "Example University"
    assert ws["D2"].value == "Another Institution"
    assert ws["E2"].value == "Department of Genetics"
    assert len(findings) == 1
    assert findings[0]["requires_review"] is True
    assert findings[0]["original_other_organisation"] == "Another Institution"


def test_unverified_primary_is_not_exported_as_external_reference():
    wb, ws = workbook_with_row(
        "Unverified University",
        "Another Institution",
        "Department of Epidemiology",
    )

    findings = _normalize_external_organisation_refs(wb, {})

    assert ws["C2"].value == ""
    assert ws["D2"].value == "Another Institution"
    assert ws["E2"].value == "Department of Epidemiology"
    assert findings[0]["original_organisation"] == "Unverified University"


def test_equivalent_aliases_resolve_without_conflict():
    wb, ws = workbook_with_row(
        "EU",
        "Example University",
        "Department of Genetics",
    )
    index = {
        "exact": {
            "EU": "Example University",
            "Example University": "Example University",
        },
        "norm": {
            "eu": "Example University",
            "exampleuniversity": "Example University",
        },
    }

    findings = _normalize_external_organisation_refs(wb, index)

    assert ws["C2"].value == "Example University"
    assert ws["D2"].value == ""
    assert ws["E2"].value == "Department of Genetics"
    assert findings == []


def test_single_unverified_label_uses_fallback():
    wb, ws = workbook_with_row(
        "Unverified University",
        "",
        "Department of Genetics",
    )

    findings = _normalize_external_organisation_refs(wb, {})

    assert ws["C2"].value == ""
    assert ws["D2"].value == "Unverified University"
    assert ws["E2"].value == "Department of Genetics"
    assert findings == []

def test_organisation_review_report_is_persisted(tmp_path):
    import json
    from fix_molgenis_staging_types_callable import (
        _write_organisation_review_report,
    )

    wb, ws = workbook_with_row(
        "Unverified University",
        "Another Institution",
        "Department of Epidemiology",
    )
    findings = _normalize_external_organisation_refs(wb, {})

    workbook_path = tmp_path / "example.xlsx"
    wb.save(workbook_path)

    report_path = _write_organisation_review_report(
        workbook_path, findings
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report_path.is_file()
    assert report["status"] == "review_required"
    assert report["finding_count"] == 1
    assert (
        report["findings"][0]["original_organisation"]
        == "Unverified University"
    )
    assert ws["E2"].value == "Department of Epidemiology"

    empty_path = _write_organisation_review_report(
        tmp_path / "empty.xlsx", []
    )
    empty_report = json.loads(
        empty_path.read_text(encoding="utf-8")
    )

    assert empty_report["status"] == "no_conflicts_detected"
    assert empty_report["finding_count"] == 0
