from __future__ import annotations

from openpyxl import Workbook

from fix_molgenis_staging_types_callable import (
    _build_ref_index,
    _normalize_external_organisation_refs,
    coerce_ref,
)
from main_cohort import (
    _resource_ref,
    _build_local_organisation_ref_map,
    _project_resource_as_collection,
    _clear_organisation_parent_reference,
    _clear_self_organisation_reference,
    _ensure_organisation_ids,
    _registry_import_columns,
    _normalize_or_infer_resource_types,
    _resolve_local_organisation_ref,
    _resource_row_from_sections,
    _serialize_multi_value,
    _serialize_runtime_array_value,
)


def test_serialize_multi_value_flattens_and_cleans_bracketed_tokens() -> None:
    assert _serialize_multi_value('["Data access provider"]') == "Data access provider"
    assert _serialize_multi_value('Data access provider"]') == "Data access provider"


def test_runtime_ref_array_serializer_normalizes_and_filters_to_allowed_values() -> None:
    meta = {
        "column_type": "ref_array",
        "allowed_values": ["Data access provider", "Data controller"],
    }
    value = '["data access provider", "unknown role", "Data access provider\"]"]'
    assert _serialize_runtime_array_value("Agents", "role", meta, value) == "Data access provider"


def test_generated_organisation_ids_enable_local_reference_resolution() -> None:
    issues: list[dict[str, str]] = []
    organisations = [
        {
            "type": "organisation",
            "name": "UK Medical Research Council",
            "id": "",
        }
    ]
    _ensure_organisation_ids(
        organisations,
        paper_label="alspac",
        pdf_path="data/alspac.pdf",
        issues=issues,
    )
    assert organisations[0]["id"]
    assert issues

    ref_map = _build_local_organisation_ref_map(organisations)
    resolved = _resolve_local_organisation_ref("UK Medical Research Council", ref_map)
    assert resolved == organisations[0]["id"]


def test_self_organisation_reference_is_cleared_to_avoid_fk_error() -> None:
    row = {"id": "MRC", "organisation": "MRC"}
    _clear_self_organisation_reference(row)
    assert row["organisation"] == ""


def test_unverified_external_organisation_uses_free_text_fallback() -> None:
    from fix_molgenis_staging_types_callable import (
        _normalize_external_organisation_refs,
    )

    wb = Workbook()
    ws = wb.active
    ws.title = "Organisations"
    ws.append([
        "id", "type", "organisation",
        "other organisation", "department",
    ])
    ws.append([
        "epidemiology",
        "Organisation",
        "Example University",
        "",
        "Department of Epidemiology",
    ])

    _normalize_external_organisation_refs(wb, {})

    assert ws["C2"].value == ""
    assert ws["D2"].value == "Example University"
    assert ws["E2"].value == "Department of Epidemiology"


def test_external_organisation_normalizer_does_not_backfill_blank_fk_from_name() -> None:
    wb = Workbook()
    ws = wb.active
    ws.title = "Organisations"
    ws.append(
        [
            "resource",
            "id",
            "type",
            "name",
            "organisation",
            "other organisation",
            "department",
            "website",
            "email",
            "logo",
            "role",
            "is lead organisation",
        ]
    )
    ws.append(
        [
            "10.1186/s12967-019-2122-x",
            "umcg-local",
            "Organisation",
            "UMCG",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
        ]
    )

    _normalize_external_organisation_refs(
        wb,
        {
            "exact": {"UMCG": "UMCG"},
            "norm": {"umcg": "UMCG"},
        },
    )

    assert ws.cell(row=2, column=5).value == ""


def test_local_organisation_index_excludes_external_affiliation_alias() -> None:
    wb = Workbook()
    ws = wb.active
    ws.title = "Organisations"
    ws.append([
        "resource", "id", "type", "name",
        "organisation", "other organisation", "department",
    ])
    ws.append([
        "example-resource",
        "umcg-local",
        "Organisation",
        "UMCG",
        "",
        "University Medical Center Groningen",
        "",
    ])

    ref_index = _build_ref_index(wb)

    assert coerce_ref(
        "Resources", "publisher", "UMCG", ref_index
    ) == "umcg-local"

    # The external free-text institution label is not, by itself,
    # proof of an equivalent local contributor identity.
    assert coerce_ref(
        "Resources",
        "publisher",
        "University Medical Center Groningen",
        ref_index,
    ) == ""


def test_resource_ref_prefers_acronym_over_name_and_pid() -> None:
    overview = {
        "pid": "10.1234/example-publication",
        "name": "Example Longitudinal Health Cohort",
        "acronym": "ELHC",
    }

    assert _resource_ref("example-paper", overview) == "ELHC"


def test_resource_ref_uses_name_when_acronym_missing() -> None:
    overview = {
        "pid": None,
        "name": "Example Longitudinal Health Cohort",
        "acronym": None,
    }

    assert _resource_ref("example-paper", overview) == "Example Longitudinal Health Cohort"


def test_resource_ref_uses_input_label_when_name_and_acronym_missing() -> None:
    overview = {
        "pid": None,
        "name": None,
        "acronym": None,
    }

    assert _resource_ref("example-paper", overview) == "example-paper"


def test_resource_row_contract_has_non_empty_type() -> None:
    row, resource_ref = _resource_row_from_sections(
        "oncolifes",
        "data/oncolifes.pdf",
        {
            "task_overview": {
                "name": "OncoLifeS",
                "pid": "10.1186/s12967-019-2122-x",
                "type": [],
                "keywords": ["biobank"],
            },
            "task_design_structure": {},
            "task_population": {},
            "task_contributors": {},
            "task_areas_of_information": {},
            "task_linkage": {},
            "task_access_conditions": {},
            "task_information": {},
            "task_subpopulations": {"subpopulations": []},
            "task_collection_events": {"collection_events": []},
        },
        [],
    )

    assert resource_ref == "OncoLifeS"
    assert row["type"]
    assert _normalize_or_infer_resource_types({"name": "OncoLifeS biobank"})


def test_collections_import_columns_keep_inherited_id() -> None:
    registry = {
        "tables": {
            "Collections": {
                "extends": "Resources",
                "fields": {
                    "id": {"column_type": "string"},
                    "type": {"column_type": "ontology_array"},
                    "keywords": {"column_type": "string_array"},
                },
                "direct_field_names": ["type", "keywords"],
            }
        }
    }

    assert _registry_import_columns(registry, "Collections") == ["id", "type", "keywords"]



def test_collection_export_does_not_copy_extracted_pid() -> None:
    sections = {
        "task_overview": {
            "name": "Example Longitudinal Health Cohort",
            "acronym": "ELHC",
            "pid": "10.1234/example-publication",
        }
    }

    resource_row, resource_ref = _resource_row_from_sections(
        "example-paper",
        "data/example-paper.pdf",
        sections,
        [],
    )

    # The runtime registry contains resolved inherited fields on Collections.
    registry = {
        "tables": {
            "Resources": {
                "fields": {
                    "id": {},
                    "pid": {},
                    "name": {},
                    "acronym": {},
                }
            },
            "Collections": {
                "extends": "Resources",
                "fields": {
                    "id": {},
                    "pid": {},
                    "name": {},
                    "acronym": {},
                },
            },
        }
    }

    columns = _registry_import_columns(registry, "Collections")
    collection_row = _project_resource_as_collection(resource_row, columns)

    assert resource_ref == "ELHC"
    assert collection_row["id"] == "ELHC"
    assert collection_row["name"] == "Example Longitudinal Health Cohort"
    assert collection_row["acronym"] == "ELHC"
    assert collection_row["pid"] == ""

    # Collection projection must not modify the original extracted resource.
    assert resource_row["pid"] == "10.1234/example-publication"




def test_verified_external_organisation_reference_is_preserved() -> None:
    from fix_molgenis_staging_types_callable import (
        _normalize_external_organisation_refs,
    )

    wb = Workbook()
    ws = wb.active
    ws.title = "Organisations"
    ws.append([
        "id", "type", "organisation",
        "other organisation", "department",
    ])
    ws.append([
        "genetics",
        "Organisation",
        "Example University",
        "",
        "Department of Genetics",
    ])

    index = {
        "exact": {"Example University": "Example University"},
        "norm": {"exampleuniversity": "Example University"},
    }
    _normalize_external_organisation_refs(wb, index)

    assert ws["C2"].value == "Example University"
    assert ws["D2"].value == ""
    assert ws["E2"].value == "Department of Genetics"


def test_serialize_multi_value_preserves_commas_inside_terms() -> None:
    # Ordinary multi-value list.
    assert _serialize_multi_value(
        ["keyword1", "keyword2", "keyword3"]
    ) == "keyword1,keyword2,keyword3"

    # One quoted ontology term containing an internal comma.
    assert _serialize_multi_value(
        '"one keyword, with a comma in name"'
    ) == '"one keyword, with a comma in name"'

    # Mixed list with one comma-containing ontology term.
    assert _serialize_multi_value(
        ["keyword1, with a comma in name", "keyword2", "keyword3"]
    ) == '"keyword1, with a comma in name",keyword2,keyword3'

    # The same mixed list supplied as an already serialized CSV string.
    assert _serialize_multi_value(
        '"keyword1, with a comma in name",keyword2,keyword3'
    ) == '"keyword1, with a comma in name",keyword2,keyword3'








def test_runtime_scalar_ontology_uses_canonical_capitalization() -> None:
    from main_cohort import _serialize_runtime_scalar_value

    meta = {
        "column_type": "ontology",
        "allowed_values": ["Oncology"],
    }

    assert _serialize_runtime_scalar_value(
        "Collections", "example ontology field", meta, "oncology"
    ) == "Oncology"

    assert _serialize_runtime_scalar_value(
        "Collections", "example ontology field", meta, "Oncology"
    ) == "Oncology"

    # An unknown term must not be accepted.
    assert _serialize_runtime_scalar_value(
        "Collections", "example ontology field", meta, "Unknown category"
    ) == ""







def test_array_parser_preserves_plain_and_quoted_values() -> None:
    from fix_molgenis_staging_types_callable import _parse_array_items

    assert _parse_array_items("Biobank") == ["Biobank"]
    assert _parse_array_items("Data provider") == ["Data provider"]

    assert _parse_array_items(
        '"one keyword, with a comma in name"'
    ) == ["one keyword, with a comma in name"]

    assert _parse_array_items(
        "Data holder,Data provider"
    ) == ["Data holder", "Data provider"]

    assert _parse_array_items(
        '["Data holder", "Data provider"]'
    ) == ["Data holder", "Data provider"]


def test_shared_institution_does_not_resolve_to_arbitrary_department() -> None:
    wb = Workbook()
    ws = wb.active
    ws.title = "Organisations"
    ws.append([
        "resource", "id", "type", "name",
        "organisation", "other organisation", "department",
    ])
    ws.append([
        "study", "epidemiology", "Organisation", "",
        "", "Example University", "Department of Epidemiology",
    ])
    ws.append([
        "study", "genetics", "Organisation", "",
        "", "Example University", "Department of Genetics",
    ])

    index = _build_ref_index(wb)

    assert coerce_ref(
        "Contacts", "organisation", "Example University", index
    ) == ""

    assert coerce_ref(
        "Contacts", "organisation", "epidemiology", index
    ) == "epidemiology"

    assert coerce_ref(
        "Contacts", "organisation", "genetics", index
    ) == "genetics"

    # An explicitly represented parent institution is a valid
    # local target, unlike an inferred departmental alias.
    ws.append([
        "study", "university", "Organisation",
        "Example University", "", "", "",
    ])

    index = _build_ref_index(wb)

    assert coerce_ref(
        "Contacts", "organisation", "Example University", index
    ) == "university"
