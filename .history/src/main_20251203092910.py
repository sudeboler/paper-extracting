from __future__ import annotations
import os
import logging
import json
import argparse
from typing import Any, Dict

# --- NIEUW: Import voor Excel ---
import pandas as pd

try:
    import tomllib as toml
except ModuleNotFoundError:
    import tomli as toml

from .llm_client import OpenAICompatibleClient
from .extract_pipeline import load_pdf_text, extract_fields, _merge_json_results

# ==========================================
# BEPAAL HIER JE VOLGORDE VOOR EXCEL & PRINT
# ==========================================
DESIRED_ORDER = [
    # --- Pass A: Main ---
    "pid",
    "study_name",
    "study_acronym",
    "study_types",
    "cohort_type",
    "website",
    "start_year",
    "end_year",
    "contact_email",
    "n_included",
    "countries",
    "regions",
    "population_age_group",

    # --- Pass B: Criteria ---
    "inclusion_criteria",
    "other_inclusion_criteria",
    "exclusion_criteria",
    "other_exclusion_criteria",
    "clinical_study_types",

    # --- Pass C: Design ---
    "design",
    "design_description",
    "data_collection_type",
    "data_collection_description",
    "description",

    # --- Pass D: Population ---
    "number_of_participants_with_samples",
    "underlying_population",
    "population_of_interest",
    "population_of_interest_other",
    "part_of_networks",
    "population_entry",
    "population_entry_other",
    "population_exit",
    "population_exit_other",
    "population_disease",
    "population_oncology_topology",
    "population_oncology_morphology",
    "population_coverage",
    "population_not_covered",
    "age_min",
    "age_max",

    # --- Pass E: Access ---
    "informed_consent_type",
    "informed_consent_required",
    "informed_consent_other",
    "access_rights",
    "data_access_conditions",
    "data_use_conditions",
    "data_access_conditions_description",
    "data_access_fee",
    "access_identifiable_data",
    "access_identifiable_data_route",
    "access_subject_details",
    "access_subject_details_route",
    "access_third_party",
    "access_third_party_conditions",
    "access_non_eu",
    "access_non_eu_conditions",

    # --- Pass F: Contributors ---
    "counts_resource",
    "counts_age_group",
    "organisations_involved_resource",
    "organisations_involved_id",
    "publisher_resource",
    "publisher_id",
    "creator_resource",
    "creator_id",
    "people_involved_resource",
    "contact_point_resource",
    "contact_point_first_name",
    "contact_point_last_name",
    "child_networks",
    "parent_networks",

    # --- Pass G: Data Model ---
    "datasets_resource",
    "datasets_name",
    "samplesets_resource",
    "samplesets_name",
    "areas_of_information",
    "areas_of_information_rwd",
    "quality_of_life_measures",
    "cause_of_death_vocabulary",
    "indication_vocabulary",
    "genetic_data_vocabulary",
    "care_setting_description",
    "medicinal_product_vocabulary",
    "prescriptions_vocabulary",
    "dispensings_vocabulary",
    "procedures_vocabulary",
    "biomarker_data_vocabulary",
    "diagnosis_medical_event_vocabulary",
    "data_dictionary_available",
    "disease_details",

    # --- Pass H: Biobank Updates ---
    "biospecimen_access",
    "biospecimen_access_conditions",
    "governance_details",
    "approval_for_publication",
    "release_type",
    "release_description",
    "number_of_records",
    "release_frequency_months",
    "refresh_time_days",
    "lag_time_days",
    "refresh_period",
    "date_last_refresh",
    "preservation_indefinite",
    "preservation_duration_years",

    # --- Pass I: Quality ---
    "standard_operating_procedures",
    "qualification",
    "qualifications_description",
    "audit_possible",
    "completeness",
    "completeness_over_time",
    "completeness_results",
    "quality_description",
    "quality_over_time",
    "access_for_validation",
    "quality_validation_frequency",
    "quality_validation_methods",
    "correction_methods",
    "quality_validation_results",
    "quality_marks",

    # --- Pass J: Linkage ---
    "biospecimen_collected",
    "languages",
    "multiple_entries",
    "has_identifier",
    "identifier_description",
    "prelinked",
    "linkage_options",
    "linkage_possibility",
    "linked_resources_names",

    # --- Pass K: Triggers ---
    "reason_sustained",
    "record_trigger",
    "unit_of_observation",
    "subpopulations_resource",
    "subpopulations_name",
    "collection_events_resource",
    "collection_events_name",
    "data_resources_included",

    # --- Pass L: CDM & Funding ---
    "cdm_mapping_source",
    "cdm_mapping_source_dataset",
    "cdm_mapping_target",
    "cdm_mapping_target_dataset",
    "cdm_other",
    "etl_vocabularies",
    "etl_vocabularies_other",
    "publications",
    "funding_sources",
    "funding_scheme",
    "funding_statement",

    # --- Pass M: Docs & Dates ---
    "citation_requirements",
    "acknowledgements",
    "provenance_statement",
    "documentation",
    "supplementary_information",
    "theme",
    "applicable_legislation",
    "collection_start_planned",
    "collection_start_actual",
    "analysis_start_planned",
    "analysis_start_actual",

    # --- Pass N: Content ---
    "data_sources",
    "medical_conditions_studied",
    "data_extraction_date",
    "analysis_plan",
    "objectives",
    "results"
]


def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
    )


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return toml.load(f)


def reorder_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Sorteert de dictionary op basis van DESIRED_ORDER."""
    ordered_data = {}
    
    # Eerst de velden die in onze voorkeurslijst staan
    for key in DESIRED_ORDER:
        if key in data:
            ordered_data[key] = data[key]
    
    # Voeg eventuele overgebleven velden toe (voor de zekerheid)
    for key, value in data.items():
        if key not in ordered_data:
            ordered_data[key] = value
            
    return ordered_data


def _print_pretty_json(data: Any) -> None:
    # Sorteer eerst voordat we printen
    ordered = reorder_data(data)
    text = json.dumps(ordered, indent=2, ensure_ascii=False)
    print(text)


def save_to_excel(data: Dict[str, Any], filename: str):
    """Slaat data op naar Excel in de juiste kolomvolgorde."""
    # Sorteer de data
    ordered_data = reorder_data(data)
    
    # Maak DataFrame (lijsten/dicts worden strings zodat ze in 1 cel passen)
    clean_data = {}
    for k, v in ordered_data.items():
        if isinstance(v, (list, dict)):
            clean_data[k] = json.dumps(v, ensure_ascii=False)
        else:
            clean_data[k] = v
            
    df = pd.DataFrame([clean_data])
    
    try:
        df.to_excel(filename, index=False)
        logging.getLogger("main").info(f"Excel succesvol opgeslagen als: {filename}")
    except Exception as e:
        logging.getLogger("main").error(f"Fout bij opslaan Excel: {e}")


def cli():
    parser = argparse.ArgumentParser(description="Run PDF extraction passes.")
    parser.add_argument(
        "-p", "--passes", 
        nargs="+", 
        default=["all"],
        help="Specify passes: A, B, C, D, E, F, G, H, I, J, K, L, M, N. Default: 'all'"
    )
    parser.add_argument(
        "-o", "--output",
        default="extraction_results.xlsx",
        help="Output Excel filename"
    )
    args = parser.parse_args()
    selected_passes = [p.upper() for p in args.passes]
    
    cfg_path = os.environ.get("PDF_EXTRACT_CONFIG", "config.toml")
    cfg = load_config(cfg_path)

    setup_logging(cfg.get("logging", {}).get("level", "INFO"))
    log = logging.getLogger("main")

    llm_cfg = cfg["llm"]
    pdf_cfg = cfg["pdf"]
    
    client = OpenAICompatibleClient(
        base_url=llm_cfg.get("base_url", "http://127.0.0.1:8080/v1"),
        api_key=llm_cfg.get("api_key", "sk-local"),
        model=llm_cfg.get("model", "numind/NuExtract-2.0-8B"),
        use_grammar=bool(llm_cfg.get("use_grammar", False)),
    )

    paper_text = load_pdf_text(pdf_cfg["path"], max_pages=pdf_cfg.get("max_pages"))
    log.info("PDF loaded (%d chars)", len(paper_text))

    def run_pass(name, cfg_section_key):
        task_cfg = cfg.get(cfg_section_key, {})
        if not task_cfg:
            log.warning(f"Config section '{cfg_section_key}' is empty or missing!")
            return {}
            
        log.info(f"--- Running {name} ---")
        return extract_fields(
            client, paper_text,
            template_json=task_cfg.get("template_json"),
            instructions=task_cfg.get("instructions"),
            use_grammar=bool(llm_cfg.get("use_grammar", False)),
            temperature=float(llm_cfg.get("temperature", 0.0)),
            max_tokens=int(llm_cfg.get("max_tokens", 2048)),
        )

    # De complete lijst met taken (Pass L toegevoegd)
    all_tasks = [
        ("A", "Pass A (Main)", "task_main"),
        ("B", "Pass B (Criteria)", "task_criteria"),
        ("C", "Pass C (Design)", "task_design_details"),
        ("D", "Pass D (Population)", "task_population"),
        ("E", "Pass E (Access)", "task_access"),
        ("F", "Pass F (Contributors)", "task_contributors"),
        ("G", "Pass G (Data Model)", "task_datamodel"),
        ("H", "Pass H (Biobank Updates)", "task_biobank_updates"),
        ("I", "Pass I (Quality & SOPs)", "task_quality"),
        ("J", "Pass J (Linkage & Specs)", "task_linkage_specs"),
        ("K", "Pass K (Triggers & Structure)", "task_triggers_subpops"),
        ("L", "Pass L (CDM & Funding)", "task_cdm_funding"), 
        ("M", "Pass M (Docs & Dates)", "task_docs_legislation_dates"),
        ("N", "Pass N (Content)", "task_study_content"),
    ]

    merged_results = {}

    for code, name, section in all_tasks:
        if "ALL" in selected_passes or code in selected_passes:
            res = run_pass(f"{name} [{code}]", section)
            merged_results = _merge_json_results(merged_results, res)
        else:
            log.info(f"Skipping {name} (not selected)")

    log.info("--- DONE EXTRACTING ---")
    
    # 1. Print JSON in de juiste volgorde
    print("\n=== JSON RESULT ===")
    _print_pretty_json(merged_results)
    
    # 2. Save to Excel
    save_to_excel(merged_results, args.output)


if __name__ == "__main__":
    cli()