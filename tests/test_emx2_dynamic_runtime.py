from __future__ import annotations

from emx2_dynamic_runtime import _load_csv_rows


def test_load_csv_rows_handles_utf8_bom(tmp_path):
    csv_path = tmp_path / "Agents.csv"
    csv_path.write_text(
        "tableName,tableExtends,columnName,profiles\n"
        "Agents,,id,UMCGCohortsStaging\n",
        encoding="utf-8-sig",
    )

    rows = _load_csv_rows(csv_path)

    assert len(rows) == 1
    assert rows[0]["tableName"] == "Agents"
    assert rows[0]["columnName"] == "id"
