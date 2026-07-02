from __future__ import annotations

import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REVIEW_MD_PATH = PROJECT_ROOT / "validation_layer" / "data_enrichment_review.md"
GAP_MATRIX_PATH = PROJECT_ROOT / "validation_layer" / "data_enrichment_gap_matrix.csv"
INVENTORY_PATH = PROJECT_ROOT / "validation_layer" / "data_enrichment_current_inventory.csv"
DECISION_JSON_PATH = PROJECT_ROOT / "decision_layer" / "data_enrichment_decision.json"


def fail(message: str) -> None:
    raise SystemExit(f"FAIL: {message}")


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        fail(f"missing output: {path.relative_to(PROJECT_ROOT)}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        fail(f"{path.relative_to(PROJECT_ROOT)} is empty")
    return rows


def main() -> None:
    if not REVIEW_MD_PATH.exists():
        fail("missing data_enrichment_review.md")
    review = REVIEW_MD_PATH.read_text(encoding="utf-8")
    for phrase in [
        "data gap review only",
        "Formal output: unchanged by this review.",
        "This does not add a data source yet.",
        "This does not train or promote a model.",
        "avoid leakage",
    ]:
        if phrase not in review:
            fail(f"data enrichment review missing phrase: {phrase}")

    if not DECISION_JSON_PATH.exists():
        fail("missing data_enrichment_decision.json")
    decision = json.loads(DECISION_JSON_PATH.read_text(encoding="utf-8"))
    if decision.get("status") != "data_gap_confirmed":
        fail("data enrichment decision must confirm data gap")
    if decision.get("recommended_next_step") != "build_event_risk_data_collector_spec":
        fail("unexpected data enrichment next step")
    if decision.get("recommended_data_need_id") != "event_risk_calendar":
        fail("event_risk_calendar must be the first recommended data need")
    if decision.get("do_not_retrain_yet") is not True:
        fail("decision must block retraining until enrichment scope is specified")
    if decision.get("formal_outputs_unchanged") is not True:
        fail("formal output must remain unchanged")

    gap_rows = read_rows(GAP_MATRIX_PATH)
    required_gap_cols = {
        "data_need_id",
        "data_family",
        "risk_question_answered",
        "current_coverage",
        "missing_fields",
        "no_leakage_rule",
        "review_score",
    }
    missing_gap_cols = required_gap_cols - set(gap_rows[0])
    if missing_gap_cols:
        fail("data_enrichment_gap_matrix.csv missing columns: " + ", ".join(sorted(missing_gap_cols)))
    gap_ids = {row["data_need_id"] for row in gap_rows}
    if "event_risk_calendar" not in gap_ids:
        fail("gap matrix must include event_risk_calendar")

    inventory_rows = read_rows(INVENTORY_PATH)
    required_inventory_cols = {
        "current_data_group",
        "source",
        "required_columns",
        "present_columns",
        "missing_columns",
        "coverage_rate",
    }
    missing_inventory_cols = required_inventory_cols - set(inventory_rows[0])
    if missing_inventory_cols:
        fail("data_enrichment_current_inventory.csv missing columns: " + ", ".join(sorted(missing_inventory_cols)))

    print("OK: data enrichment review contract passed")
    print(f"REPORT: {REVIEW_MD_PATH}")


if __name__ == "__main__":
    main()
