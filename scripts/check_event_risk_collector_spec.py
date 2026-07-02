from __future__ import annotations

import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "project_config.json"
CONTRACT_PATH = PROJECT_ROOT / "data_layer" / "event_risk_calendar_source_contract.md"
SCHEMA_PATH = PROJECT_ROOT / "data_layer" / "event_risk_calendar_schema.csv"
REVIEW_PATH = PROJECT_ROOT / "validation_layer" / "event_risk_collector_spec_review.md"
DECISION_PATH = PROJECT_ROOT / "decision_layer" / "event_risk_collector_spec_decision.json"


def fail(message: str) -> None:
    raise SystemExit(f"FAIL: {message}")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        fail(f"missing output: {path.relative_to(PROJECT_ROOT)}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        fail(f"{path.relative_to(PROJECT_ROOT)} is empty")
    return rows


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    allowed = config.get("allowed_inputs", {})
    if set(allowed) != {"stock_daily_all", "market_daily", "theme_group"}:
        fail("collector spec must not change approved input list")
    if any("event_risk_calendar" in str(value) for value in allowed.values()):
        fail("event_risk_calendar must not be enabled as a model input in this step")

    if not CONTRACT_PATH.exists():
        fail("missing event risk source contract")
    contract = CONTRACT_PATH.read_text(encoding="utf-8")
    for phrase in [
        "specification only; no new data source is enabled yet",
        "Future file path: `inputs/event_risk_calendar.csv`",
        "This file is not added to `project_config.json` in this step.",
        "Main model features may only use events already public by signal-day close.",
        "Post-close events cannot enter the main model unless the decision-time contract is explicitly changed.",
        "Missing `announcement_datetime` means the row is not eligible for training.",
        "This step does not train or promote a model.",
    ]:
        if phrase not in contract:
            fail(f"contract missing required phrase: {phrase}")

    schema_rows = read_csv_rows(SCHEMA_PATH)
    required_schema_columns = {
        "field_name",
        "field_type",
        "required",
        "description",
        "example",
        "no_leakage_rule",
    }
    missing_schema_cols = required_schema_columns - set(schema_rows[0])
    if missing_schema_cols:
        fail("event risk schema missing columns: " + ", ".join(sorted(missing_schema_cols)))
    field_names = {row["field_name"] for row in schema_rows}
    for field_name in [
        "stock_id",
        "event_type",
        "announcement_datetime",
        "signal_usable_date",
        "known_before_signal_close",
        "post_close_pre_next_open",
    ]:
        if field_name not in field_names:
            fail(f"schema missing required timing field: {field_name}")

    if not REVIEW_PATH.exists():
        fail("missing event risk spec review")
    review = REVIEW_PATH.read_text(encoding="utf-8")
    for phrase in [
        "collector specification only",
        "Formal output: unchanged.",
        "Model training: not executed.",
        "New input source: not enabled.",
        "Do not retrain yet: `true`",
    ]:
        if phrase not in review:
            fail(f"review missing required phrase: {phrase}")

    if not DECISION_PATH.exists():
        fail("missing event risk collector spec decision")
    decision = json.loads(DECISION_PATH.read_text(encoding="utf-8"))
    if decision.get("status") != "collector_spec_ready":
        fail("event risk spec decision must be collector_spec_ready")
    if decision.get("recommended_next_step") != "implement_event_risk_calendar_collector":
        fail("unexpected event risk next step")
    if decision.get("data_need_id") != "event_risk_calendar":
        fail("event risk decision must reference event_risk_calendar")
    if decision.get("do_not_retrain_yet") is not True:
        fail("event risk spec must block retraining")
    if decision.get("formal_outputs_unchanged") is not True:
        fail("formal output must remain unchanged")

    required_types = {
        "material_info",
        "suspension",
        "resumption",
        "attention",
        "disposition",
        "investor_meeting",
        "ex_dividend",
        "corporate_action",
    }
    if set(decision.get("required_event_types", [])) != required_types:
        fail("required event type set is incomplete")

    print("OK: event risk collector spec contract passed")
    print(f"CONTRACT: {CONTRACT_PATH}")


if __name__ == "__main__":
    main()
