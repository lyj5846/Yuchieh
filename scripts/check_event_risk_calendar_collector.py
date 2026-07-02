from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "project_config.json"
SCHEMA_PATH = PROJECT_ROOT / "data_layer" / "event_risk_calendar_schema.csv"
OUTPUT_PATH = PROJECT_ROOT / "inputs" / "event_risk_calendar.csv"
SOURCE_SUMMARY_PATH = PROJECT_ROOT / "validation_layer" / "event_risk_calendar_source_summary.csv"
REPORT_PATH = PROJECT_ROOT / "validation_layer" / "event_risk_calendar_collector_report.md"
DECISION_PATH = PROJECT_ROOT / "decision_layer" / "event_risk_calendar_collector_decision.json"

EVENT_TYPES = {
    "material_info",
    "suspension",
    "resumption",
    "attention",
    "disposition",
    "investor_meeting",
    "ex_dividend",
    "corporate_action",
}


def fail(message: str) -> None:
    raise SystemExit(f"FAIL: {message}")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        fail(f"missing file: {path.relative_to(PROJECT_ROOT)}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows


def schema_columns() -> list[str]:
    rows = read_csv_rows(SCHEMA_PATH)
    if not rows:
        fail("schema file is empty")
    return [row["field_name"] for row in rows]


def required_fields() -> list[str]:
    rows = read_csv_rows(SCHEMA_PATH)
    return [row["field_name"] for row in rows if row.get("required") == "yes"]


def parse_datetime(value: str) -> datetime:
    try:
        return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    except ValueError as exc:
        raise ValueError(f"bad announcement_datetime: {value}") from exc


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    allowed = config.get("allowed_inputs", {})
    if set(allowed) != {"stock_daily_all", "market_daily", "theme_group"}:
        fail("approved model inputs changed")
    if any("event_risk_calendar" in str(value) for value in allowed.values()):
        fail("event_risk_calendar must not be enabled as a model input yet")

    expected_columns = schema_columns()
    rows = read_csv_rows(OUTPUT_PATH)
    with OUTPUT_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != expected_columns:
            fail("event_risk_calendar.csv columns do not match schema")

    if not rows:
        fail("event_risk_calendar.csv has no rows")

    required = required_fields()
    seen: set[tuple[str, str, str, str, str]] = set()
    for index, row in enumerate(rows, start=2):
        for field in required:
            if not row.get(field, "").strip():
                fail(f"row {index} missing required field: {field}")
        if row["event_type"] not in EVENT_TYPES:
            fail(f"row {index} has invalid event_type: {row['event_type']}")
        parse_datetime(row["announcement_datetime"])
        if row["known_before_signal_close"] not in {"true", "false"}:
            fail(f"row {index} has invalid known_before_signal_close")
        if row["post_close_pre_next_open"] not in {"true", "false"}:
            fail(f"row {index} has invalid post_close_pre_next_open")
        if row["known_before_signal_close"] == "true" and row["post_close_pre_next_open"] == "true":
            fail(f"row {index} cannot be both known before close and post-close")
        if row["known_before_signal_close"] == "false" and row["post_close_pre_next_open"] == "false":
            fail(f"row {index} must be either known before close or post-close")
        if not row["raw_payload_hash"].startswith("sha256:"):
            fail(f"row {index} missing raw payload hash")
        key = (
            row["stock_id"],
            row["event_type"],
            row["announcement_datetime"],
            row["event_title"],
            row["source_name"],
        )
        if key in seen:
            fail(f"duplicate event row at line {index}")
        seen.add(key)

    source_rows = read_csv_rows(SOURCE_SUMMARY_PATH)
    if not source_rows:
        fail("source summary is empty")
    source_total = sum(int(row.get("kept_events") or 0) for row in source_rows)
    if source_total < len(rows):
        fail("source summary kept_events is smaller than output rows")

    if not REPORT_PATH.exists():
        fail("missing collector report")
    report = REPORT_PATH.read_text(encoding="utf-8")
    for phrase in [
        "official API snapshot collection",
        "Formal output: unchanged.",
        "Model training: not executed.",
        "New input source: produced but not enabled in `project_config.json`.",
        "This collector does not train or promote a model.",
    ]:
        if phrase not in report:
            fail(f"collector report missing phrase: {phrase}")

    if not DECISION_PATH.exists():
        fail("missing collector decision")
    decision = json.loads(DECISION_PATH.read_text(encoding="utf-8"))
    if decision.get("status") != "collector_output_ready":
        fail("collector output must be ready")
    if decision.get("recommended_next_step") != "review_event_risk_calendar_coverage":
        fail("unexpected next step after collector")
    if decision.get("new_input_not_enabled") is not True:
        fail("new input must remain disabled")
    if decision.get("do_not_retrain_yet") is not True:
        fail("collector must block immediate retraining")
    if decision.get("formal_outputs_unchanged") is not True:
        fail("formal output must remain unchanged")
    if int(decision.get("rows", 0)) != len(rows):
        fail("collector decision row count does not match output")

    print("OK: event risk calendar collector contract passed")
    print(f"ROWS: {len(rows)}")
    print(f"OUTPUT: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
