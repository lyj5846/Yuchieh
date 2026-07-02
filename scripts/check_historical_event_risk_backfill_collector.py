from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "project_config.json"
SCHEMA_PATH = PROJECT_ROOT / "data_layer" / "event_risk_calendar_schema.csv"
OUTPUT_PATH = PROJECT_ROOT / "inputs" / "event_risk_calendar_backfilled.csv"
SOURCE_SUMMARY_PATH = PROJECT_ROOT / "validation_layer" / "historical_event_risk_backfill_source_summary.csv"
REPORT_PATH = PROJECT_ROOT / "validation_layer" / "historical_event_risk_backfill_report.md"
DECISION_PATH = PROJECT_ROOT / "decision_layer" / "historical_event_risk_backfill_decision.json"


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


def schema_columns() -> list[str]:
    return [row["field_name"] for row in read_rows(SCHEMA_PATH)]


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    allowed = config.get("allowed_inputs", {})
    if set(allowed) != {"stock_daily_all", "market_daily", "theme_group"}:
        fail("approved model inputs changed")
    if any("event_risk_calendar" in str(value) for value in allowed.values()):
        fail("event calendar output must not be enabled as a model input")

    rows = read_rows(OUTPUT_PATH)
    with OUTPUT_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != schema_columns():
            fail("backfilled calendar columns do not match schema")
    if len(rows) <= 145:
        fail("historical backfill did not add rows beyond the original snapshot")

    seen: set[tuple[str, str, str, str, str]] = set()
    split_counts = {"train": 0, "development": 0, "holdout": 0}
    train_end = datetime.strptime(config["time_split"]["train_end"], "%Y-%m-%d")
    dev_start = datetime.strptime(config["time_split"]["dev_start"], "%Y-%m-%d")
    dev_end = datetime.strptime(config["time_split"]["dev_end"], "%Y-%m-%d")
    holdout_start = datetime.strptime(config["time_split"]["holdout_start"], "%Y-%m-%d")
    for index, row in enumerate(rows, start=2):
        if not row["stock_id"] or not row["event_type"] or not row["announcement_datetime"]:
            fail(f"row {index} missing required fields")
        datetime.strptime(row["announcement_datetime"], "%Y-%m-%d %H:%M:%S")
        signal_day = datetime.strptime(row["signal_usable_date"], "%Y-%m-%d")
        if signal_day <= train_end:
            split_counts["train"] += 1
        elif dev_start <= signal_day <= dev_end:
            split_counts["development"] += 1
        elif signal_day >= holdout_start:
            split_counts["holdout"] += 1
        if row["known_before_signal_close"] not in {"true", "false"}:
            fail(f"row {index} has invalid known_before_signal_close")
        if row["post_close_pre_next_open"] not in {"true", "false"}:
            fail(f"row {index} has invalid post_close_pre_next_open")
        if not row["raw_payload_hash"].startswith("sha256:"):
            fail(f"row {index} missing raw hash")
        key = (
            row["stock_id"],
            row["event_type"],
            row["announcement_datetime"],
            row["event_title"],
            row["source_name"],
        )
        if key in seen:
            fail(f"duplicate backfill row at line {index}")
        seen.add(key)

    if split_counts["train"] == 0:
        fail("historical backfill must add train rows")
    if split_counts["development"] == 0:
        fail("historical backfill must add development rows")

    source_rows = read_rows(SOURCE_SUMMARY_PATH)
    source_names = {row["source_name"] for row in source_rows}
    if "TWSE historical attention" not in source_names:
        fail("source summary missing TWSE historical attention")
    if "TWSE historical disposition" not in source_names:
        fail("source summary missing TWSE historical disposition")

    if not REPORT_PATH.exists():
        fail("missing historical backfill report")
    report = REPORT_PATH.read_text(encoding="utf-8")
    for phrase in [
        "historical backfill collector output",
        "Formal output: unchanged.",
        "Model training: not executed.",
        "not enabled in `project_config.json`",
        "still need separate source work",
    ]:
        if phrase not in report:
            fail(f"backfill report missing phrase: {phrase}")

    if not DECISION_PATH.exists():
        fail("missing historical backfill decision")
    decision = json.loads(DECISION_PATH.read_text(encoding="utf-8"))
    if decision.get("status") not in {"historical_backfill_output_ready", "historical_backfill_still_insufficient"}:
        fail("unexpected historical backfill status")
    if decision.get("recommended_next_step") != "review_historical_event_risk_backfill_coverage":
        fail("unexpected next step after historical backfill")
    if decision.get("new_input_not_enabled") is not True:
        fail("backfilled input must remain disabled")
    if decision.get("do_not_retrain_yet") is not True:
        fail("historical backfill must still block immediate retraining")
    if decision.get("formal_outputs_unchanged") is not True:
        fail("formal output must remain unchanged")

    print("OK: historical event risk backfill contract passed")
    print(f"ROWS: {len(rows)}")
    print(f"SPLITS: {split_counts}")


if __name__ == "__main__":
    main()
