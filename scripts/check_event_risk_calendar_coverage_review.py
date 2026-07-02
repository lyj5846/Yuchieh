from __future__ import annotations

import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "project_config.json"
REVIEW_PATH = PROJECT_ROOT / "validation_layer" / "event_risk_calendar_coverage_review.md"
SPLIT_COVERAGE_PATH = PROJECT_ROOT / "validation_layer" / "event_risk_calendar_coverage_by_split.csv"
MONTH_COVERAGE_PATH = PROJECT_ROOT / "validation_layer" / "event_risk_calendar_coverage_by_month.csv"
EVENT_TYPE_COVERAGE_PATH = PROJECT_ROOT / "validation_layer" / "event_risk_calendar_coverage_by_type.csv"
DECISION_PATH = PROJECT_ROOT / "decision_layer" / "event_risk_calendar_coverage_decision.json"


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
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    allowed = config.get("allowed_inputs", {})
    if set(allowed) != {"stock_daily_all", "market_daily", "theme_group"}:
        fail("approved model inputs changed")
    if any("event_risk_calendar" in str(value) for value in allowed.values()):
        fail("event_risk_calendar must not be enabled before coverage review passes")

    if not REVIEW_PATH.exists():
        fail("missing coverage review")
    review = REVIEW_PATH.read_text(encoding="utf-8")
    for phrase in [
        "coverage review only",
        "Formal output: unchanged.",
        "Model training: not executed.",
        "not enabled in `project_config.json`",
        "historical backfill is required before model use",
    ]:
        if phrase not in review:
            fail(f"coverage review missing phrase: {phrase}")

    if not DECISION_PATH.exists():
        fail("missing coverage decision")
    decision = json.loads(DECISION_PATH.read_text(encoding="utf-8"))
    if decision.get("status") != "coverage_not_training_ready":
        fail("current event calendar snapshot should not be training-ready")
    if decision.get("recommended_next_step") != "build_historical_event_risk_backfill_collector":
        fail("unexpected next step after coverage review")
    if decision.get("new_input_not_enabled") is not True:
        fail("event input must remain disabled")
    if decision.get("do_not_retrain_yet") is not True:
        fail("coverage review must block retraining when history is insufficient")
    if decision.get("formal_outputs_unchanged") is not True:
        fail("formal output must remain unchanged")

    split_rows = read_rows(SPLIT_COVERAGE_PATH)
    split_names = {row["split"] for row in split_rows}
    if split_names != {"train", "development", "holdout"}:
        fail("split coverage must include train, development, and holdout")
    required_split_cols = {
        "split",
        "start_date",
        "end_date",
        "trading_days",
        "event_rows",
        "event_trading_days",
        "event_stock_count",
        "event_day_coverage_rate",
    }
    missing_split_cols = required_split_cols - set(split_rows[0])
    if missing_split_cols:
        fail("split coverage missing columns: " + ", ".join(sorted(missing_split_cols)))
    train = next(row for row in split_rows if row["split"] == "train")
    development = next(row for row in split_rows if row["split"] == "development")
    if int(train["event_rows"]) != 0:
        fail("current snapshot unexpectedly has train event coverage")
    if int(development["event_rows"]) != 0:
        fail("current snapshot unexpectedly has development event coverage")

    month_rows = read_rows(MONTH_COVERAGE_PATH)
    if "month" not in month_rows[0] or "event_rows" not in month_rows[0]:
        fail("month coverage has unexpected schema")

    type_rows = read_rows(EVENT_TYPE_COVERAGE_PATH)
    if "event_type" not in type_rows[0] or "event_rows" not in type_rows[0]:
        fail("event type coverage has unexpected schema")

    print("OK: event risk calendar coverage review contract passed")
    print(f"STATUS: {decision['status']}")
    print(f"NEXT_STEP: {decision['recommended_next_step']}")


if __name__ == "__main__":
    main()
