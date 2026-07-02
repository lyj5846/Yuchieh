from __future__ import annotations

import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "project_config.json"
REVIEW_PATH = PROJECT_ROOT / "validation_layer" / "historical_event_risk_backfill_coverage_review.md"
SPLIT_COVERAGE_PATH = PROJECT_ROOT / "validation_layer" / "historical_event_risk_backfill_coverage_by_split.csv"
TYPE_COVERAGE_PATH = PROJECT_ROOT / "validation_layer" / "historical_event_risk_backfill_coverage_by_type.csv"
SOURCE_COVERAGE_PATH = PROJECT_ROOT / "validation_layer" / "historical_event_risk_backfill_coverage_by_source.csv"
MONTH_COVERAGE_PATH = PROJECT_ROOT / "validation_layer" / "historical_event_risk_backfill_coverage_by_month.csv"
DECISION_PATH = PROJECT_ROOT / "decision_layer" / "historical_event_risk_backfill_coverage_decision.json"


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
        fail("backfilled event data must not be enabled as a model input yet")

    if not REVIEW_PATH.exists():
        fail("missing historical backfill coverage review")
    review = REVIEW_PATH.read_text(encoding="utf-8")
    for phrase in [
        "historical backfill coverage review only",
        "Formal output: unchanged.",
        "Model training: not executed.",
        "produced but not enabled in `project_config.json`",
        "Full event-risk wording is blocked",
    ]:
        if phrase not in review:
            fail(f"coverage review missing phrase: {phrase}")

    if not DECISION_PATH.exists():
        fail("missing historical backfill coverage decision")
    decision = json.loads(DECISION_PATH.read_text(encoding="utf-8"))
    if decision.get("status") != "coverage_ready_for_limited_attention_disposition_features":
        fail("expected limited attention/disposition coverage status")
    if decision.get("recommended_next_step") != "prepare_limited_attention_disposition_feature_contract":
        fail("unexpected next step after historical coverage review")
    if decision.get("allowed_scope") != "attention_disposition_only":
        fail("allowed scope must stay limited")
    if decision.get("complete_event_ready") is not False:
        fail("full event-risk coverage must not be marked ready")
    if decision.get("new_input_not_enabled") is not True:
        fail("event input must remain disabled")
    if decision.get("do_not_retrain_yet") is not True:
        fail("coverage review must not directly allow retraining")
    if decision.get("formal_outputs_unchanged") is not True:
        fail("formal output must remain unchanged")

    split_rows = read_rows(SPLIT_COVERAGE_PATH)
    split_names = {row["split"] for row in split_rows}
    if split_names != {"train", "development", "holdout"}:
        fail("split coverage must include train, development, and holdout")
    for split_name in ["train", "development", "holdout"]:
        row = next(item for item in split_rows if item["split"] == split_name)
        if int(row["event_rows"]) <= 0:
            fail(f"{split_name} has no backfilled events")
        if float(row["event_day_coverage_rate"]) <= 0.20:
            fail(f"{split_name} event day coverage too low")

    type_rows = read_rows(TYPE_COVERAGE_PATH)
    top_type = type_rows[0]
    if top_type["event_type"] != "attention":
        fail("expected attention to be the top event type")
    if float(top_type["event_row_share"]) <= 0.80:
        fail("attention concentration should be explicitly visible")

    source_rows = read_rows(SOURCE_COVERAGE_PATH)
    top_source = source_rows[0]
    if "TWSE historical attention" not in top_source["source_name"]:
        fail("expected TWSE historical attention to be the dominant source")
    if float(top_source["event_row_share"]) <= 0.80:
        fail("source concentration should be explicitly visible")

    month_rows = read_rows(MONTH_COVERAGE_PATH)
    if "month" not in month_rows[0] or "event_rows" not in month_rows[0]:
        fail("month coverage has unexpected schema")

    print("OK: historical event risk backfill coverage contract passed")
    print(f"STATUS: {decision['status']}")
    print(f"NEXT_STEP: {decision['recommended_next_step']}")


if __name__ == "__main__":
    main()
