from __future__ import annotations

import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "project_config.json"
SUMMARY_PATH = PROJECT_ROOT / "validation_layer" / "attention_disposition_feature_generation_summary.csv"
SPLIT_PATH = PROJECT_ROOT / "validation_layer" / "attention_disposition_feature_generation_by_split.csv"
LEAKAGE_AUDIT_PATH = PROJECT_ROOT / "validation_layer" / "attention_disposition_feature_generation_leakage_audit.csv"
REVIEW_PATH = PROJECT_ROOT / "validation_layer" / "attention_disposition_feature_generation_review.md"
DECISION_PATH = PROJECT_ROOT / "decision_layer" / "attention_disposition_feature_generation_decision.json"


REQUIRED_FEATURES = {
    "attention_disposition_known_count_1d",
    "attention_disposition_known_count_3d",
    "attention_disposition_known_count_10d",
    "attention_active_on_signal_date",
    "disposition_active_on_signal_date",
    "days_since_last_attention_disposition",
    "has_attention_disposition_history_20d",
}


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


def metric_value(rows: list[dict[str, str]], metric: str) -> str:
    for row in rows:
        if row.get("metric") == metric:
            return row.get("value", "")
    fail(f"missing metric: {metric}")


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    allowed = config.get("allowed_inputs", {})
    if set(allowed) != {"stock_daily_all", "market_daily", "theme_group"}:
        fail("approved model inputs changed")
    if any("event_risk_calendar" in str(value) for value in allowed.values()):
        fail("event data must not be enabled as approved model input")

    if not DECISION_PATH.exists():
        fail("missing generation decision")
    decision = json.loads(DECISION_PATH.read_text(encoding="utf-8"))
    if decision.get("status") != "attention_disposition_feature_generation_check_passed":
        fail("generation check did not pass")
    if decision.get("recommended_next_step") != "prepare_attention_disposition_model_input_approval_review":
        fail("unexpected next step")
    if set(decision.get("feature_names", [])) != REQUIRED_FEATURES:
        fail("generation decision feature names do not match contract")
    if decision.get("future_event_join_violations") != 0:
        fail("future event join violations must be zero")
    if decision.get("large_feature_matrix_written") is not False:
        fail("large feature matrix must not be written in this check")
    if decision.get("new_input_not_enabled") is not True:
        fail("new input must remain disabled")
    if decision.get("do_not_retrain_yet") is not True:
        fail("generation check must not permit retraining")
    if decision.get("formal_outputs_unchanged") is not True:
        fail("formal outputs must remain unchanged")
    if decision.get("requires_red_light_before_model_input") is not True:
        fail("model input enablement must remain red-light gated")

    summary_rows = read_rows(SUMMARY_PATH)
    if int(float(metric_value(summary_rows, "signal_rows_checked"))) <= 0:
        fail("no signal rows checked")
    if int(float(metric_value(summary_rows, "eligible_attention_disposition_event_rows"))) <= 0:
        fail("no eligible attention/disposition event rows")
    if int(float(metric_value(summary_rows, "rows_with_known_10d"))) <= 0:
        fail("generated features never mark any 10d known-event row")

    split_rows = read_rows(SPLIT_PATH)
    if {row["split"] for row in split_rows} != {"train", "development", "holdout"}:
        fail("split generation summary must include train, development, and holdout")
    for row in split_rows:
        if int(float(row["signal_rows"])) <= 0:
            fail(f"{row['split']} has no signal rows")

    leakage_rows = read_rows(LEAKAGE_AUDIT_PATH)
    if int(float(metric_value(leakage_rows, "future_event_join_violations"))) != 0:
        fail("future event leakage must be zero")
    if int(float(metric_value(leakage_rows, "eligible_event_rows"))) != int(decision["eligible_event_rows"]):
        fail("eligible event row count mismatch")

    if not REVIEW_PATH.exists():
        fail("missing generation review")
    review = REVIEW_PATH.read_text(encoding="utf-8")
    for phrase in [
        "feature generation leakage check only",
        "Model training: not executed.",
        "New model input: not enabled.",
        "Large feature matrix: not written.",
        "交易日窗口安全生成",
        "Every event must satisfy `signal_usable_date <= signal_date`.",
        "validation summaries, not approved model input files",
    ]:
        if phrase not in review:
            fail(f"generation review missing phrase: {phrase}")

    print("OK: attention/disposition feature generation check passed")
    print(f"STATUS: {decision['status']}")
    print(f"NEXT_STEP: {decision['recommended_next_step']}")


if __name__ == "__main__":
    main()
