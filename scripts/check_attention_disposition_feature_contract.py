from __future__ import annotations

import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "project_config.json"
FEATURE_CONTRACT_PATH = PROJECT_ROOT / "feature_layer" / "attention_disposition_feature_contract.md"
FEATURE_SCHEMA_PATH = PROJECT_ROOT / "feature_layer" / "attention_disposition_feature_schema.csv"
REVIEW_PATH = PROJECT_ROOT / "validation_layer" / "attention_disposition_feature_contract_review.md"
DECISION_PATH = PROJECT_ROOT / "decision_layer" / "attention_disposition_feature_contract_decision.json"


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


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    allowed = config.get("allowed_inputs", {})
    if set(allowed) != {"stock_daily_all", "market_daily", "theme_group"}:
        fail("approved model inputs changed")
    if any("event_risk_calendar" in str(value) for value in allowed.values()):
        fail("event data must not be enabled as an approved model input yet")

    if not FEATURE_CONTRACT_PATH.exists():
        fail("missing attention/disposition feature contract")
    contract = FEATURE_CONTRACT_PATH.read_text(encoding="utf-8")
    for phrase in [
        "contract only; not enabled for training",
        "limited attention/disposition features only",
        "Event file remains outside `project_config.json` approved inputs.",
        "Do not use event rows as success or failure labels.",
        "Do not call these features complete event-risk features.",
        "重訓與正式輸出仍不能在本步驟發生",
    ]:
        if phrase not in contract:
            fail(f"feature contract missing required phrase: {phrase}")
    blocked_phrases = [
        "complete event-risk features are approved",
        "full event-risk features are approved",
        "Status: enabled for training",
        "approved for model training",
    ]
    for phrase in blocked_phrases:
        if phrase in contract:
            fail(f"feature contract contains blocked phrase: {phrase}")

    if not FEATURE_SCHEMA_PATH.exists():
        fail("missing attention/disposition feature schema")
    with FEATURE_SCHEMA_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        fail("feature schema is empty")
    names = {row["feature_name"] for row in rows}
    if names != REQUIRED_FEATURES:
        fail("feature schema does not match the approved feature set")
    for row in rows:
        source_types = row.get("source_event_types", "")
        if "attention" not in source_types and "disposition" not in source_types:
            fail(f"feature does not use attention/disposition source rows: {row['feature_name']}")
        timing_rule = row.get("timing_rule", "")
        if "signal" not in timing_rule:
            fail(f"feature missing signal-date timing rule: {row['feature_name']}")

    if not DECISION_PATH.exists():
        fail("missing attention/disposition feature decision")
    decision = json.loads(DECISION_PATH.read_text(encoding="utf-8"))
    if decision.get("status") != "limited_attention_disposition_feature_contract_ready":
        fail("feature contract status is not ready")
    if decision.get("recommended_next_step") != "build_attention_disposition_feature_generation_check":
        fail("unexpected next step")
    if decision.get("allowed_scope") != "attention_disposition_only":
        fail("allowed scope must remain attention/disposition only")
    if decision.get("approved_feature_count") != len(REQUIRED_FEATURES):
        fail("approved feature count mismatch")
    if decision.get("new_input_not_enabled") is not True:
        fail("new input must remain disabled")
    if decision.get("do_not_retrain_yet") is not True:
        fail("feature contract must not permit retraining")
    if decision.get("formal_outputs_unchanged") is not True:
        fail("formal outputs must remain unchanged")
    if decision.get("requires_red_light_before_model_input") is not True:
        fail("model input enablement must remain red-light gated")

    if not REVIEW_PATH.exists():
        fail("missing attention/disposition feature review")
    review = REVIEW_PATH.read_text(encoding="utf-8")
    for phrase in [
        "contract review only",
        "New model input: not enabled.",
        "Training with the event file before feature generation leakage checks pass.",
        "Formal candidate output.",
    ]:
        if phrase not in review:
            fail(f"review missing required phrase: {phrase}")

    print("OK: attention/disposition feature contract passed")
    print(f"STATUS: {decision['status']}")
    print(f"NEXT_STEP: {decision['recommended_next_step']}")


if __name__ == "__main__":
    main()
