from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "project_config.json"
GENERATION_DECISION_PATH = PROJECT_ROOT / "decision_layer" / "attention_disposition_feature_generation_decision.json"
REVIEW_PATH = PROJECT_ROOT / "validation_layer" / "attention_disposition_model_input_approval_review.md"
DECISION_PATH = PROJECT_ROOT / "decision_layer" / "attention_disposition_model_input_approval_decision.json"
BACKFILLED_EVENT_PATH = PROJECT_ROOT / "inputs" / "event_risk_calendar_backfilled.csv"


def fail(message: str) -> None:
    raise SystemExit(f"FAIL: {message}")


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    allowed = config.get("allowed_inputs", {})
    if set(allowed) != {"stock_daily_all", "market_daily", "theme_group"}:
        fail("core allowed_inputs changed")
    if any("event_risk_calendar" in str(value) for value in allowed.values()):
        fail("event file must not be added to allowed_inputs")

    candidate_inputs = config.get("candidate_model_feature_inputs", {})
    candidate = candidate_inputs.get("attention_disposition_events")
    if not isinstance(candidate, dict):
        fail("missing attention_disposition_events candidate input")
    if Path(candidate.get("path", "")) != BACKFILLED_EVENT_PATH:
        fail("candidate input path mismatch")
    if candidate.get("status") != "approved_for_next_training_candidate":
        fail("candidate input status mismatch")
    if candidate.get("scope") != "attention_disposition_only":
        fail("candidate input scope mismatch")
    if candidate.get("approval_decision") != str(DECISION_PATH.relative_to(PROJECT_ROOT)):
        fail("candidate input approval decision mismatch")

    generation_decision = json.loads(GENERATION_DECISION_PATH.read_text(encoding="utf-8"))
    if generation_decision.get("status") != "attention_disposition_feature_generation_check_passed":
        fail("generation check has not passed")
    if generation_decision.get("future_event_join_violations") != 0:
        fail("generation check has leakage violations")
    if generation_decision.get("do_not_retrain_yet") is not True:
        fail("generation decision should still block retraining before approval review")

    if not DECISION_PATH.exists():
        fail("missing approval decision")
    decision = json.loads(DECISION_PATH.read_text(encoding="utf-8"))
    if decision.get("status") != "attention_disposition_candidate_model_input_approved":
        fail("approval status mismatch")
    if decision.get("recommended_next_step") != "wire_attention_disposition_features_into_main_training_pipeline":
        fail("unexpected next step")
    if decision.get("candidate_input_key") != "attention_disposition_events":
        fail("candidate input key mismatch")
    if decision.get("allowed_scope") != "attention_disposition_only":
        fail("approval scope mismatch")
    if decision.get("core_allowed_inputs_unchanged") is not True:
        fail("approval must keep core inputs unchanged")
    if decision.get("model_training_executed") is not False:
        fail("approval review must not train")
    if decision.get("formal_outputs_unchanged") is not True:
        fail("formal outputs must remain unchanged")
    if decision.get("may_be_used_by_next_training_pipeline") is not True:
        fail("candidate input must be allowed for the next training pipeline")
    if decision.get("still_requires_holdout_validation_before_formal_output") is not True:
        fail("formal output must still require holdout validation")

    if not REVIEW_PATH.exists():
        fail("missing approval review")
    review = REVIEW_PATH.read_text(encoding="utf-8")
    for phrase in [
        "candidate model input approval only",
        "Model training: not executed.",
        "Core allowed inputs remain the original three CSV files.",
        "下一次主模型重訓的候選特徵輸入",
        "This approval does not update formal candidates.",
        "This approval does not train the model.",
    ]:
        if phrase not in review:
            fail(f"approval review missing phrase: {phrase}")

    print("OK: attention/disposition model input approval contract passed")
    print(f"STATUS: {decision['status']}")
    print(f"NEXT_STEP: {decision['recommended_next_step']}")


if __name__ == "__main__":
    main()
