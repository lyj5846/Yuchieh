from __future__ import annotations

import json
from pathlib import Path

from run_main_model_training_pipeline import ATTENTION_DISPOSITION_FEATURES, build_training_frame, load_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "project_config.json"
APPROVAL_DECISION_PATH = PROJECT_ROOT / "decision_layer" / "attention_disposition_model_input_approval_decision.json"


def fail(message: str) -> None:
    raise SystemExit(f"FAIL: {message}")


def main() -> None:
    config = load_json(CONFIG_PATH)
    allowed = config.get("allowed_inputs", {})
    if set(allowed) != {"stock_daily_all", "market_daily", "theme_group"}:
        fail("core allowed_inputs changed")
    if any("event_risk_calendar" in str(value) for value in allowed.values()):
        fail("event file must not be placed in core allowed_inputs")

    candidates = config.get("candidate_model_feature_inputs", {})
    candidate = candidates.get("attention_disposition_events")
    if not isinstance(candidate, dict):
        fail("missing approved attention/disposition candidate input")
    if candidate.get("status") != "approved_for_next_training_candidate":
        fail("attention/disposition candidate input is not approved for next training")
    if candidate.get("scope") != "attention_disposition_only":
        fail("attention/disposition candidate scope must remain limited")
    if candidate.get("approval_decision") != str(APPROVAL_DECISION_PATH.relative_to(PROJECT_ROOT)):
        fail("attention/disposition candidate approval path mismatch")

    approval = json.loads(APPROVAL_DECISION_PATH.read_text(encoding="utf-8"))
    if approval.get("status") != "attention_disposition_candidate_model_input_approved":
        fail("attention/disposition model input approval has not passed")
    if approval.get("model_training_executed") is not False:
        fail("approval review must not have trained the model")
    if approval.get("formal_outputs_unchanged") is not True:
        fail("formal outputs must remain unchanged")

    frame, feature_cols = build_training_frame(config)
    missing = sorted(set(ATTENTION_DISPOSITION_FEATURES) - set(feature_cols))
    if missing:
        fail("main training feature list missing attention/disposition features: " + ", ".join(missing))
    if frame.empty:
        fail("training frame is empty")
    for feature in ATTENTION_DISPOSITION_FEATURES:
        if feature not in frame.columns:
            fail(f"training frame missing feature column: {feature}")
    signal_features = [
        "attention_disposition_known_count_1d",
        "attention_disposition_known_count_3d",
        "attention_disposition_known_count_10d",
        "attention_active_on_signal_date",
        "disposition_active_on_signal_date",
        "has_attention_disposition_history_20d",
    ]
    non_zero = {feature: int((frame[feature].astype(float) > 0).sum()) for feature in signal_features}
    if max(non_zero.values()) <= 0:
        fail("attention/disposition features contain no non-zero signal")
    if frame["days_since_last_attention_disposition"].isna().any():
        fail("days_since_last_attention_disposition must be imputed before training")
    if frame["days_since_last_attention_disposition"].lt(0).any():
        fail("days_since_last_attention_disposition must not be negative")

    print("OK: attention/disposition main training wiring passed")
    print("FEATURES: " + ", ".join(ATTENTION_DISPOSITION_FEATURES))
    print("NON_ZERO: " + ", ".join(f"{key}={value}" for key, value in non_zero.items()))


if __name__ == "__main__":
    main()
