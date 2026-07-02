from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "project_config.json"
GENERATION_DECISION_PATH = PROJECT_ROOT / "decision_layer" / "attention_disposition_feature_generation_decision.json"
FEATURE_CONTRACT_PATH = PROJECT_ROOT / "feature_layer" / "attention_disposition_feature_contract.md"
FEATURE_SCHEMA_PATH = PROJECT_ROOT / "feature_layer" / "attention_disposition_feature_schema.csv"
BACKFILLED_EVENT_PATH = PROJECT_ROOT / "inputs" / "event_risk_calendar_backfilled.csv"
REVIEW_PATH = PROJECT_ROOT / "validation_layer" / "attention_disposition_model_input_approval_review.md"
DECISION_PATH = PROJECT_ROOT / "decision_layer" / "attention_disposition_model_input_approval_decision.json"


def fail(message: str) -> None:
    raise SystemExit(f"FAIL: {message}")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        fail(f"missing required file: {path.relative_to(PROJECT_ROOT)}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_config(config: dict[str, Any], generation_decision: dict[str, Any]) -> dict[str, Any]:
    allowed = config.get("allowed_inputs", {})
    if set(allowed) != {"stock_daily_all", "market_daily", "theme_group"}:
        fail("allowed_inputs must remain the three core source files")
    if any("event_risk_calendar" in str(value) for value in allowed.values()):
        fail("event data must not be placed in allowed_inputs")

    candidate_inputs = config.get("candidate_model_feature_inputs", {})
    candidate = candidate_inputs.get("attention_disposition_events")
    if not isinstance(candidate, dict):
        fail("missing candidate_model_feature_inputs.attention_disposition_events")
    if Path(candidate.get("path", "")) != BACKFILLED_EVENT_PATH:
        fail("attention/disposition candidate path is not the approved backfilled event file")
    if candidate.get("status") != "approved_for_next_training_candidate":
        fail("attention/disposition candidate input is not explicitly approved as a training candidate")
    if candidate.get("scope") != "attention_disposition_only":
        fail("attention/disposition candidate scope must stay limited")
    if candidate.get("approval_decision") != str(DECISION_PATH.relative_to(PROJECT_ROOT)):
        fail("candidate input must point to this approval decision")

    if generation_decision.get("status") != "attention_disposition_feature_generation_check_passed":
        fail("feature generation check must pass before input approval")
    if generation_decision.get("future_event_join_violations") != 0:
        fail("feature generation has future event join violations")
    if generation_decision.get("large_feature_matrix_written") is not False:
        fail("generation step must not have written a large feature matrix")
    if generation_decision.get("new_input_not_enabled") is not True:
        fail("generation step must not have already enabled the input")
    if generation_decision.get("requires_red_light_before_model_input") is not True:
        fail("generation step must require red-light approval")

    if not FEATURE_CONTRACT_PATH.exists() or not FEATURE_SCHEMA_PATH.exists() or not BACKFILLED_EVENT_PATH.exists():
        fail("missing feature contract, feature schema, or backfilled event file")
    return candidate


def write_review(decision: dict[str, Any]) -> None:
    lines = [
        "# Attention / Disposition Model Input Approval Review",
        "",
        "- Scope: candidate model input approval only.",
        "- Formal output: unchanged.",
        "- Model training: not executed.",
        "- Core allowed inputs remain the original three CSV files.",
        "- Approved only as a limited candidate feature input for the next main-model training run.",
        "",
        "## 白話結論",
        "",
        "注意/處置事件資料已通過覆蓋率、特徵契約與防偷看生成檢查，因此可以列為下一次主模型重訓的候選特徵輸入；但這一步沒有重訓，也沒有產生正式候選。",
        "",
        f"- Status: `{decision['status']}`",
        f"- Recommended next step: `{decision['recommended_next_step']}`",
        f"- Candidate input key: `{decision['candidate_input_key']}`",
        f"- Scope: `{decision['allowed_scope']}`",
        f"- Approved feature count: {decision['approved_feature_count']}",
        "",
        "## Boundaries",
        "",
        "- The raw event file still cannot be used outside the approved attention/disposition features.",
        "- Event titles and source text are still blocked as NLP inputs.",
        "- Events are still not labels and cannot define success or failure.",
        "- This approval does not update formal candidates.",
        "- This approval does not train the model.",
        "",
        "## Next Step",
        "",
        "下一步可以修改唯一主模型訓練管線，讓它在重訓時讀取這個候選輸入並產生注意/處置特徵；重訓後仍必須通過 holdout 驗證才可考慮正式輸出。",
        "",
    ]
    REVIEW_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    config = load_json(CONFIG_PATH)
    generation_decision = load_json(GENERATION_DECISION_PATH)
    candidate = validate_config(config, generation_decision)

    decision = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "attention_disposition_candidate_model_input_approved",
        "recommended_next_step": "wire_attention_disposition_features_into_main_training_pipeline",
        "candidate_input_key": "attention_disposition_events",
        "candidate_input_path": candidate["path"],
        "allowed_scope": "attention_disposition_only",
        "approved_feature_count": len(generation_decision.get("feature_names", [])),
        "approved_features": generation_decision.get("feature_names", []),
        "core_allowed_inputs_unchanged": True,
        "model_training_executed": False,
        "formal_outputs_unchanged": True,
        "may_be_used_by_next_training_pipeline": True,
        "still_requires_holdout_validation_before_formal_output": True,
    }
    DECISION_PATH.write_text(json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_review(decision)

    print("OK: attention/disposition model input approval review completed")
    print(f"STATUS: {decision['status']}")
    print(f"NEXT_STEP: {decision['recommended_next_step']}")
    print(f"REVIEW: {REVIEW_PATH}")


if __name__ == "__main__":
    main()
