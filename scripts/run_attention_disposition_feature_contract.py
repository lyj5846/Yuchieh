from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "project_config.json"
COVERAGE_DECISION_PATH = PROJECT_ROOT / "decision_layer" / "historical_event_risk_backfill_coverage_decision.json"
SOURCE_CONTRACT_PATH = PROJECT_ROOT / "data_layer" / "event_risk_calendar_source_contract.md"
FEATURE_CONTRACT_PATH = PROJECT_ROOT / "feature_layer" / "attention_disposition_feature_contract.md"
FEATURE_SCHEMA_PATH = PROJECT_ROOT / "feature_layer" / "attention_disposition_feature_schema.csv"
REVIEW_PATH = PROJECT_ROOT / "validation_layer" / "attention_disposition_feature_contract_review.md"
DECISION_PATH = PROJECT_ROOT / "decision_layer" / "attention_disposition_feature_contract_decision.json"


FEATURE_ROWS = [
    {
        "feature_name": "attention_disposition_known_count_1d",
        "type": "numeric",
        "window": "1 trading day",
        "source_event_types": "attention | disposition",
        "meaning": "Count of known attention or disposition rows usable on the signal day.",
        "timing_rule": "signal_usable_date <= signal_date and known_before_signal_close = true",
        "missing_rule": "fill 0 when no eligible event exists",
    },
    {
        "feature_name": "attention_disposition_known_count_3d",
        "type": "numeric",
        "window": "3 trading days",
        "source_event_types": "attention | disposition",
        "meaning": "Recent density of known attention or disposition rows.",
        "timing_rule": "rolling window ends on signal_date; no future or post-close rows",
        "missing_rule": "fill 0 when no eligible event exists",
    },
    {
        "feature_name": "attention_disposition_known_count_10d",
        "type": "numeric",
        "window": "10 trading days",
        "source_event_types": "attention | disposition",
        "meaning": "Short-term accumulation of exchange warning or disposition pressure.",
        "timing_rule": "rolling window ends on signal_date; no future or post-close rows",
        "missing_rule": "fill 0 when no eligible event exists",
    },
    {
        "feature_name": "attention_active_on_signal_date",
        "type": "binary",
        "window": "active range",
        "source_event_types": "attention",
        "meaning": "Whether attention status is active and already known on the signal date.",
        "timing_rule": "event_effective_start_date <= signal_date <= event_effective_end_date when end is available; otherwise use usable date only",
        "missing_rule": "fill 0 when no eligible attention row exists",
    },
    {
        "feature_name": "disposition_active_on_signal_date",
        "type": "binary",
        "window": "active range",
        "source_event_types": "disposition",
        "meaning": "Whether disposition status is active and already known on the signal date.",
        "timing_rule": "event_effective_start_date <= signal_date <= event_effective_end_date when end is available; otherwise use usable date only",
        "missing_rule": "fill 0 when no eligible disposition row exists",
    },
    {
        "feature_name": "days_since_last_attention_disposition",
        "type": "numeric",
        "window": "historical",
        "source_event_types": "attention | disposition",
        "meaning": "Trading days since the latest eligible attention or disposition row.",
        "timing_rule": "latest eligible signal_usable_date must be <= signal_date",
        "missing_rule": "cap or null-impute by training pipeline; missing flag required",
    },
    {
        "feature_name": "has_attention_disposition_history_20d",
        "type": "binary",
        "window": "20 trading days",
        "source_event_types": "attention | disposition",
        "meaning": "Whether the stock recently had any known attention or disposition history.",
        "timing_rule": "rolling window ends on signal_date; no future or post-close rows",
        "missing_rule": "fill 0 when no eligible event exists",
    },
]


def fail(message: str) -> None:
    raise SystemExit(f"FAIL: {message}")


def load_json(path: Path) -> dict:
    if not path.exists():
        fail(f"missing required file: {path.relative_to(PROJECT_ROOT)}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_inputs(config: dict, coverage_decision: dict) -> None:
    allowed = config.get("allowed_inputs", {})
    if set(allowed) != {"stock_daily_all", "market_daily", "theme_group"}:
        fail("feature contract must preserve the three approved model inputs")
    if any("event_risk_calendar" in str(value) for value in allowed.values()):
        fail("event calendar files must not be enabled as approved model inputs in this step")
    if coverage_decision.get("status") != "coverage_ready_for_limited_attention_disposition_features":
        fail("limited attention/disposition coverage is not ready")
    if coverage_decision.get("recommended_next_step") != "prepare_limited_attention_disposition_feature_contract":
        fail("coverage decision does not recommend this feature contract")
    if coverage_decision.get("allowed_scope") != "attention_disposition_only":
        fail("feature contract must stay limited to attention/disposition")
    if coverage_decision.get("complete_event_ready") is not False:
        fail("full event-risk wording is not allowed by the coverage decision")
    if coverage_decision.get("do_not_retrain_yet") is not True:
        fail("coverage decision must still block retraining")


def write_feature_schema() -> None:
    with FEATURE_SCHEMA_PATH.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(FEATURE_ROWS[0].keys()))
        writer.writeheader()
        writer.writerows(FEATURE_ROWS)


def write_feature_contract(coverage_decision: dict) -> None:
    lines = [
        "# Attention / Disposition Feature Contract",
        "",
        "- Status: contract only; not enabled for training.",
        "- Scope: limited attention/disposition features only.",
        "- Formal output: unchanged.",
        "- Model training: not executed.",
        "- Event file remains outside `project_config.json` approved inputs.",
        "",
        "## 白話結論",
        "",
        "歷史回補資料可以先做注意股與處置股的有限風險特徵，但不能寫成完整事件風險，也不能直接拿去重訓。",
        "",
        "## Allowed Source Rows",
        "",
        "- `event_type` must be `attention` or `disposition`.",
        "- `known_before_signal_close` must be true.",
        "- `post_close_pre_next_open` must be false for main-model features.",
        "- `signal_usable_date` must be on or before the signal date.",
        "- Rows beyond the latest available market date are excluded from coverage and cannot be scored as historical evidence.",
        "",
        "## Approved Feature Names",
        "",
        "| feature | type | window | meaning |",
        "|---|---|---|---|",
    ]
    for row in FEATURE_ROWS:
        lines.append(f"| `{row['feature_name']}` | {row['type']} | {row['window']} | {row['meaning']} |")
    lines.extend(
        [
            "",
            "## Forbidden Uses",
            "",
            "- Do not use event titles or source text as NLP signals in this contract.",
            "- Do not use post-close events unless the decision-time contract is changed first.",
            "- Do not use event rows as success or failure labels.",
            "- Do not call these features complete event-risk features.",
            "- Do not add the event file to approved model inputs without a separate red-light approval step.",
            "",
            "## Coverage Basis",
            "",
            f"- Usable historical rows: {coverage_decision['event_rows_total']}",
            f"- Attention rows: {coverage_decision['attention_rows']}",
            f"- Disposition rows: {coverage_decision['disposition_rows']}",
            f"- Other rows: {coverage_decision['other_event_rows']}",
            f"- Dominant source: {coverage_decision['top_source_name']} ({coverage_decision['top_source_share']:.2%})",
            "",
            "## Next Boundary",
            "",
            "下一步若要把這些特徵接進主模型，必須先做 feature generation 防偷看檢查；重訓與正式輸出仍不能在本步驟發生。",
            "",
        ]
    )
    FEATURE_CONTRACT_PATH.write_text("\n".join(lines), encoding="utf-8")


def write_review(coverage_decision: dict, decision: dict) -> None:
    lines = [
        "# Attention / Disposition Feature Contract Review",
        "",
        "- Scope: contract review only.",
        "- Formal output: unchanged.",
        "- Model training: not executed.",
        "- New model input: not enabled.",
        "",
        "## 白話結論",
        "",
        "契約已把事件資料限制在注意/處置用途；它只能描述已知市場警示狀態，不能當作完整事件風險，也不能當成答案標籤。",
        "",
        f"- Status: `{decision['status']}`",
        f"- Recommended next step: `{decision['recommended_next_step']}`",
        f"- Feature rows approved: {decision['approved_feature_count']}",
        f"- Allowed scope: `{decision['allowed_scope']}`",
        f"- Coverage source status: `{coverage_decision['status']}`",
        "",
        "## Approved Feature Families",
        "",
        "- Recent known attention/disposition counts.",
        "- Active attention or disposition status on signal date.",
        "- Days since the latest known attention/disposition row.",
        "- Recent history flag.",
        "",
        "## Still Blocked",
        "",
        "- Full event-risk wording.",
        "- Material information, suspension, resumption, investor meeting, ex-dividend, and corporate-action features.",
        "- Training with the event file before feature generation leakage checks pass.",
        "- Formal candidate output.",
        "",
    ]
    REVIEW_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    config = load_json(CONFIG_PATH)
    coverage_decision = load_json(COVERAGE_DECISION_PATH)
    if not SOURCE_CONTRACT_PATH.exists():
        fail("missing event risk source contract")
    validate_inputs(config, coverage_decision)
    write_feature_schema()
    write_feature_contract(coverage_decision)

    decision = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "limited_attention_disposition_feature_contract_ready",
        "recommended_next_step": "build_attention_disposition_feature_generation_check",
        "allowed_scope": "attention_disposition_only",
        "approved_feature_count": len(FEATURE_ROWS),
        "feature_contract_path": str(FEATURE_CONTRACT_PATH.relative_to(PROJECT_ROOT)),
        "feature_schema_path": str(FEATURE_SCHEMA_PATH.relative_to(PROJECT_ROOT)),
        "new_input_not_enabled": True,
        "do_not_retrain_yet": True,
        "formal_outputs_unchanged": True,
        "requires_red_light_before_model_input": True,
    }
    DECISION_PATH.write_text(json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_review(coverage_decision, decision)

    print("OK: attention/disposition feature contract completed")
    print(f"STATUS: {decision['status']}")
    print(f"NEXT_STEP: {decision['recommended_next_step']}")
    print(f"CONTRACT: {FEATURE_CONTRACT_PATH}")


if __name__ == "__main__":
    main()
