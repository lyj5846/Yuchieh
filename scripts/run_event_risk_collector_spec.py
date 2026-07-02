from __future__ import annotations

import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "project_config.json"
DATA_ENRICHMENT_DECISION_PATH = PROJECT_ROOT / "decision_layer" / "data_enrichment_decision.json"

DATA_DIR = PROJECT_ROOT / "data_layer"
VALIDATION_DIR = PROJECT_ROOT / "validation_layer"
DECISION_DIR = PROJECT_ROOT / "decision_layer"

CONTRACT_PATH = DATA_DIR / "event_risk_calendar_source_contract.md"
SCHEMA_PATH = DATA_DIR / "event_risk_calendar_schema.csv"
REVIEW_PATH = VALIDATION_DIR / "event_risk_collector_spec_review.md"
DECISION_PATH = DECISION_DIR / "event_risk_collector_spec_decision.json"


def fail(message: str) -> None:
    raise SystemExit(f"FAIL: {message}")


def read_json(path: Path) -> dict:
    if not path.exists():
        fail(f"missing required file: {path.relative_to(PROJECT_ROOT)}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_project_inputs(config: dict) -> None:
    allowed = config.get("allowed_inputs", {})
    if set(allowed) != {"stock_daily_all", "market_daily", "theme_group"}:
        fail("project must still allow exactly the three approved input files")
    old_marker = "stock" + "_raw_only" + "_project"
    for name, value in allowed.items():
        path = Path(value)
        if old_marker in str(path):
            fail(f"old project path is not allowed: {name}")
        if not path.exists():
            fail(f"missing approved input {name}: {path}")


def validate_previous_decision(decision: dict) -> None:
    if decision.get("status") != "data_gap_confirmed":
        fail("data enrichment decision must confirm the data gap before this spec")
    if decision.get("recommended_next_step") != "build_event_risk_data_collector_spec":
        fail("data enrichment decision must recommend this collector spec")
    if decision.get("recommended_data_need_id") != "event_risk_calendar":
        fail("collector spec is only valid for event_risk_calendar")
    if decision.get("do_not_retrain_yet") is not True:
        fail("collector spec must be created before any retraining")


def schema_rows() -> list[dict[str, str]]:
    rows = [
        {
            "field_name": "stock_id",
            "field_type": "string",
            "required": "yes",
            "description": "Taiwan stock code, normalized to text.",
            "example": "2449",
            "no_leakage_rule": "Identifier only; no timing risk.",
        },
        {
            "field_name": "stock_name",
            "field_type": "string",
            "required": "yes",
            "description": "Stock name from the event source or mapped from the approved theme file.",
            "example": "京元電子",
            "no_leakage_rule": "Name only; no timing risk.",
        },
        {
            "field_name": "event_type",
            "field_type": "enum",
            "required": "yes",
            "description": "Mechanical event class: material_info, suspension, resumption, attention, disposition, investor_meeting, ex_dividend, corporate_action.",
            "example": "attention",
            "no_leakage_rule": "Use source category as published; do not rewrite it as a human opinion.",
        },
        {
            "field_name": "event_subtype",
            "field_type": "string",
            "required": "no",
            "description": "More detailed source subtype when available.",
            "example": "處置期間延長",
            "no_leakage_rule": "Use only the published subtype visible at the source timestamp.",
        },
        {
            "field_name": "event_title",
            "field_type": "string",
            "required": "yes",
            "description": "Original event title or brief source text.",
            "example": "公告本公司召開法說會",
            "no_leakage_rule": "Store the source text; do not summarize with future price knowledge.",
        },
        {
            "field_name": "source_name",
            "field_type": "string",
            "required": "yes",
            "description": "Source family name, such as exchange notice, public disclosure, or company event feed.",
            "example": "TWSE attention notice",
            "no_leakage_rule": "Source metadata only.",
        },
        {
            "field_name": "source_url",
            "field_type": "string",
            "required": "no",
            "description": "Source URL or stable reference when available.",
            "example": "https://example.invalid/source",
            "no_leakage_rule": "URL must point to the source item, not to a later news explanation.",
        },
        {
            "field_name": "announcement_datetime",
            "field_type": "datetime",
            "required": "yes",
            "description": "The first public timestamp when the event was known.",
            "example": "2026-06-15 13:20:00",
            "no_leakage_rule": "This is the main anti-leakage field; missing timestamps cannot enter training.",
        },
        {
            "field_name": "event_effective_start_date",
            "field_type": "date",
            "required": "no",
            "description": "Date when the event starts to apply, such as disposition start date.",
            "example": "2026-06-16",
            "no_leakage_rule": "Can be later than announcement date, but feature availability is still controlled by announcement_datetime.",
        },
        {
            "field_name": "event_effective_end_date",
            "field_type": "date",
            "required": "no",
            "description": "Date when the event stops applying.",
            "example": "2026-06-24",
            "no_leakage_rule": "Can only be used after the original announcement is known.",
        },
        {
            "field_name": "signal_usable_date",
            "field_type": "date",
            "required": "yes",
            "description": "First signal date on which this event can be used under the strict close-time rule.",
            "example": "2026-06-15",
            "no_leakage_rule": "If announced after signal close, usable date moves to the next trading day unless the decision-time contract is changed.",
        },
        {
            "field_name": "known_before_signal_close",
            "field_type": "boolean",
            "required": "yes",
            "description": "Whether the event was publicly known by the signal day's close.",
            "example": "true",
            "no_leakage_rule": "Only true rows may become main model features under the current contract.",
        },
        {
            "field_name": "post_close_pre_next_open",
            "field_type": "boolean",
            "required": "yes",
            "description": "Whether the event was known after close but before next open.",
            "example": "false",
            "no_leakage_rule": "Store separately; do not use in the main model unless the decision-time contract is explicitly revised.",
        },
        {
            "field_name": "raw_payload_hash",
            "field_type": "string",
            "required": "no",
            "description": "Hash of raw downloaded payload for audit and de-duplication.",
            "example": "sha256:...",
            "no_leakage_rule": "Audit only; not a model feature.",
        },
    ]
    return rows


def feature_contract_rows() -> list[dict[str, str]]:
    return [
        {
            "feature_name": "event_any_known_1d",
            "window": "1 trading day",
            "source_fields": "event_type | announcement_datetime | signal_usable_date",
            "meaning": "Any known event near the signal date.",
            "use_rule": "Count only events with signal_usable_date <= signal_date.",
        },
        {
            "feature_name": "event_any_known_3d",
            "window": "3 trading days",
            "source_fields": "event_type | announcement_datetime | signal_usable_date",
            "meaning": "Recent event density before the signal.",
            "use_rule": "Rolling count known by signal close.",
        },
        {
            "feature_name": "event_attention_or_disposition_active",
            "window": "active date range",
            "source_fields": "event_type | event_effective_start_date | event_effective_end_date | signal_usable_date",
            "meaning": "Whether attention or disposition status is active and already known.",
            "use_rule": "Active on signal date and published before the usable cutoff.",
        },
        {
            "feature_name": "event_suspension_or_resumption_nearby",
            "window": "10 trading days",
            "source_fields": "event_type | event_effective_start_date | event_effective_end_date | signal_usable_date",
            "meaning": "Whether trading halt or restart event is close enough to affect price behavior.",
            "use_rule": "Use only after source announcement is known.",
        },
        {
            "feature_name": "event_material_info_known_5d",
            "window": "5 trading days",
            "source_fields": "event_type | event_title | signal_usable_date",
            "meaning": "Material information event count before the signal.",
            "use_rule": "No NLP judgement in the first collector; count and type only.",
        },
        {
            "feature_name": "event_ex_dividend_or_corporate_action_10d",
            "window": "10 trading days",
            "source_fields": "event_type | event_effective_start_date | signal_usable_date",
            "meaning": "Known corporate action that may distort price path.",
            "use_rule": "Use announced events only; no later adjustment backfill.",
        },
    ]


def source_catalog_rows() -> list[dict[str, str]]:
    return [
        {
            "source_family": "public_disclosure_material_info",
            "event_types": "material_info, investor_meeting, corporate_action",
            "required_timestamp": "announcement_datetime",
            "collector_note": "Use the first published timestamp and retain source title.",
            "training_use": "allowed only after signal_usable_date is reached",
        },
        {
            "source_family": "exchange_attention_disposition",
            "event_types": "attention, disposition",
            "required_timestamp": "announcement_datetime",
            "collector_note": "Capture start and end dates when provided.",
            "training_use": "allowed only after signal_usable_date is reached",
        },
        {
            "source_family": "exchange_suspension_resumption",
            "event_types": "suspension, resumption",
            "required_timestamp": "announcement_datetime",
            "collector_note": "Capture effective trading date and reason text.",
            "training_use": "allowed only after signal_usable_date is reached",
        },
        {
            "source_family": "corporate_action_calendar",
            "event_types": "ex_dividend, corporate_action",
            "required_timestamp": "announcement_datetime",
            "collector_note": "Capture announced date and effective date separately.",
            "training_use": "allowed only after signal_usable_date is reached",
        },
    ]


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        fail(f"no rows to write for {path.name}")
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_contract(decision: dict) -> None:
    schema = schema_rows()
    features = feature_contract_rows()
    sources = source_catalog_rows()
    lines = [
        "# Event Risk Calendar Source Contract",
        "",
        "- Status: specification only; no new data source is enabled yet.",
        "- Purpose: define the event-risk data needed before the next model retraining attempt.",
        "- Current formal output: unchanged.",
        f"- Source decision: `{DATA_ENRICHMENT_DECISION_PATH.relative_to(PROJECT_ROOT)}`",
        "",
        "## Why This Exists",
        "",
        decision["reason"],
        "",
        "The current three CSV files cover price, volume, chip, market state, and static industry groups. They do not show whether a stock had a known event that could explain a first adverse move before the +3% target is reached.",
        "",
        "## Proposed Future File",
        "",
        "- Future file path: `inputs/event_risk_calendar.csv`",
        "- This file is not added to `project_config.json` in this step.",
        "- A collector must be reviewed before this file can become an approved model input.",
        "",
        "## Strict Timing Rule",
        "",
        "- Main model features may only use events already public by signal-day close.",
        "- Events published after signal-day close but before next open must be stored separately.",
        "- Post-close events cannot enter the main model unless the decision-time contract is explicitly changed.",
        "- Missing `announcement_datetime` means the row is not eligible for training.",
        "- Later news explanations or human interpretations cannot be backfilled into earlier signal dates.",
        "",
        "## Raw Event Schema",
        "",
        "| field | type | required | description |",
        "|---|---|---|---|",
    ]
    for row in schema:
        lines.append(
            f"| {row['field_name']} | {row['field_type']} | {row['required']} | {row['description']} |"
        )
    lines.extend(
        [
            "",
            "## Source Families",
            "",
            "| source family | event types | required timestamp | collector note |",
            "|---|---|---|---|",
        ]
    )
    for row in sources:
        lines.append(
            f"| {row['source_family']} | {row['event_types']} | {row['required_timestamp']} | {row['collector_note']} |"
        )
    lines.extend(
        [
            "",
            "## Future Feature Contract",
            "",
            "| feature | window | meaning | use rule |",
            "|---|---|---|---|",
        ]
    )
    for row in features:
        lines.append(
            f"| {row['feature_name']} | {row['window']} | {row['meaning']} | {row['use_rule']} |"
        )
    lines.extend(
        [
            "",
            "## Boundaries",
            "",
            "- This step does not fetch data.",
            "- This step does not train or promote a model.",
            "- This step does not choose stocks.",
            "- This step does not change the three approved CSV inputs.",
            "- Event categories are mechanical source facts, not manual success or failure explanations.",
            "",
        ]
    )
    CONTRACT_PATH.write_text("\n".join(lines), encoding="utf-8")


def write_review(decision: dict) -> None:
    lines = [
        "# Event Risk Collector Spec Review",
        "",
        "- Scope: collector specification only.",
        "- Formal output: unchanged.",
        "- Model training: not executed.",
        "- New input source: not enabled.",
        "",
        "## 白話結論",
        "",
        "下一步不是重訓，而是先把事件風險資料定義乾淨。若沒有 `announcement_datetime` 與 `signal_usable_date`，這類資料很容易偷看未來，因此先做規格比直接寫抓取程式更重要。",
        "",
        "## Required Future Data",
        "",
        "- 重大訊息",
        "- 停牌與復牌",
        "- 注意股與處置股",
        "- 法說會",
        "- 除權息與重大公司事件",
        "",
        "## Anti-Leakage Decision",
        "",
        "- Main model can use only rows known by signal-day close.",
        "- Post-close rows are stored but blocked from the main model until the decision-time contract changes.",
        "- Missing public timestamp means unusable for training.",
        "",
        "## Decision",
        "",
        "- Status: `collector_spec_ready`",
        "- Recommended next step: `implement_event_risk_calendar_collector`",
        "- Do not retrain yet: `true`",
        "",
        "## Outputs",
        "",
        f"- `{CONTRACT_PATH.relative_to(PROJECT_ROOT)}`",
        f"- `{SCHEMA_PATH.relative_to(PROJECT_ROOT)}`",
        f"- `{DECISION_PATH.relative_to(PROJECT_ROOT)}`",
        "",
    ]
    REVIEW_PATH.write_text("\n".join(lines), encoding="utf-8")


def write_decision(previous_decision: dict) -> None:
    payload = {
        "status": "collector_spec_ready",
        "recommended_next_step": "implement_event_risk_calendar_collector",
        "data_need_id": "event_risk_calendar",
        "proposed_future_input": "inputs/event_risk_calendar.csv",
        "source_decision_status": previous_decision.get("status"),
        "source_decision_next_step": previous_decision.get("recommended_next_step"),
        "do_not_retrain_yet": True,
        "formal_outputs_unchanged": True,
        "required_timestamp_fields": [
            "announcement_datetime",
            "signal_usable_date",
            "known_before_signal_close",
            "post_close_pre_next_open",
        ],
        "required_event_types": [
            "material_info",
            "suspension",
            "resumption",
            "attention",
            "disposition",
            "investor_meeting",
            "ex_dividend",
            "corporate_action",
        ],
    }
    DECISION_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    config = read_json(CONFIG_PATH)
    validate_project_inputs(config)
    previous_decision = read_json(DATA_ENRICHMENT_DECISION_PATH)
    validate_previous_decision(previous_decision)

    write_csv(SCHEMA_PATH, schema_rows())
    write_contract(previous_decision)
    write_review(previous_decision)
    write_decision(previous_decision)

    print("OK: event risk collector specification completed")
    print(f"STATUS: collector_spec_ready")
    print(f"NEXT_STEP: implement_event_risk_calendar_collector")
    print(f"CONTRACT: {CONTRACT_PATH}")


if __name__ == "__main__":
    main()
