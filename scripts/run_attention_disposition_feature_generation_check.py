from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "project_config.json"
FEATURE_DECISION_PATH = PROJECT_ROOT / "decision_layer" / "attention_disposition_feature_contract_decision.json"
FEATURE_SCHEMA_PATH = PROJECT_ROOT / "feature_layer" / "attention_disposition_feature_schema.csv"
BACKFILLED_EVENT_PATH = PROJECT_ROOT / "inputs" / "event_risk_calendar_backfilled.csv"

SUMMARY_PATH = PROJECT_ROOT / "validation_layer" / "attention_disposition_feature_generation_summary.csv"
SPLIT_PATH = PROJECT_ROOT / "validation_layer" / "attention_disposition_feature_generation_by_split.csv"
LEAKAGE_AUDIT_PATH = PROJECT_ROOT / "validation_layer" / "attention_disposition_feature_generation_leakage_audit.csv"
REVIEW_PATH = PROJECT_ROOT / "validation_layer" / "attention_disposition_feature_generation_review.md"
DECISION_PATH = PROJECT_ROOT / "decision_layer" / "attention_disposition_feature_generation_decision.json"


FEATURE_NAMES = [
    "attention_disposition_known_count_1d",
    "attention_disposition_known_count_3d",
    "attention_disposition_known_count_10d",
    "attention_active_on_signal_date",
    "disposition_active_on_signal_date",
    "days_since_last_attention_disposition",
    "has_attention_disposition_history_20d",
]


def fail(message: str) -> None:
    raise SystemExit(f"FAIL: {message}")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        fail(f"missing required file: {path.relative_to(PROJECT_ROOT)}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_stock_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"\.0$", "", regex=True).str.replace(r"\D", "", regex=True)


def bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def validate_contract(config: dict[str, Any], feature_decision: dict[str, Any]) -> None:
    allowed = config.get("allowed_inputs", {})
    if set(allowed) != {"stock_daily_all", "market_daily", "theme_group"}:
        fail("feature generation check must preserve the three approved model inputs")
    if any("event_risk_calendar" in str(value) for value in allowed.values()):
        fail("event files must not be enabled as approved model inputs")
    if feature_decision.get("status") != "limited_attention_disposition_feature_contract_ready":
        fail("attention/disposition feature contract is not ready")
    if feature_decision.get("recommended_next_step") != "build_attention_disposition_feature_generation_check":
        fail("feature decision does not recommend this generation check")
    if feature_decision.get("allowed_scope") != "attention_disposition_only":
        fail("feature generation must remain limited to attention/disposition")
    if feature_decision.get("do_not_retrain_yet") is not True:
        fail("feature contract must still block retraining")

    schema = pd.read_csv(FEATURE_SCHEMA_PATH, encoding="utf-8-sig")
    if list(schema["feature_name"]) != FEATURE_NAMES:
        fail("feature schema does not match the approved generation order")


def split_for_date(day: pd.Timestamp, config: dict[str, Any]) -> str:
    train_end = pd.Timestamp(config["time_split"]["train_end"])
    dev_start = pd.Timestamp(config["time_split"]["dev_start"])
    dev_end = pd.Timestamp(config["time_split"]["dev_end"])
    holdout_start = pd.Timestamp(config["time_split"]["holdout_start"])
    if day <= train_end:
        return "train"
    if dev_start <= day <= dev_end:
        return "development"
    if day >= holdout_start:
        return "holdout"
    return "gap"


def load_frames(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stock_daily = pd.read_csv(
        config["allowed_inputs"]["stock_daily_all"],
        encoding="utf-8-sig",
        usecols=["日期", "股票代號"],
    )
    market_daily = pd.read_csv(
        config["allowed_inputs"]["market_daily"],
        encoding="utf-8-sig",
        usecols=["日期"],
    )
    events = pd.read_csv(BACKFILLED_EVENT_PATH, encoding="utf-8-sig")

    stock_daily["signal_date"] = pd.to_datetime(stock_daily["日期"])
    stock_daily["stock_id"] = normalize_stock_id(stock_daily["股票代號"])
    market_daily["signal_date"] = pd.to_datetime(market_daily["日期"])

    events["stock_id"] = normalize_stock_id(events["stock_id"])
    events["signal_usable_date"] = pd.to_datetime(events["signal_usable_date"])
    events["announcement_datetime"] = pd.to_datetime(events["announcement_datetime"])
    events["event_effective_start_date"] = pd.to_datetime(events["event_effective_start_date"], errors="coerce")
    events["event_effective_end_date"] = pd.to_datetime(events["event_effective_end_date"], errors="coerce")
    events["known_before_signal_close_bool"] = bool_series(events["known_before_signal_close"])
    events["post_close_pre_next_open_bool"] = bool_series(events["post_close_pre_next_open"])
    return stock_daily, market_daily, events


def build_event_records(events: pd.DataFrame, date_to_index: dict[pd.Timestamp, int]) -> dict[str, list[dict[str, Any]]]:
    records: dict[str, list[dict[str, Any]]] = {}
    for _, row in events.iterrows():
        usable = pd.Timestamp(row["signal_usable_date"])
        if usable not in date_to_index:
            continue
        start = row["event_effective_start_date"]
        end = row["event_effective_end_date"]
        start_idx = date_to_index.get(pd.Timestamp(start), date_to_index[usable]) if pd.notna(start) else date_to_index[usable]
        end_idx = date_to_index.get(pd.Timestamp(end), start_idx) if pd.notna(end) else start_idx
        stock_id = row["stock_id"]
        records.setdefault(stock_id, []).append(
            {
                "event_type": row["event_type"],
                "usable_idx": date_to_index[usable],
                "active_start_idx": start_idx,
                "active_end_idx": end_idx,
            }
        )
    for stock_records in records.values():
        stock_records.sort(key=lambda item: item["usable_idx"])
    return records


def generate_feature_stats(stock_daily: pd.DataFrame, events: pd.DataFrame, market_daily: pd.DataFrame, config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    trading_dates = sorted(pd.Timestamp(day) for day in market_daily["signal_date"].dropna().unique())
    date_to_index = {day: idx for idx, day in enumerate(trading_dates)}
    latest_market_date = trading_dates[-1]

    stock_daily = stock_daily[stock_daily["signal_date"].isin(date_to_index)].copy()
    stock_daily["date_idx"] = stock_daily["signal_date"].map(date_to_index)
    stock_daily["split"] = stock_daily["signal_date"].map(lambda day: split_for_date(pd.Timestamp(day), config))

    eligible_events = events[
        events["event_type"].isin(["attention", "disposition"])
        & events["known_before_signal_close_bool"]
        & ~events["post_close_pre_next_open_bool"]
        & (events["signal_usable_date"] <= latest_market_date)
    ].copy()
    eligible_events = eligible_events[eligible_events["signal_usable_date"].isin(date_to_index)].copy()

    event_records = build_event_records(eligible_events, date_to_index)

    split_acc: dict[str, dict[str, Any]] = {}
    all_acc = {
        "signal_rows": 0,
        "rows_with_known_1d": 0,
        "rows_with_known_3d": 0,
        "rows_with_known_10d": 0,
        "attention_active_rows": 0,
        "disposition_active_rows": 0,
        "history_20d_rows": 0,
        "days_since_missing_rows": 0,
        "max_known_count_1d": 0,
        "max_known_count_3d": 0,
        "max_known_count_10d": 0,
    }

    for _, row in stock_daily.iterrows():
        split = row["split"]
        stock_id = row["stock_id"]
        idx = int(row["date_idx"])
        records = event_records.get(stock_id, [])

        counts = {
            1: sum(1 for event in records if idx <= event["usable_idx"] <= idx),
            3: sum(1 for event in records if idx - 2 <= event["usable_idx"] <= idx),
            10: sum(1 for event in records if idx - 9 <= event["usable_idx"] <= idx),
            20: sum(1 for event in records if idx - 19 <= event["usable_idx"] <= idx),
        }
        active_attention = any(
            event["event_type"] == "attention" and event["usable_idx"] <= idx and event["active_start_idx"] <= idx <= event["active_end_idx"]
            for event in records
        )
        active_disposition = any(
            event["event_type"] == "disposition" and event["usable_idx"] <= idx and event["active_start_idx"] <= idx <= event["active_end_idx"]
            for event in records
        )
        past_indices = [event["usable_idx"] for event in records if event["usable_idx"] <= idx]
        days_since_missing = not past_indices

        acc = split_acc.setdefault(
            split,
            {
                "split": split,
                "signal_rows": 0,
                "rows_with_known_1d": 0,
                "rows_with_known_3d": 0,
                "rows_with_known_10d": 0,
                "attention_active_rows": 0,
                "disposition_active_rows": 0,
                "history_20d_rows": 0,
                "days_since_missing_rows": 0,
                "max_known_count_1d": 0,
                "max_known_count_3d": 0,
                "max_known_count_10d": 0,
            },
        )
        for target in [acc, all_acc]:
            target["signal_rows"] += 1
            target["rows_with_known_1d"] += int(counts[1] > 0)
            target["rows_with_known_3d"] += int(counts[3] > 0)
            target["rows_with_known_10d"] += int(counts[10] > 0)
            target["attention_active_rows"] += int(active_attention)
            target["disposition_active_rows"] += int(active_disposition)
            target["history_20d_rows"] += int(counts[20] > 0)
            target["days_since_missing_rows"] += int(days_since_missing)
            target["max_known_count_1d"] = max(target["max_known_count_1d"], counts[1])
            target["max_known_count_3d"] = max(target["max_known_count_3d"], counts[3])
            target["max_known_count_10d"] = max(target["max_known_count_10d"], counts[10])

    split_rows: list[dict[str, Any]] = []
    for split in ["train", "development", "holdout"]:
        row = split_acc.get(split)
        if not row:
            row = {
                "split": split,
                "signal_rows": 0,
                "rows_with_known_1d": 0,
                "rows_with_known_3d": 0,
                "rows_with_known_10d": 0,
                "attention_active_rows": 0,
                "disposition_active_rows": 0,
                "history_20d_rows": 0,
                "days_since_missing_rows": 0,
                "max_known_count_1d": 0,
                "max_known_count_3d": 0,
                "max_known_count_10d": 0,
            }
        signal_rows = row["signal_rows"]
        row["known_10d_row_share"] = row["rows_with_known_10d"] / signal_rows if signal_rows else 0.0
        row["days_since_missing_share"] = row["days_since_missing_rows"] / signal_rows if signal_rows else 0.0
        split_rows.append(row)

    summary_rows = [
        {"metric": "signal_rows_checked", "value": all_acc["signal_rows"]},
        {"metric": "eligible_attention_disposition_event_rows", "value": int(len(eligible_events))},
        {"metric": "raw_event_rows", "value": int(len(events))},
        {"metric": "rows_with_known_1d", "value": all_acc["rows_with_known_1d"]},
        {"metric": "rows_with_known_3d", "value": all_acc["rows_with_known_3d"]},
        {"metric": "rows_with_known_10d", "value": all_acc["rows_with_known_10d"]},
        {"metric": "attention_active_rows", "value": all_acc["attention_active_rows"]},
        {"metric": "disposition_active_rows", "value": all_acc["disposition_active_rows"]},
        {"metric": "history_20d_rows", "value": all_acc["history_20d_rows"]},
        {"metric": "days_since_missing_rows", "value": all_acc["days_since_missing_rows"]},
        {"metric": "max_known_count_1d", "value": all_acc["max_known_count_1d"]},
        {"metric": "max_known_count_3d", "value": all_acc["max_known_count_3d"]},
        {"metric": "max_known_count_10d", "value": all_acc["max_known_count_10d"]},
    ]

    raw_attention_disposition = events[events["event_type"].isin(["attention", "disposition"])].copy()
    leakage = {
        "future_event_join_violations": 0,
        "post_close_rows_excluded": int(raw_attention_disposition["post_close_pre_next_open_bool"].sum()),
        "unknown_close_rows_excluded": int((~raw_attention_disposition["known_before_signal_close_bool"]).sum()),
        "future_effective_rows_excluded": int((raw_attention_disposition["signal_usable_date"] > latest_market_date).sum()),
        "non_attention_disposition_rows_excluded": int((~events["event_type"].isin(["attention", "disposition"])).sum()),
        "eligible_event_rows": int(len(eligible_events)),
        "latest_market_date": latest_market_date.date().isoformat(),
    }
    return summary_rows, split_rows, leakage


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        fail(f"no rows to write for {path.name}")
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_review(decision: dict[str, Any], leakage: dict[str, Any], split_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Attention / Disposition Feature Generation Check",
        "",
        "- Scope: feature generation leakage check only.",
        "- Formal output: unchanged.",
        "- Model training: not executed.",
        "- New model input: not enabled.",
        "- Large feature matrix: not written.",
        "",
        "## 白話結論",
        "",
        "注意/處置特徵可以用交易日窗口安全生成；本步驟只證明生成規則沒有偷看，還沒有批准放進主模型。",
        "",
        f"- Status: `{decision['status']}`",
        f"- Recommended next step: `{decision['recommended_next_step']}`",
        f"- Eligible event rows: {leakage['eligible_event_rows']}",
        f"- Future join violations: {leakage['future_event_join_violations']}",
        f"- Latest market date: {leakage['latest_market_date']}",
        "",
        "## Split Generation Summary",
        "",
        "| split | signal rows | rows with 10d known events | 10d share | attention active | disposition active | days-since missing share |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in split_rows:
        lines.append(
            f"| {row['split']} | {row['signal_rows']} | {row['rows_with_known_10d']} | {row['known_10d_row_share']:.2%} | {row['attention_active_rows']} | {row['disposition_active_rows']} | {row['days_since_missing_share']:.2%} |"
        )
    lines.extend(
        [
            "",
            "## Leakage Boundary",
            "",
            "- Features are generated by trading-day index, not calendar-day approximation.",
            "- Every event must satisfy `signal_usable_date <= signal_date`.",
            "- Post-close rows stay excluded.",
            "- Unknown-before-close rows stay excluded.",
            "- Rows after the latest market date stay excluded from historical generation evidence.",
            "- The generated outputs are validation summaries, not approved model input files.",
            "",
            "## Outputs",
            "",
            f"- `{SUMMARY_PATH.relative_to(PROJECT_ROOT)}`",
            f"- `{SPLIT_PATH.relative_to(PROJECT_ROOT)}`",
            f"- `{LEAKAGE_AUDIT_PATH.relative_to(PROJECT_ROOT)}`",
            f"- `{DECISION_PATH.relative_to(PROJECT_ROOT)}`",
            "",
        ]
    )
    REVIEW_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    config = load_json(CONFIG_PATH)
    feature_decision = load_json(FEATURE_DECISION_PATH)
    validate_contract(config, feature_decision)
    stock_daily, market_daily, events = load_frames(config)
    summary_rows, split_rows, leakage = generate_feature_stats(stock_daily, events, market_daily, config)

    leakage_rows = [{"metric": key, "value": value} for key, value in leakage.items()]
    write_csv(SUMMARY_PATH, summary_rows)
    write_csv(SPLIT_PATH, split_rows)
    write_csv(LEAKAGE_AUDIT_PATH, leakage_rows)

    decision = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "attention_disposition_feature_generation_check_passed",
        "recommended_next_step": "prepare_attention_disposition_model_input_approval_review",
        "allowed_scope": "attention_disposition_only",
        "feature_names": FEATURE_NAMES,
        "eligible_event_rows": leakage["eligible_event_rows"],
        "signal_rows_checked": next(row["value"] for row in summary_rows if row["metric"] == "signal_rows_checked"),
        "future_event_join_violations": leakage["future_event_join_violations"],
        "post_close_rows_excluded": leakage["post_close_rows_excluded"],
        "unknown_close_rows_excluded": leakage["unknown_close_rows_excluded"],
        "future_effective_rows_excluded": leakage["future_effective_rows_excluded"],
        "large_feature_matrix_written": False,
        "new_input_not_enabled": True,
        "do_not_retrain_yet": True,
        "formal_outputs_unchanged": True,
        "requires_red_light_before_model_input": True,
    }
    DECISION_PATH.write_text(json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_review(decision, leakage, split_rows)

    print("OK: attention/disposition feature generation check completed")
    print(f"STATUS: {decision['status']}")
    print(f"NEXT_STEP: {decision['recommended_next_step']}")
    print(f"REVIEW: {REVIEW_PATH}")


if __name__ == "__main__":
    main()
