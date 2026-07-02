from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "project_config.json"
BACKFILL_DECISION_PATH = PROJECT_ROOT / "decision_layer" / "historical_event_risk_backfill_decision.json"
BACKFILLED_EVENT_PATH = PROJECT_ROOT / "inputs" / "event_risk_calendar_backfilled.csv"

REVIEW_PATH = PROJECT_ROOT / "validation_layer" / "historical_event_risk_backfill_coverage_review.md"
SPLIT_COVERAGE_PATH = PROJECT_ROOT / "validation_layer" / "historical_event_risk_backfill_coverage_by_split.csv"
TYPE_COVERAGE_PATH = PROJECT_ROOT / "validation_layer" / "historical_event_risk_backfill_coverage_by_type.csv"
SOURCE_COVERAGE_PATH = PROJECT_ROOT / "validation_layer" / "historical_event_risk_backfill_coverage_by_source.csv"
MONTH_COVERAGE_PATH = PROJECT_ROOT / "validation_layer" / "historical_event_risk_backfill_coverage_by_month.csv"
DECISION_PATH = PROJECT_ROOT / "decision_layer" / "historical_event_risk_backfill_coverage_decision.json"


def fail(message: str) -> None:
    raise SystemExit(f"FAIL: {message}")


def load_json(path: Path) -> dict:
    if not path.exists():
        fail(f"missing required file: {path.relative_to(PROJECT_ROOT)}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_inputs(config: dict, backfill_decision: dict) -> None:
    allowed = config.get("allowed_inputs", {})
    if set(allowed) != {"stock_daily_all", "market_daily", "theme_group"}:
        fail("coverage review must preserve the three approved model inputs")
    if any("event_risk_calendar" in str(value) for value in allowed.values()):
        fail("event calendar files must not be enabled as model inputs before feature approval")
    if backfill_decision.get("status") != "historical_backfill_output_ready":
        fail("historical backfill output must be ready before coverage review")
    if backfill_decision.get("recommended_next_step") != "review_historical_event_risk_backfill_coverage":
        fail("historical backfill decision must recommend this coverage review")
    if backfill_decision.get("do_not_retrain_yet") is not True:
        fail("coverage review must happen before retraining")


def split_for_date(day: pd.Timestamp, config: dict) -> str:
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


def load_frames(config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stock_daily = pd.read_csv(config["allowed_inputs"]["stock_daily_all"], encoding="utf-8-sig", usecols=["日期", "股票代號"])
    market_daily = pd.read_csv(config["allowed_inputs"]["market_daily"], encoding="utf-8-sig", usecols=["日期"])
    event_calendar = pd.read_csv(BACKFILLED_EVENT_PATH, encoding="utf-8-sig")

    stock_daily["日期"] = pd.to_datetime(stock_daily["日期"])
    market_daily["日期"] = pd.to_datetime(market_daily["日期"])
    stock_daily["股票代號"] = stock_daily["股票代號"].astype(str).str.replace(r"\.0$", "", regex=True).str.replace(r"\D", "", regex=True)
    event_calendar["stock_id"] = event_calendar["stock_id"].astype(str).str.replace(r"\D", "", regex=True)
    event_calendar["signal_usable_date"] = pd.to_datetime(event_calendar["signal_usable_date"])
    event_calendar["announcement_datetime"] = pd.to_datetime(event_calendar["announcement_datetime"])
    event_calendar["split"] = event_calendar["signal_usable_date"].map(lambda day: split_for_date(day, config))
    return stock_daily, market_daily, event_calendar


def split_rows(stock_daily: pd.DataFrame, market_daily: pd.DataFrame, events: pd.DataFrame, config: dict) -> list[dict]:
    rows: list[dict] = []
    stock_universe = int(stock_daily["股票代號"].nunique())
    for split in ["train", "development", "holdout"]:
        market_split = market_daily[market_daily["日期"].map(lambda day: split_for_date(day, config) == split)]
        event_split = events[events["split"] == split]
        trading_days = int(market_split["日期"].nunique())
        event_days = int(event_split["signal_usable_date"].dt.date.nunique()) if not event_split.empty else 0
        event_stocks = int(event_split["stock_id"].nunique()) if not event_split.empty else 0
        rows.append(
            {
                "split": split,
                "start_date": market_split["日期"].min().date().isoformat() if not market_split.empty else "",
                "end_date": market_split["日期"].max().date().isoformat() if not market_split.empty else "",
                "trading_days": trading_days,
                "event_rows": int(len(event_split)),
                "event_trading_days": event_days,
                "event_day_coverage_rate": event_days / trading_days if trading_days else 0.0,
                "event_stock_count": event_stocks,
                "event_stock_coverage_rate": event_stocks / stock_universe if stock_universe else 0.0,
                "attention_rows": int((event_split["event_type"] == "attention").sum()) if not event_split.empty else 0,
                "disposition_rows": int((event_split["event_type"] == "disposition").sum()) if not event_split.empty else 0,
                "other_event_rows": int((~event_split["event_type"].isin(["attention", "disposition"])).sum()) if not event_split.empty else 0,
            }
        )
    return rows


def type_rows(events: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    total = len(events)
    for event_type, group in events.groupby("event_type"):
        rows.append(
            {
                "event_type": event_type,
                "event_rows": int(len(group)),
                "event_row_share": len(group) / total if total else 0.0,
                "event_stock_count": int(group["stock_id"].nunique()),
                "first_signal_usable_date": group["signal_usable_date"].min().date().isoformat(),
                "last_signal_usable_date": group["signal_usable_date"].max().date().isoformat(),
            }
        )
    return sorted(rows, key=lambda row: row["event_rows"], reverse=True)


def source_rows(events: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    total = len(events)
    for source_name, group in events.groupby("source_name"):
        rows.append(
            {
                "source_name": source_name,
                "event_rows": int(len(group)),
                "event_row_share": len(group) / total if total else 0.0,
                "event_stock_count": int(group["stock_id"].nunique()),
                "event_types": " | ".join(sorted(group["event_type"].dropna().unique())),
                "first_signal_usable_date": group["signal_usable_date"].min().date().isoformat(),
                "last_signal_usable_date": group["signal_usable_date"].max().date().isoformat(),
            }
        )
    return sorted(rows, key=lambda row: row["event_rows"], reverse=True)


def month_rows(market_daily: pd.DataFrame, events: pd.DataFrame) -> list[dict]:
    market = market_daily.copy()
    market["month"] = market["日期"].dt.strftime("%Y-%m")
    events = events.copy()
    events["month"] = events["signal_usable_date"].dt.strftime("%Y-%m")
    rows: list[dict] = []
    for month, market_group in market.groupby("month"):
        event_group = events[events["month"] == month]
        rows.append(
            {
                "month": month,
                "trading_days": int(market_group["日期"].nunique()),
                "event_rows": int(len(event_group)),
                "event_days": int(event_group["signal_usable_date"].dt.date.nunique()) if not event_group.empty else 0,
                "attention_rows": int((event_group["event_type"] == "attention").sum()) if not event_group.empty else 0,
                "disposition_rows": int((event_group["event_type"] == "disposition").sum()) if not event_group.empty else 0,
                "other_event_rows": int((~event_group["event_type"].isin(["attention", "disposition"])).sum()) if not event_group.empty else 0,
            }
        )
    return rows


def decide(
    split_metrics: list[dict],
    type_metrics: list[dict],
    source_metrics: list[dict],
    all_events: pd.DataFrame,
    usable_events: pd.DataFrame,
) -> dict:
    train = next(row for row in split_metrics if row["split"] == "train")
    development = next(row for row in split_metrics if row["split"] == "development")
    holdout = next(row for row in split_metrics if row["split"] == "holdout")
    total_rows = len(usable_events)
    attention_rows = int((usable_events["event_type"] == "attention").sum())
    disposition_rows = int((usable_events["event_type"] == "disposition").sum())
    adverse_rows = attention_rows + disposition_rows
    other_rows = total_rows - adverse_rows
    top_type = type_metrics[0] if type_metrics else {"event_type": "", "event_row_share": 0.0}
    top_source = source_metrics[0] if source_metrics else {"source_name": "", "event_row_share": 0.0}

    split_ready = (
        train["event_rows"] >= 100
        and development["event_rows"] >= 50
        and holdout["event_rows"] >= 50
        and train["event_day_coverage_rate"] >= 0.20
        and development["event_day_coverage_rate"] >= 0.20
        and holdout["event_day_coverage_rate"] >= 0.20
    )
    adverse_ready = (
        train["attention_rows"] + train["disposition_rows"] >= 100
        and development["attention_rows"] + development["disposition_rows"] >= 50
        and holdout["attention_rows"] + holdout["disposition_rows"] >= 50
    )
    complete_event_ready = (
        split_ready
        and other_rows >= 500
        and top_type["event_row_share"] <= 0.80
        and top_source["event_row_share"] <= 0.80
    )

    if complete_event_ready:
        status = "coverage_ready_for_full_event_feature_contract"
        next_step = "prepare_full_event_risk_feature_contract"
        reason = "歷史事件資料跨三段覆蓋足夠，且事件類型與來源未過度集中。"
        allowed_scope = "full_event_risk"
        do_not_retrain = True
    elif split_ready and adverse_ready:
        status = "coverage_ready_for_limited_attention_disposition_features"
        next_step = "prepare_limited_attention_disposition_feature_contract"
        reason = "資料跨 train/development/holdout 已足夠，但事件高度集中於 TWSE 注意/處置；只能先做有限風險特徵，不能宣稱完整事件風險。"
        allowed_scope = "attention_disposition_only"
        do_not_retrain = True
    else:
        status = "coverage_not_ready"
        next_step = "continue_historical_event_source_backfill"
        reason = "歷史事件資料仍未跨三段達到最低覆蓋標準。"
        allowed_scope = "none"
        do_not_retrain = True

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": status,
        "recommended_next_step": next_step,
        "reason": reason,
        "allowed_scope": allowed_scope,
        "event_rows_total": int(total_rows),
        "event_rows_raw_total": int(len(all_events)),
        "future_event_rows_excluded_from_coverage": int(len(all_events) - len(usable_events)),
        "attention_rows": attention_rows,
        "disposition_rows": disposition_rows,
        "attention_disposition_rows": adverse_rows,
        "other_event_rows": int(other_rows),
        "top_event_type": top_type["event_type"],
        "top_event_type_share": top_type["event_row_share"],
        "top_source_name": top_source["source_name"],
        "top_source_share": top_source["event_row_share"],
        "split_ready": bool(split_ready),
        "complete_event_ready": bool(complete_event_ready),
        "new_input_not_enabled": True,
        "do_not_retrain_yet": do_not_retrain,
        "formal_outputs_unchanged": True,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        fail(f"no rows to write for {path.name}")
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_review(decision: dict, split_metrics: list[dict], type_metrics: list[dict], source_metrics: list[dict]) -> None:
    lines = [
        "# Historical Event Risk Backfill Coverage Review",
        "",
        "- Scope: historical backfill coverage review only.",
        "- Formal output: unchanged.",
        "- Model training: not executed.",
        "- Backfilled event input: produced but not enabled in `project_config.json`.",
        "",
        "## 白話結論",
        "",
        decision["reason"],
        "",
        f"- Status: `{decision['status']}`",
        f"- Recommended next step: `{decision['recommended_next_step']}`",
        f"- Allowed scope: `{decision['allowed_scope']}`",
        f"- Usable rows by market latest date: {decision['event_rows_total']}",
        f"- Raw rows including future effective dates: {decision['event_rows_raw_total']}",
        f"- Future rows excluded from coverage: {decision['future_event_rows_excluded_from_coverage']}",
        f"- Top event type: {decision['top_event_type']} ({decision['top_event_type_share']:.2%})",
        f"- Top source: {decision['top_source_name']} ({decision['top_source_share']:.2%})",
        "",
        "## Split Coverage",
        "",
        "| split | event rows | event days | day coverage | stock coverage | attention | disposition | other |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in split_metrics:
        lines.append(
            f"| {row['split']} | {row['event_rows']} | {row['event_trading_days']} | {row['event_day_coverage_rate']:.2%} | {row['event_stock_coverage_rate']:.2%} | {row['attention_rows']} | {row['disposition_rows']} | {row['other_event_rows']} |"
        )
    lines.extend(["", "## Event Type Concentration", "", "| event type | rows | share | stocks |", "|---|---:|---:|---:|"])
    for row in type_metrics:
        lines.append(f"| {row['event_type']} | {row['event_rows']} | {row['event_row_share']:.2%} | {row['event_stock_count']} |")
    lines.extend(["", "## Source Concentration", "", "| source | rows | share | event types |", "|---|---:|---:|---|"])
    for row in source_metrics:
        lines.append(f"| {row['source_name']} | {row['event_rows']} | {row['event_row_share']:.2%} | {row['event_types']} |")
    lines.extend(
        [
            "",
            "## Decision Boundary",
            "",
            "- This review does not approve model training.",
            "- If allowed scope is limited, features must be named as attention/disposition risk only.",
            "- Full event-risk wording is blocked until TPEx, material information, and corporate action history are separately covered.",
            "- `event_risk_calendar_backfilled.csv` remains outside approved model inputs.",
            "",
            "## Outputs",
            "",
            f"- `{SPLIT_COVERAGE_PATH.relative_to(PROJECT_ROOT)}`",
            f"- `{TYPE_COVERAGE_PATH.relative_to(PROJECT_ROOT)}`",
            f"- `{SOURCE_COVERAGE_PATH.relative_to(PROJECT_ROOT)}`",
            f"- `{MONTH_COVERAGE_PATH.relative_to(PROJECT_ROOT)}`",
            f"- `{DECISION_PATH.relative_to(PROJECT_ROOT)}`",
            "",
        ]
    )
    REVIEW_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    config = load_json(CONFIG_PATH)
    backfill_decision = load_json(BACKFILL_DECISION_PATH)
    validate_inputs(config, backfill_decision)
    stock_daily, market_daily, events = load_frames(config)
    latest_market_date = market_daily["日期"].max()
    usable_events = events[events["signal_usable_date"] <= latest_market_date].copy()
    split_metrics = split_rows(stock_daily, market_daily, usable_events, config)
    type_metrics = type_rows(usable_events)
    source_metrics = source_rows(usable_events)
    month_metrics = month_rows(market_daily, usable_events)
    decision = decide(split_metrics, type_metrics, source_metrics, events, usable_events)

    write_csv(SPLIT_COVERAGE_PATH, split_metrics)
    write_csv(TYPE_COVERAGE_PATH, type_metrics)
    write_csv(SOURCE_COVERAGE_PATH, source_metrics)
    write_csv(MONTH_COVERAGE_PATH, month_metrics)
    DECISION_PATH.write_text(json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_review(decision, split_metrics, type_metrics, source_metrics)

    print("OK: historical event risk backfill coverage review completed")
    print(f"STATUS: {decision['status']}")
    print(f"NEXT_STEP: {decision['recommended_next_step']}")
    print(f"REVIEW: {REVIEW_PATH}")


if __name__ == "__main__":
    main()
