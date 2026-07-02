from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "project_config.json"
COLLECTOR_DECISION_PATH = PROJECT_ROOT / "decision_layer" / "event_risk_calendar_collector_decision.json"
EVENT_CALENDAR_PATH = PROJECT_ROOT / "inputs" / "event_risk_calendar.csv"

REVIEW_PATH = PROJECT_ROOT / "validation_layer" / "event_risk_calendar_coverage_review.md"
SPLIT_COVERAGE_PATH = PROJECT_ROOT / "validation_layer" / "event_risk_calendar_coverage_by_split.csv"
MONTH_COVERAGE_PATH = PROJECT_ROOT / "validation_layer" / "event_risk_calendar_coverage_by_month.csv"
EVENT_TYPE_COVERAGE_PATH = PROJECT_ROOT / "validation_layer" / "event_risk_calendar_coverage_by_type.csv"
DECISION_PATH = PROJECT_ROOT / "decision_layer" / "event_risk_calendar_coverage_decision.json"


def fail(message: str) -> None:
    raise SystemExit(f"FAIL: {message}")


def load_json(path: Path) -> dict:
    if not path.exists():
        fail(f"missing required file: {path.relative_to(PROJECT_ROOT)}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_inputs(config: dict, collector_decision: dict) -> None:
    allowed = config.get("allowed_inputs", {})
    if set(allowed) != {"stock_daily_all", "market_daily", "theme_group"}:
        fail("coverage review must preserve the three approved model inputs")
    if any("event_risk_calendar" in str(value) for value in allowed.values()):
        fail("event_risk_calendar must not be enabled as a model input before coverage approval")
    if collector_decision.get("status") != "collector_output_ready":
        fail("collector output must be ready before coverage review")
    if collector_decision.get("recommended_next_step") != "review_event_risk_calendar_coverage":
        fail("collector decision must recommend coverage review")
    if collector_decision.get("do_not_retrain_yet") is not True:
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
    event_calendar = pd.read_csv(EVENT_CALENDAR_PATH, encoding="utf-8-sig")

    stock_daily["日期"] = pd.to_datetime(stock_daily["日期"])
    market_daily["日期"] = pd.to_datetime(market_daily["日期"])
    event_calendar["announcement_datetime"] = pd.to_datetime(event_calendar["announcement_datetime"])
    event_calendar["signal_usable_date"] = pd.to_datetime(event_calendar["signal_usable_date"])
    event_calendar["stock_id"] = event_calendar["stock_id"].astype(str).str.replace(r"\D", "", regex=True)
    stock_daily["股票代號"] = stock_daily["股票代號"].astype(str).str.replace(r"\.0$", "", regex=True).str.replace(r"\D", "", regex=True)
    return stock_daily, market_daily, event_calendar


def split_rows(config: dict, market_daily: pd.DataFrame, event_calendar: pd.DataFrame) -> list[dict]:
    latest_market_date = market_daily["日期"].max()
    rows: list[dict] = []
    for split_name in ["train", "development", "holdout"]:
        market_split = market_daily[market_daily["日期"].map(lambda day: split_for_date(day, config) == split_name)]
        if split_name == "holdout":
            market_split = market_split[market_split["日期"] <= latest_market_date]
        event_split = event_calendar[
            event_calendar["signal_usable_date"].map(lambda day: split_for_date(day, config) == split_name)
            & (event_calendar["signal_usable_date"] <= latest_market_date)
        ]
        trading_days = int(market_split["日期"].nunique())
        event_days = int(event_split["signal_usable_date"].dt.date.nunique())
        rows.append(
            {
                "split": split_name,
                "start_date": market_split["日期"].min().date().isoformat() if not market_split.empty else "",
                "end_date": market_split["日期"].max().date().isoformat() if not market_split.empty else "",
                "trading_days": trading_days,
                "event_rows": int(len(event_split)),
                "event_trading_days": event_days,
                "event_stock_count": int(event_split["stock_id"].nunique()) if not event_split.empty else 0,
                "event_day_coverage_rate": event_days / trading_days if trading_days else 0.0,
            }
        )
    return rows


def month_rows(market_daily: pd.DataFrame, event_calendar: pd.DataFrame) -> list[dict]:
    market = market_daily.copy()
    market["month"] = market["日期"].dt.strftime("%Y-%m")
    events = event_calendar[event_calendar["signal_usable_date"] <= market["日期"].max()].copy()
    events["month"] = events["signal_usable_date"].dt.strftime("%Y-%m")
    rows: list[dict] = []
    for month, market_group in market.groupby("month"):
        event_group = events[events["month"] == month]
        rows.append(
            {
                "month": month,
                "trading_days": int(market_group["日期"].nunique()),
                "event_rows": int(len(event_group)),
                "event_trading_days": int(event_group["signal_usable_date"].dt.date.nunique()) if not event_group.empty else 0,
                "event_stock_count": int(event_group["stock_id"].nunique()) if not event_group.empty else 0,
            }
        )
    return rows


def event_type_rows(event_calendar: pd.DataFrame, latest_market_date: pd.Timestamp) -> list[dict]:
    usable = event_calendar[event_calendar["signal_usable_date"] <= latest_market_date]
    rows: list[dict] = []
    for event_type, group in usable.groupby("event_type"):
        rows.append(
            {
                "event_type": event_type,
                "event_rows": int(len(group)),
                "event_stock_count": int(group["stock_id"].nunique()),
                "first_signal_usable_date": group["signal_usable_date"].min().date().isoformat(),
                "last_signal_usable_date": group["signal_usable_date"].max().date().isoformat(),
            }
        )
    return sorted(rows, key=lambda row: row["event_type"])


def decide(
    config: dict,
    stock_daily: pd.DataFrame,
    market_daily: pd.DataFrame,
    event_calendar: pd.DataFrame,
    split_metrics: list[dict],
) -> dict:
    latest_market_date = market_daily["日期"].max()
    usable_events = event_calendar[event_calendar["signal_usable_date"] <= latest_market_date]
    train = next(row for row in split_metrics if row["split"] == "train")
    dev = next(row for row in split_metrics if row["split"] == "development")
    holdout = next(row for row in split_metrics if row["split"] == "holdout")
    total_trading_days = int(market_daily["日期"].nunique())
    event_days = int(usable_events["signal_usable_date"].dt.date.nunique()) if not usable_events.empty else 0
    stock_universe = int(stock_daily["股票代號"].nunique())
    event_stock_count = int(usable_events["stock_id"].nunique()) if not usable_events.empty else 0
    earliest_event = usable_events["signal_usable_date"].min().date().isoformat() if not usable_events.empty else ""
    latest_event = usable_events["signal_usable_date"].max().date().isoformat() if not usable_events.empty else ""

    training_ready = (
        train["event_rows"] >= 100
        and dev["event_rows"] >= 30
        and holdout["event_rows"] >= 30
        and train["event_day_coverage_rate"] >= 0.20
        and dev["event_day_coverage_rate"] >= 0.20
    )
    status = "coverage_training_ready" if training_ready else "coverage_not_training_ready"
    next_step = "prepare_event_features_for_model_contract" if training_ready else "build_historical_event_risk_backfill_collector"
    reason = (
        "事件資料已跨 train/development/holdout 有足夠覆蓋，可進入特徵契約審查。"
        if training_ready
        else "目前事件資料只是官方 API 快照，主要落在最近日期，train/development 幾乎沒有覆蓋；不能拿去重訓。"
    )
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": status,
        "recommended_next_step": next_step,
        "reason": reason,
        "event_rows_total": int(len(event_calendar)),
        "event_rows_usable_by_market_latest": int(len(usable_events)),
        "event_first_signal_usable_date": earliest_event,
        "event_last_signal_usable_date": latest_event,
        "market_start_date": market_daily["日期"].min().date().isoformat(),
        "market_latest_date": latest_market_date.date().isoformat(),
        "total_trading_days": total_trading_days,
        "event_trading_days": event_days,
        "overall_event_day_coverage_rate": event_days / total_trading_days if total_trading_days else 0.0,
        "stock_universe_count": stock_universe,
        "event_stock_count": event_stock_count,
        "event_stock_coverage_rate": event_stock_count / stock_universe if stock_universe else 0.0,
        "split_metrics": split_metrics,
        "new_input_not_enabled": True,
        "do_not_retrain_yet": not training_ready,
        "formal_outputs_unchanged": True,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        fail(f"no rows to write for {path.name}")
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_review(decision: dict, split_metrics: list[dict], type_metrics: list[dict]) -> None:
    lines = [
        "# Event Risk Calendar Coverage Review",
        "",
        "- Scope: coverage review only.",
        "- Formal output: unchanged.",
        "- Model training: not executed.",
        "- Event input source: produced but not enabled in `project_config.json`.",
        "",
        "## 白話結論",
        "",
        decision["reason"],
        "",
        f"- Status: `{decision['status']}`",
        f"- Recommended next step: `{decision['recommended_next_step']}`",
        f"- Usable rows by market latest date: {decision['event_rows_usable_by_market_latest']}",
        f"- Event date range: {decision['event_first_signal_usable_date']} to {decision['event_last_signal_usable_date']}",
        "",
        "## Split Coverage",
        "",
        "| split | trading days | event rows | event days | coverage | stocks |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in split_metrics:
        lines.append(
            f"| {row['split']} | {row['trading_days']} | {row['event_rows']} | {row['event_trading_days']} | {row['event_day_coverage_rate']:.2%} | {row['event_stock_count']} |"
        )
    lines.extend(
        [
            "",
            "## Event Type Coverage",
            "",
            "| event type | rows | stocks | first date | last date |",
            "|---|---:|---:|---|---|",
        ]
    )
    for row in type_metrics:
        lines.append(
            f"| {row['event_type']} | {row['event_rows']} | {row['event_stock_count']} | {row['first_signal_usable_date']} | {row['last_signal_usable_date']} |"
        )
    lines.extend(
        [
            "",
            "## Decision Boundary",
            "",
            "- This is not training-ready unless train and development both have enough historical coverage.",
            "- Current official API snapshot can be kept as the newest event layer, but historical backfill is required before model use.",
            "- No model may use `inputs/event_risk_calendar.csv` until the backfill and feature contract pass.",
            "",
            "## Outputs",
            "",
            f"- `{SPLIT_COVERAGE_PATH.relative_to(PROJECT_ROOT)}`",
            f"- `{MONTH_COVERAGE_PATH.relative_to(PROJECT_ROOT)}`",
            f"- `{EVENT_TYPE_COVERAGE_PATH.relative_to(PROJECT_ROOT)}`",
            f"- `{DECISION_PATH.relative_to(PROJECT_ROOT)}`",
            "",
        ]
    )
    REVIEW_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    config = load_json(CONFIG_PATH)
    collector_decision = load_json(COLLECTOR_DECISION_PATH)
    validate_inputs(config, collector_decision)
    stock_daily, market_daily, event_calendar = load_frames(config)
    split_metrics = split_rows(config, market_daily, event_calendar)
    month_metrics = month_rows(market_daily, event_calendar)
    type_metrics = event_type_rows(event_calendar, market_daily["日期"].max())
    decision = decide(config, stock_daily, market_daily, event_calendar, split_metrics)

    write_csv(SPLIT_COVERAGE_PATH, split_metrics)
    write_csv(MONTH_COVERAGE_PATH, month_metrics)
    write_csv(EVENT_TYPE_COVERAGE_PATH, type_metrics)
    DECISION_PATH.write_text(json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_review(decision, split_metrics, type_metrics)

    print("OK: event risk calendar coverage review completed")
    print(f"STATUS: {decision['status']}")
    print(f"NEXT_STEP: {decision['recommended_next_step']}")
    print(f"REVIEW: {REVIEW_PATH}")


if __name__ == "__main__":
    main()
