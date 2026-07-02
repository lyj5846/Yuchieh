from __future__ import annotations

import csv
import hashlib
import json
import re
import urllib.parse
import urllib.request
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "project_config.json"
COVERAGE_DECISION_PATH = PROJECT_ROOT / "decision_layer" / "event_risk_calendar_coverage_decision.json"
SCHEMA_PATH = PROJECT_ROOT / "data_layer" / "event_risk_calendar_schema.csv"
CURRENT_EVENT_PATH = PROJECT_ROOT / "inputs" / "event_risk_calendar.csv"

OUTPUT_PATH = PROJECT_ROOT / "inputs" / "event_risk_calendar_backfilled.csv"
SOURCE_SUMMARY_PATH = PROJECT_ROOT / "validation_layer" / "historical_event_risk_backfill_source_summary.csv"
REPORT_PATH = PROJECT_ROOT / "validation_layer" / "historical_event_risk_backfill_report.md"
DECISION_PATH = PROJECT_ROOT / "decision_layer" / "historical_event_risk_backfill_decision.json"

SIGNAL_CLOSE = time(13, 30, 0)


def fail(message: str) -> None:
    raise SystemExit(f"FAIL: {message}")


def load_json(path: Path) -> dict:
    if not path.exists():
        fail(f"missing required file: {path.relative_to(PROJECT_ROOT)}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_contract(config: dict, coverage_decision: dict) -> None:
    allowed = config.get("allowed_inputs", {})
    if set(allowed) != {"stock_daily_all", "market_daily", "theme_group"}:
        fail("backfill collector must preserve the three approved model inputs")
    if any("event_risk_calendar" in str(value) for value in allowed.values()):
        fail("event calendar files must not be enabled as model inputs in this step")
    if coverage_decision.get("status") != "coverage_not_training_ready":
        fail("backfill is only valid after coverage is judged insufficient")
    if coverage_decision.get("recommended_next_step") != "build_historical_event_risk_backfill_collector":
        fail("coverage decision does not request historical backfill")
    if coverage_decision.get("do_not_retrain_yet") is not True:
        fail("backfill collector must not trigger retraining")


def schema_field_names() -> list[str]:
    with SCHEMA_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        return [row["field_name"] for row in csv.DictReader(f)]


def normalize_stock_id(value: Any) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return re.sub(r"\D", "", text)


def load_universe(config: dict) -> tuple[set[str], dict[str, str], list[date], date, date]:
    stock_daily = pd.read_csv(config["allowed_inputs"]["stock_daily_all"], encoding="utf-8-sig", usecols=["日期", "股票代號"])
    stock_ids = {normalize_stock_id(value) for value in stock_daily["股票代號"].dropna().unique()}

    theme = pd.read_csv(config["allowed_inputs"]["theme_group"], encoding="utf-8-sig", usecols=["股票代號", "股票名稱"])
    names = {
        normalize_stock_id(row["股票代號"]): str(row["股票名稱"])
        for _, row in theme.dropna(subset=["股票代號", "股票名稱"]).iterrows()
    }

    market = pd.read_csv(config["allowed_inputs"]["market_daily"], encoding="utf-8-sig", usecols=["日期"])
    trading_dates = sorted(pd.to_datetime(market["日期"]).dt.date.dropna().unique().tolist())
    if not trading_dates:
        fail("market trading dates are empty")
    return stock_ids, names, trading_dates, trading_dates[0], trading_dates[-1]


def month_chunks(start: date, end: date) -> list[tuple[date, date]]:
    chunks: list[tuple[date, date]] = []
    current = date(start.year, start.month, 1)
    while current <= end:
        next_month = date(current.year + (current.month // 12), (current.month % 12) + 1, 1)
        chunk_start = max(start, current)
        chunk_end = min(end, next_month - timedelta(days=1))
        chunks.append((chunk_start, chunk_end))
        current = next_month
    return chunks


def fetch_twse_rwd(path: str, start: date, end: date) -> dict[str, Any]:
    query = urllib.parse.urlencode(
        {
            "response": "json",
            "startDate": start.strftime("%Y%m%d"),
            "endDate": end.strftime("%Y%m%d"),
        }
    )
    url = f"https://www.twse.com.tw/rwd/zh/{path}?{query}"
    request = urllib.request.Request(url, headers={"User-Agent": "stock-ai-probability-v2/1.0"})
    with urllib.request.urlopen(request, timeout=40) as response:
        payload = response.read().decode("utf-8")
    data = json.loads(payload)
    if not isinstance(data, dict) or data.get("stat") != "OK":
        raise ValueError(f"unexpected TWSE response for {path}: {data!r}")
    return data


def parse_roc_or_iso_date(value: Any) -> date | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    digits = re.sub(r"\D", "", text)
    if len(digits) == 8 and digits.startswith("20"):
        return datetime.strptime(digits, "%Y%m%d").date()
    if len(digits) == 7:
        year = int(digits[:3]) + 1911
        return datetime.strptime(f"{year}{digits[3:]}", "%Y%m%d").date()
    if len(digits) == 6:
        year = int(digits[:2]) + 1911
        return datetime.strptime(f"{year}{digits[2:]}", "%Y%m%d").date()
    return None


def next_trading_day(current: date, trading_dates: list[date]) -> date:
    for day in trading_dates:
        if day > current:
            return day
    return current + timedelta(days=1)


def signal_usable_fields(announced_at: datetime, trading_dates: list[date]) -> tuple[str, str, str]:
    if announced_at.time() <= SIGNAL_CLOSE:
        return announced_at.date().isoformat(), "true", "false"
    return next_trading_day(announced_at.date(), trading_dates).isoformat(), "false", "true"


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def raw_hash(source: str, row: dict[str, Any]) -> str:
    payload = json.dumps({"source": source, "row": row}, ensure_ascii=False, sort_keys=True, default=str)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def row_dict(fields: list[str], values: list[Any]) -> dict[str, Any]:
    return {field: values[index] if index < len(values) else "" for index, field in enumerate(fields)}


def parse_disposition_period(value: str) -> tuple[str, str]:
    dates = re.findall(r"\d{2,3}[./]\d{1,2}[./]\d{1,2}|\d{6,8}", value)
    if len(dates) < 2:
        return "", ""
    start = parse_roc_or_iso_date(dates[0])
    end = parse_roc_or_iso_date(dates[1])
    return (start.isoformat() if start else "", end.isoformat() if end else "")


def normalize_twse_notice(
    raw: dict[str, Any],
    stock_ids: set[str],
    names: dict[str, str],
    trading_dates: list[date],
) -> dict[str, str] | None:
    stock_id = normalize_stock_id(raw.get("證券代號", ""))
    if stock_id not in stock_ids:
        return None
    event_date = parse_roc_or_iso_date(raw.get("日期"))
    if event_date is None:
        return None
    announced_at = datetime.combine(event_date, time(0, 0, 0))
    usable_date, known_before_close, post_close = signal_usable_fields(announced_at, trading_dates)
    title = clean_text(raw.get("注意交易資訊")) or "TWSE historical attention"
    return {
        "stock_id": stock_id,
        "stock_name": clean_text(raw.get("證券名稱")) or names.get(stock_id, ""),
        "event_type": "attention",
        "event_subtype": clean_text(raw.get("累計次數")),
        "event_title": title,
        "source_name": "TWSE historical attention",
        "source_url": "https://www.twse.com.tw/rwd/zh/announcement/notice",
        "announcement_datetime": announced_at.strftime("%Y-%m-%d %H:%M:%S"),
        "event_effective_start_date": "",
        "event_effective_end_date": "",
        "signal_usable_date": usable_date,
        "known_before_signal_close": known_before_close,
        "post_close_pre_next_open": post_close,
        "raw_payload_hash": raw_hash("TWSE historical attention", raw),
    }


def normalize_twse_punish(
    raw: dict[str, Any],
    stock_ids: set[str],
    names: dict[str, str],
    trading_dates: list[date],
) -> dict[str, str] | None:
    stock_id = normalize_stock_id(raw.get("證券代號", ""))
    if stock_id not in stock_ids:
        return None
    event_date = parse_roc_or_iso_date(raw.get("公布日期"))
    if event_date is None:
        return None
    announced_at = datetime.combine(event_date, time(0, 0, 0))
    usable_date, known_before_close, post_close = signal_usable_fields(announced_at, trading_dates)
    start_date, end_date = parse_disposition_period(clean_text(raw.get("處置起迄時間")))
    title = clean_text(raw.get("處置條件")) or clean_text(raw.get("處置措施")) or "TWSE historical disposition"
    return {
        "stock_id": stock_id,
        "stock_name": clean_text(raw.get("證券名稱")) or names.get(stock_id, ""),
        "event_type": "disposition",
        "event_subtype": clean_text(raw.get("處置措施")),
        "event_title": title,
        "source_name": "TWSE historical disposition",
        "source_url": "https://www.twse.com.tw/rwd/zh/announcement/punish",
        "announcement_datetime": announced_at.strftime("%Y-%m-%d %H:%M:%S"),
        "event_effective_start_date": start_date,
        "event_effective_end_date": end_date,
        "signal_usable_date": usable_date,
        "known_before_signal_close": known_before_close,
        "post_close_pre_next_open": post_close,
        "raw_payload_hash": raw_hash("TWSE historical disposition", raw),
    }


def dedupe_events(events: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str, str, str, str]] = set()
    deduped: list[dict[str, str]] = []
    for event in events:
        key = (
            event["stock_id"],
            event["event_type"],
            event["announcement_datetime"],
            event["event_title"],
            event["source_name"],
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(event)
    return sorted(deduped, key=lambda item: (item["announcement_datetime"], item["stock_id"], item["event_type"]))


def load_current_snapshot(columns: list[str]) -> list[dict[str, str]]:
    if not CURRENT_EVENT_PATH.exists():
        return []
    with CURRENT_EVENT_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        return [{column: row.get(column, "") for column in columns} for row in csv.DictReader(f)]


def collect_historical(config: dict, columns: list[str]) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    stock_ids, names, trading_dates, start_date, end_date = load_universe(config)
    chunks = month_chunks(start_date, end_date)
    events: list[dict[str, str]] = []
    summaries: list[dict[str, Any]] = []
    sources = [
        {
            "source_name": "TWSE historical attention",
            "path": "announcement/notice",
            "normalizer": normalize_twse_notice,
        },
        {
            "source_name": "TWSE historical disposition",
            "path": "announcement/punish",
            "normalizer": normalize_twse_punish,
        },
    ]
    for source in sources:
        raw_rows = 0
        kept_events = 0
        errors: list[str] = []
        for chunk_start, chunk_end in chunks:
            try:
                payload = fetch_twse_rwd(source["path"], chunk_start, chunk_end)
                fields = [str(field) for field in payload.get("fields", [])]
                data_rows = payload.get("data", []) or []
                raw_rows += len(data_rows)
                for values in data_rows:
                    raw = row_dict(fields, list(values))
                    event = source["normalizer"](raw, stock_ids, names, trading_dates)
                    if event is None:
                        continue
                    kept_events += 1
                    events.append(event)
            except Exception as exc:
                errors.append(f"{chunk_start.isoformat()}..{chunk_end.isoformat()}: {exc}")
        summaries.append(
            {
                "source_name": source["source_name"],
                "source_url": f"https://www.twse.com.tw/rwd/zh/{source['path']}",
                "query_granularity": "monthly",
                "raw_rows": raw_rows,
                "kept_events": kept_events,
                "status": "ok" if not errors else "partial_error",
                "error": " | ".join(errors[:5]),
            }
        )

    snapshot_events = load_current_snapshot(columns)
    events.extend(snapshot_events)
    summaries.append(
        {
            "source_name": "current official OpenAPI snapshot",
            "source_url": str(CURRENT_EVENT_PATH.relative_to(PROJECT_ROOT)),
            "query_granularity": "snapshot",
            "raw_rows": len(snapshot_events),
            "kept_events": len(snapshot_events),
            "status": "ok" if snapshot_events else "empty",
            "error": "",
        }
    )
    return dedupe_events(events), summaries


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str] | None = None) -> None:
    if not rows:
        fail(f"no rows to write for {path.name}")
    fieldnames = columns or list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def split_counts(config: dict, events: list[dict[str, str]]) -> dict[str, int]:
    counts = {"train": 0, "development": 0, "holdout": 0}
    train_end = pd.Timestamp(config["time_split"]["train_end"])
    dev_start = pd.Timestamp(config["time_split"]["dev_start"])
    dev_end = pd.Timestamp(config["time_split"]["dev_end"])
    holdout_start = pd.Timestamp(config["time_split"]["holdout_start"])
    for event in events:
        day = pd.Timestamp(event["signal_usable_date"])
        if day <= train_end:
            counts["train"] += 1
        elif dev_start <= day <= dev_end:
            counts["development"] += 1
        elif day >= holdout_start:
            counts["holdout"] += 1
    return counts


def write_report(config: dict, events: list[dict[str, str]], summaries: list[dict[str, Any]]) -> dict[str, Any]:
    counts = split_counts(config, events)
    by_type: dict[str, int] = {}
    for event in events:
        by_type[event["event_type"]] = by_type.get(event["event_type"], 0) + 1
    has_train_dev = counts["train"] > 0 and counts["development"] > 0
    decision = {
        "status": "historical_backfill_output_ready" if has_train_dev else "historical_backfill_still_insufficient",
        "recommended_next_step": "review_historical_event_risk_backfill_coverage",
        "output_path": str(OUTPUT_PATH.relative_to(PROJECT_ROOT)),
        "rows": len(events),
        "split_event_rows": counts,
        "new_input_not_enabled": True,
        "do_not_retrain_yet": True,
        "formal_outputs_unchanged": True,
        "known_limitation": "TWSE attention/disposition historical rows are backfilled; material information, TPEx attention/disposition, and ex-dividend historical backfill still need separate source work.",
    }
    lines = [
        "# Historical Event Risk Backfill Report",
        "",
        "- Scope: historical backfill collector output.",
        "- Formal output: unchanged.",
        "- Model training: not executed.",
        "- New input source: produced but not enabled in `project_config.json`.",
        "",
        "## 白話結論",
        "",
        "已建立歷史事件回補檔，但目前回補主要來自 TWSE 歷史注意/處置資料；TPEx 與重大訊息歷史仍需要另外處理，and still need separate source work. 因此這一步仍不允許重訓。",
        "",
        "## Split Rows",
        "",
        "| split | rows |",
        "|---|---:|",
    ]
    for split_name in ["train", "development", "holdout"]:
        lines.append(f"| {split_name} | {counts[split_name]} |")
    lines.extend(["", "## Event Counts", "", "| event type | rows |", "|---|---:|"])
    for event_type in sorted(by_type):
        lines.append(f"| {event_type} | {by_type[event_type]} |")
    lines.extend(["", "## Source Summary", "", "| source | raw rows | kept events | status |", "|---|---:|---:|---|"])
    for row in summaries:
        lines.append(f"| {row['source_name']} | {row['raw_rows']} | {row['kept_events']} | {row['status']} |")
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- This file is not added to approved model inputs.",
            "- This file needs a separate coverage review before any model feature contract.",
            "- This step keeps current snapshot rows but does not overwrite the snapshot file.",
            "",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    DECISION_PATH.write_text(json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return decision


def main() -> None:
    config = load_json(CONFIG_PATH)
    coverage_decision = load_json(COVERAGE_DECISION_PATH)
    validate_contract(config, coverage_decision)
    columns = schema_field_names()
    events, summaries = collect_historical(config, columns)
    write_csv(OUTPUT_PATH, events, columns)
    write_csv(SOURCE_SUMMARY_PATH, summaries)
    decision = write_report(config, events, summaries)
    print("OK: historical event risk backfill completed")
    print(f"STATUS: {decision['status']}")
    print(f"ROWS: {decision['rows']}")
    print(f"NEXT_STEP: {decision['recommended_next_step']}")


if __name__ == "__main__":
    main()
