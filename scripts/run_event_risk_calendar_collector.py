from __future__ import annotations

import csv
import hashlib
import json
import re
import urllib.request
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "project_config.json"
SPEC_DECISION_PATH = PROJECT_ROOT / "decision_layer" / "event_risk_collector_spec_decision.json"
SCHEMA_PATH = PROJECT_ROOT / "data_layer" / "event_risk_calendar_schema.csv"

OUTPUT_PATH = PROJECT_ROOT / "inputs" / "event_risk_calendar.csv"
SOURCE_SUMMARY_PATH = PROJECT_ROOT / "validation_layer" / "event_risk_calendar_source_summary.csv"
REPORT_PATH = PROJECT_ROOT / "validation_layer" / "event_risk_calendar_collector_report.md"
DECISION_PATH = PROJECT_ROOT / "decision_layer" / "event_risk_calendar_collector_decision.json"

SIGNAL_CLOSE = time(13, 30, 0)
EVENT_TYPES = {
    "material_info",
    "suspension",
    "resumption",
    "attention",
    "disposition",
    "investor_meeting",
    "ex_dividend",
    "corporate_action",
}


SOURCE_ENDPOINTS = [
    {
        "source_name": "TWSE material information",
        "source_url": "https://openapi.twse.com.tw/v1/opendata/t187ap04_L",
        "event_type": "material_info",
        "market": "twse",
    },
    {
        "source_name": "TWSE attention securities",
        "source_url": "https://openapi.twse.com.tw/v1/announcement/notice",
        "event_type": "attention",
        "market": "twse",
    },
    {
        "source_name": "TWSE disposition securities",
        "source_url": "https://openapi.twse.com.tw/v1/announcement/punish",
        "event_type": "disposition",
        "market": "twse",
    },
    {
        "source_name": "TWSE trading halt and resumption",
        "source_url": "https://openapi.twse.com.tw/v1/exchangeReport/TWTAWU",
        "event_type": "suspension_resumption",
        "market": "twse",
    },
    {
        "source_name": "TWSE ex-dividend preview",
        "source_url": "https://openapi.twse.com.tw/v1/exchangeReport/TWT48U_ALL",
        "event_type": "ex_dividend",
        "market": "twse",
    },
    {
        "source_name": "TPEx material information",
        "source_url": "https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap04_O",
        "event_type": "material_info",
        "market": "tpex",
    },
    {
        "source_name": "TPEx attention securities",
        "source_url": "https://www.tpex.org.tw/openapi/v1/tpex_trading_warning_information",
        "event_type": "attention",
        "market": "tpex",
    },
    {
        "source_name": "TPEx disposition securities",
        "source_url": "https://www.tpex.org.tw/openapi/v1/tpex_disposal_information",
        "event_type": "disposition",
        "market": "tpex",
    },
    {
        "source_name": "TPEx trading halt and resumption",
        "source_url": "https://www.tpex.org.tw/openapi/v1/tpex_spendi_today",
        "event_type": "suspension_resumption",
        "market": "tpex",
    },
    {
        "source_name": "TPEx ex-dividend preview",
        "source_url": "https://www.tpex.org.tw/openapi/v1/tpex_exright_prepost",
        "event_type": "ex_dividend",
        "market": "tpex",
    },
]


def fail(message: str) -> None:
    raise SystemExit(f"FAIL: {message}")


def load_json(path: Path) -> dict:
    if not path.exists():
        fail(f"missing required file: {path.relative_to(PROJECT_ROOT)}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_contract(config: dict, spec_decision: dict) -> None:
    allowed = config.get("allowed_inputs", {})
    if set(allowed) != {"stock_daily_all", "market_daily", "theme_group"}:
        fail("collector must start from the clean three-source project contract")
    if any("event_risk_calendar" in str(value) for value in allowed.values()):
        fail("event risk calendar is not allowed as a model input in this step")
    if spec_decision.get("status") != "collector_spec_ready":
        fail("collector spec must be ready before collection")
    if spec_decision.get("recommended_next_step") != "implement_event_risk_calendar_collector":
        fail("spec decision does not allow collector implementation")
    if spec_decision.get("do_not_retrain_yet") is not True:
        fail("collector implementation must not trigger retraining")


def schema_field_names() -> list[str]:
    with SCHEMA_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        return [row["field_name"] for row in csv.DictReader(f)]


def load_universe(config: dict) -> tuple[set[str], dict[str, str], list[datetime.date]]:
    stock_path = Path(config["allowed_inputs"]["stock_daily_all"])
    theme_path = Path(config["allowed_inputs"]["theme_group"])
    market_path = Path(config["allowed_inputs"]["market_daily"])

    stock_daily = pd.read_csv(stock_path, encoding="utf-8-sig", usecols=["日期", "股票代號"])
    stock_ids = {normalize_stock_id(value) for value in stock_daily["股票代號"].dropna().unique()}

    theme = pd.read_csv(theme_path, encoding="utf-8-sig", usecols=["股票代號", "股票名稱"])
    names = {
        normalize_stock_id(row["股票代號"]): str(row["股票名稱"])
        for _, row in theme.dropna(subset=["股票代號", "股票名稱"]).iterrows()
    }

    market = pd.read_csv(market_path, encoding="utf-8-sig", usecols=["日期"])
    trading_dates = sorted(pd.to_datetime(market["日期"]).dt.date.dropna().unique().tolist())
    return stock_ids, names, trading_dates


def normalize_stock_id(value: Any) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return re.sub(r"\D", "", text)


def fetch_json(url: str) -> list[dict[str, Any]]:
    request = urllib.request.Request(url, headers={"User-Agent": "stock-ai-probability-v2/1.0"})
    with urllib.request.urlopen(request, timeout=40) as response:
        payload = response.read().decode("utf-8")
    data = json.loads(payload)
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if isinstance(data, dict):
        return [data]
    return []


def parse_roc_or_iso_date(value: Any) -> datetime.date | None:
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


def parse_time(value: Any) -> time:
    if value is None:
        return time(0, 0, 0)
    digits = re.sub(r"\D", "", str(value).strip())
    if not digits:
        return time(0, 0, 0)
    digits = digits.zfill(6)[-6:]
    hour = min(int(digits[:2]), 23)
    minute = min(int(digits[2:4]), 59)
    second = min(int(digits[4:6]), 59)
    return time(hour, minute, second)


def next_trading_day(current: datetime.date, trading_dates: list[datetime.date]) -> datetime.date:
    for day in trading_dates:
        if day > current:
            return day
    return current + timedelta(days=1)


def signal_usable_fields(
    announced_at: datetime,
    trading_dates: list[datetime.date],
) -> tuple[str, str, str]:
    if announced_at.time() <= SIGNAL_CLOSE:
        return announced_at.date().isoformat(), "true", "false"
    return next_trading_day(announced_at.date(), trading_dates).isoformat(), "false", "true"


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def raw_hash(row: dict[str, Any]) -> str:
    payload = json.dumps(row, ensure_ascii=False, sort_keys=True, default=str)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def first_value(row: dict[str, Any], keys: list[str]) -> str:
    for key in keys:
        if key in row and clean_text(row[key]):
            return clean_text(row[key])
    return ""


def material_info_event_type(title: str) -> str:
    if "法說" in title or "法人說明" in title:
        return "investor_meeting"
    if "除權" in title or "除息" in title or "股利" in title:
        return "ex_dividend"
    return "material_info"


def event_title_for(endpoint: dict[str, str], row: dict[str, Any], event_type: str) -> str:
    if event_type in {"material_info", "investor_meeting"}:
        return first_value(row, ["主旨 ", "主旨", "Subject", "Title"]) or endpoint["source_name"]
    if event_type == "attention":
        return first_value(row, ["TradingInfoForAttention", "TradingInformation"]) or endpoint["source_name"]
    if event_type == "disposition":
        return first_value(row, ["ReasonsOfDisposition", "DispositionReasons", "DispositionMeasures", "DisposalCondition"]) or endpoint["source_name"]
    if event_type in {"suspension", "resumption"}:
        return endpoint["source_name"]
    if event_type == "ex_dividend":
        return first_value(row, ["Exdividend", "ExRrightsExDividend"]) or endpoint["source_name"]
    return endpoint["source_name"]


def stock_id_for(row: dict[str, Any]) -> str:
    return normalize_stock_id(first_value(row, ["公司代號", "Code", "SecuritiesCompanyCode"]))


def stock_name_for(row: dict[str, Any], names: dict[str, str], stock_id: str) -> str:
    return first_value(row, ["公司名稱", "Name", "CompanyName"]) or names.get(stock_id, "")


def date_for(endpoint: dict[str, str], row: dict[str, Any], event_type: str) -> datetime.date | None:
    if event_type in {"material_info", "investor_meeting"}:
        return parse_roc_or_iso_date(first_value(row, ["發言日期", "Date", "出表日期"]))
    if event_type == "suspension":
        return parse_roc_or_iso_date(first_value(row, ["TradingHaltDate", "Date"]))
    if event_type == "resumption":
        return parse_roc_or_iso_date(first_value(row, ["TradingResumptionDate", "Date"]))
    if event_type == "ex_dividend":
        return parse_roc_or_iso_date(first_value(row, ["Date", "ExRrightsExDividendDate"]))
    return parse_roc_or_iso_date(first_value(row, ["Date", "出表日期"]))


def time_for(row: dict[str, Any], event_type: str) -> time:
    if event_type in {"material_info", "investor_meeting"}:
        return parse_time(first_value(row, ["發言時間"]))
    if event_type == "suspension":
        return parse_time(first_value(row, ["TradingHaltTime"]))
    if event_type == "resumption":
        return parse_time(first_value(row, ["TradingResumptionTime"]))
    return time(0, 0, 0)


def effective_dates(row: dict[str, Any], event_type: str) -> tuple[str, str]:
    if event_type == "disposition":
        period = first_value(row, ["DispositionPeriod"])
        dates = re.findall(r"\d{6,8}", period)
        if len(dates) >= 2:
            start = parse_roc_or_iso_date(dates[0])
            end = parse_roc_or_iso_date(dates[1])
            return (start.isoformat() if start else "", end.isoformat() if end else "")
    if event_type == "suspension":
        start = parse_roc_or_iso_date(first_value(row, ["TradingHaltDate"]))
        return (start.isoformat() if start else "", "")
    if event_type == "resumption":
        start = parse_roc_or_iso_date(first_value(row, ["TradingResumptionDate"]))
        return (start.isoformat() if start else "", "")
    if event_type == "ex_dividend":
        start = parse_roc_or_iso_date(first_value(row, ["Date", "ExRrightsExDividendDate"]))
        return (start.isoformat() if start else "", "")
    return "", ""


def expand_event_types(endpoint: dict[str, str], row: dict[str, Any]) -> list[str]:
    base = endpoint["event_type"]
    if base == "suspension_resumption":
        event_types: list[str] = []
        if first_value(row, ["TradingHaltDate", "暫停交易"]):
            event_types.append("suspension")
        if first_value(row, ["TradingResumptionDate", "恢復交易"]):
            event_types.append("resumption")
        return event_types
    if base == "material_info":
        title = event_title_for(endpoint, row, "material_info")
        return [material_info_event_type(title)]
    return [base]


def normalize_event(
    endpoint: dict[str, str],
    row: dict[str, Any],
    stock_ids: set[str],
    names: dict[str, str],
    trading_dates: list[datetime.date],
) -> list[dict[str, str]]:
    stock_id = stock_id_for(row)
    if not stock_id or stock_id not in stock_ids:
        return []
    events: list[dict[str, str]] = []
    for event_type in expand_event_types(endpoint, row):
        if event_type not in EVENT_TYPES:
            continue
        event_date = date_for(endpoint, row, event_type)
        if event_date is None:
            continue
        announced_at = datetime.combine(event_date, time_for(row, event_type))
        usable_date, known_before_close, post_close = signal_usable_fields(announced_at, trading_dates)
        start_date, end_date = effective_dates(row, event_type)
        title = event_title_for(endpoint, row, event_type)
        event_subtype = first_value(
            row,
            ["符合條款", "ReasonsOfDisposition", "DispositionReasons", "DispositionMeasures", "TradingInfoForAttention", "TradingInformation"],
        )
        events.append(
            {
                "stock_id": stock_id,
                "stock_name": stock_name_for(row, names, stock_id),
                "event_type": event_type,
                "event_subtype": event_subtype,
                "event_title": title,
                "source_name": endpoint["source_name"],
                "source_url": endpoint["source_url"],
                "announcement_datetime": announced_at.strftime("%Y-%m-%d %H:%M:%S"),
                "event_effective_start_date": start_date,
                "event_effective_end_date": end_date,
                "signal_usable_date": usable_date,
                "known_before_signal_close": known_before_close,
                "post_close_pre_next_open": post_close,
                "raw_payload_hash": raw_hash(row),
            }
        )
    return events


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


def collect_events(config: dict) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    stock_ids, names, trading_dates = load_universe(config)
    events: list[dict[str, str]] = []
    summaries: list[dict[str, Any]] = []
    for endpoint in SOURCE_ENDPOINTS:
        status = "ok"
        error = ""
        raw_count = 0
        kept_count = 0
        try:
            rows = fetch_json(endpoint["source_url"])
            raw_count = len(rows)
            for row in rows:
                normalized = normalize_event(endpoint, row, stock_ids, names, trading_dates)
                kept_count += len(normalized)
                events.extend(normalized)
        except Exception as exc:  # network and source drift are reported, not hidden
            status = "error"
            error = str(exc)
        summaries.append(
            {
                "source_name": endpoint["source_name"],
                "source_url": endpoint["source_url"],
                "event_type": endpoint["event_type"],
                "status": status,
                "raw_rows": raw_count,
                "kept_events": kept_count,
                "error": error,
            }
        )
    return dedupe_events(events), summaries


def write_events(events: list[dict[str, str]], columns: list[str]) -> None:
    with OUTPUT_PATH.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for event in events:
            writer.writerow({column: event.get(column, "") for column in columns})


def write_summary(summaries: list[dict[str, Any]]) -> None:
    fieldnames = ["source_name", "source_url", "event_type", "status", "raw_rows", "kept_events", "error"]
    with SOURCE_SUMMARY_PATH.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)


def write_report(events: list[dict[str, str]], summaries: list[dict[str, Any]]) -> None:
    by_type: dict[str, int] = {}
    for event in events:
        by_type[event["event_type"]] = by_type.get(event["event_type"], 0) + 1
    source_errors = [row for row in summaries if row["status"] != "ok"]
    lines = [
        "# Event Risk Calendar Collector Report",
        "",
        "- Scope: official API snapshot collection.",
        "- Formal output: unchanged.",
        "- Model training: not executed.",
        "- New input source: produced but not enabled in `project_config.json`.",
        "",
        "## 白話結論",
        "",
        f"已產生 `inputs/event_risk_calendar.csv`，共 {len(events)} 筆事件。這份檔案目前只供覆蓋率與防偷看檢查，尚未允許進主模型。",
        "",
        "## Event Counts",
        "",
        "| event type | rows |",
        "|---|---:|",
    ]
    for event_type in sorted(by_type):
        lines.append(f"| {event_type} | {by_type[event_type]} |")
    lines.extend(
        [
            "",
            "## Source Summary",
            "",
            "| source | raw rows | kept events | status |",
            "|---|---:|---:|---|",
        ]
    )
    for row in summaries:
        lines.append(f"| {row['source_name']} | {row['raw_rows']} | {row['kept_events']} | {row['status']} |")
    if source_errors:
        lines.extend(["", "## Source Errors", ""])
        for row in source_errors:
            lines.append(f"- {row['source_name']}: {row['error']}")
    lines.extend(
        [
            "",
            "## Boundaries",
            "",
            "- This collector does not modify the approved three model inputs.",
            "- This collector does not train or promote a model.",
            "- Rows announced after signal-day close are retained but marked `post_close_pre_next_open=true`.",
            "- Rows without parseable public announcement date/time are excluded.",
            "",
            "## Outputs",
            "",
            f"- `{OUTPUT_PATH.relative_to(PROJECT_ROOT)}`",
            f"- `{SOURCE_SUMMARY_PATH.relative_to(PROJECT_ROOT)}`",
            f"- `{DECISION_PATH.relative_to(PROJECT_ROOT)}`",
            "",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def write_decision(events: list[dict[str, str]], summaries: list[dict[str, Any]]) -> None:
    source_errors = [row for row in summaries if row["status"] != "ok"]
    payload = {
        "status": "collector_output_ready" if events else "collector_output_empty",
        "recommended_next_step": "review_event_risk_calendar_coverage",
        "output_path": str(OUTPUT_PATH.relative_to(PROJECT_ROOT)),
        "rows": len(events),
        "source_count": len(summaries),
        "source_error_count": len(source_errors),
        "new_input_not_enabled": True,
        "do_not_retrain_yet": True,
        "formal_outputs_unchanged": True,
        "official_source_families": [
            "TWSE OpenAPI",
            "TPEx OpenAPI",
        ],
    }
    DECISION_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    config = load_json(CONFIG_PATH)
    spec_decision = load_json(SPEC_DECISION_PATH)
    validate_contract(config, spec_decision)
    columns = schema_field_names()
    events, summaries = collect_events(config)
    write_events(events, columns)
    write_summary(summaries)
    write_report(events, summaries)
    write_decision(events, summaries)

    print("OK: event risk calendar collector completed")
    print(f"ROWS: {len(events)}")
    print(f"OUTPUT: {OUTPUT_PATH}")
    print(f"NEXT_STEP: review_event_risk_calendar_coverage")


if __name__ == "__main__":
    main()
