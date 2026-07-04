from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "project_config.json"

DATA_STATUS_PATH = PROJECT_ROOT / "data_layer" / "main_pipeline_data_status.md"
LABEL_CONTRACT_PATH = PROJECT_ROOT / "label_layer" / "label_contract.md"
FEATURE_CONTRACT_PATH = PROJECT_ROOT / "feature_layer" / "feature_contract.md"
MODEL_CONTRACT_PATH = PROJECT_ROOT / "model_layer" / "main_model_contract.md"
VALIDATION_CONTRACT_PATH = PROJECT_ROOT / "validation_layer" / "validation_contract.md"
DECISION_PATH = PROJECT_ROOT / "decision_layer" / "main_pipeline_decision.md"
FORMAL_STATUS_PATH = PROJECT_ROOT / "formal_layer" / "formal_status.md"
FORMAL_CANDIDATES_PATH = PROJECT_ROOT / "formal_layer" / "formal_candidates.csv"
FORMAL_SIGNAL_LEDGER_PATH = PROJECT_ROOT / "formal_layer" / "formal_signal_ledger.csv"
FORMAL_TRACKING_CSV_PATH = PROJECT_ROOT / "formal_layer" / "formal_candidate_tracking.csv"
FORMAL_TRACKING_MD_PATH = PROJECT_ROOT / "formal_layer" / "formal_candidate_tracking.md"
FORMAL_DAILY_REPORT_MD_PATH = PROJECT_ROOT / "formal_layer" / "formal_daily_report.md"
MAIN_MODEL_DECISION_PATH = PROJECT_ROOT / "decision_layer" / "main_model_decision.json"
MAIN_MODEL_SCORES_PATH = PROJECT_ROOT / "model_layer" / "main_model_scores.csv"
MAIN_MODEL_VALIDATION_SUMMARY_PATH = PROJECT_ROOT / "validation_layer" / "main_model_validation_summary.csv"

TRACKING_COLUMNS = [
    "as_of_date",
    "signal_date",
    "stock_id",
    "stock_name",
    "research_score",
    "daily_rank",
    "consecutive_recommendation_count",
    "buy_date",
    "buy_open",
    "target_close_plus3",
    "risk_low_minus3",
    "observed_trading_days",
    "days_remaining",
    "latest_close_return",
    "max_close_return_so_far",
    "max_close_return_date",
    "min_low_return_so_far",
    "hit_plus3_close_date",
    "hit_minus3_low_date",
    "tracking_status",
]

LEDGER_COLUMNS = [
    "signal_date",
    "stock_id",
    "stock_name",
    "original_rank",
    "research_score",
    "consecutive_recommendation_count",
    "candidate_type",
    "created_at",
    "last_updated_at",
    "as_of_date",
    "buy_date",
    "buy_open",
    "target_close_plus3",
    "risk_low_minus3",
    "observed_trading_days",
    "days_remaining",
    "latest_close_return",
    "max_close_return_so_far",
    "max_close_return_date",
    "min_low_return_so_far",
    "hit_plus3_close_date",
    "hit_minus3_low_date",
    "tracking_status",
]


def read_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def fail(message: str) -> None:
    raise SystemExit(f"FAIL: {message}")


def parse_date(value: str) -> datetime:
    value = str(value).strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%y/%m/%d"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    fail(f"cannot parse date: {value}")


def csv_stats(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if not reader.fieldnames:
            fail(f"missing header: {path}")
    date_values = [parse_date(row["日期"]) for row in rows if row.get("日期")]
    if not date_values:
        fail(f"missing date values: {path}")
    return {
        "path": path,
        "rows": len(rows),
        "columns": len(reader.fieldnames or []),
        "latest_date": max(date_values).strftime("%Y-%m-%d"),
        "first_columns": (reader.fieldnames or [])[:12],
    }


def csv_date_set(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return {parse_date(row["日期"]).strftime("%Y-%m-%d") for row in reader if row.get("日期")}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the formal main pipeline.")
    parser.add_argument(
        "--as-of-date",
        default=None,
        help="Use this completed trading date for formal output, even if raw CSVs contain newer incomplete rows.",
    )
    return parser.parse_args()


def resolve_as_of_date(raw_as_of_date: str | None, stock_stats: dict, market_stats: dict, paths: dict[str, Path]) -> str:
    if raw_as_of_date is None:
        if stock_stats["latest_date"] != market_stats["latest_date"]:
            fail(
                "stock_daily_all and market_daily latest dates do not match: "
                f"{stock_stats['latest_date']} vs {market_stats['latest_date']}. "
                "Use --as-of-date with a completed date that exists in both files."
            )
        return stock_stats["latest_date"]

    as_of_date = parse_date(raw_as_of_date).strftime("%Y-%m-%d")
    if parse_date(as_of_date) > parse_date(stock_stats["latest_date"]):
        fail(f"--as-of-date {as_of_date} is after stock latest date {stock_stats['latest_date']}")
    if parse_date(as_of_date) > parse_date(market_stats["latest_date"]):
        fail(f"--as-of-date {as_of_date} is after market latest date {market_stats['latest_date']}")

    stock_dates = csv_date_set(paths["stock_daily_all"])
    market_dates = csv_date_set(paths["market_daily"])
    if as_of_date not in stock_dates:
        fail(f"--as-of-date {as_of_date} is missing from stock_daily_all.csv")
    if as_of_date not in market_dates:
        fail(f"--as-of-date {as_of_date} is missing from market_daily.csv")
    return as_of_date


def parse_float(value: str | int | float | None) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt_float(value: str | int | float | None, digits: int = 6) -> str:
    parsed = parse_float(value)
    if parsed is None:
        return ""
    return f"{parsed:.{digits}f}"


def fmt_pct(value: str | int | float | None) -> str:
    parsed = parse_float(value)
    if parsed is None:
        return "N/A"
    return f"{parsed:.2%}"


def sort_key_date(row: dict) -> datetime:
    return parse_date(str(row["日期"]))


def validate_inputs(config: dict) -> dict[str, Path]:
    allowed = config.get("allowed_inputs", {})
    if set(allowed) != {"stock_daily_all", "market_daily", "theme_group"}:
        fail("allowed_inputs must contain exactly stock_daily_all, market_daily, theme_group")
    old_marker = "stock" + "_raw_only" + "_project"
    paths = {name: Path(value) for name, value in allowed.items()}
    for name, path in paths.items():
        if old_marker in str(path):
            fail(f"input points to old project: {name}")
        if not path.exists():
            fail(f"missing input {name}: {path}")
        if name == "theme_group" and PROJECT_ROOT not in path.parents:
            fail("theme_group must be inside this clean project")
    return paths


def write_layer_contracts(stock_stats: dict, market_stats: dict, theme_path: Path, as_of_date: str) -> None:
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    DATA_STATUS_PATH.write_text(
        "\n".join(
            [
                "# Main Pipeline Data Status",
                "",
                f"- Generated: {generated}",
                f"- Stock latest date: {stock_stats['latest_date']}",
                f"- Market latest date: {market_stats['latest_date']}",
                f"- Formal report as-of date: {as_of_date}",
                f"- Stock rows: {stock_stats['rows']}",
                f"- Market rows: {market_stats['rows']}",
                f"- Theme file: `{theme_path}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    LABEL_CONTRACT_PATH.write_text(
        "\n".join(
            [
                "# Label Contract",
                "",
                "- Signal time: after the signal-day close.",
                "- Buy assumption: next trading day open.",
                "- Success: a close in the next 1 to 10 trading days reaches +3 percent from the buy open.",
                "- Drawdown side risk: any -3 percent low is a risk label, not an automatic failure.",
                "- If price first reaches -3 percent low and later reaches +3 percent close, primary success remains true.",
                "- `risk_adjusted_10d_success` is kept only as a hard-risk comparison field.",
                "- Unfinished: if the future 10 trading day window is incomplete, the sample is tracking-only.",
                "- Same-day market comparison is validation support, not a stock label by itself.",
                "- Episode grouping prevents the same stock from being repeatedly counted as new within one short wave.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    FEATURE_CONTRACT_PATH.write_text(
        "\n".join(
            [
                "# Feature Contract",
                "",
                "Allowed features must be knowable on or before the signal day.",
                "",
                "- Stock price and volume history.",
                "- Institution and margin data history.",
                "- Day-trade data history.",
                "- Market index, volume, breadth, institution, and margin data history.",
                "- Theme group as a categorical feature or validation grouping only.",
                "",
                "Forbidden inputs include future prices, manual conclusions, old report outputs, and post-result explanations.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    MODEL_CONTRACT_PATH.write_text(
        "\n".join(
            [
                "# Main Model Contract",
                "",
                "The project has one formal model route.",
                "",
                "The model may learn four tasks inside one integrated training flow:",
                "",
                "- 10 trading day success.",
                "- Failure risk.",
                "- Same-day relative advantage.",
                "- Episode starting point.",
                "",
                "A raw model score is only a research ranking score. It can be called a success rate only after calibration passes.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    VALIDATION_CONTRACT_PATH.write_text(
        "\n".join(
            [
                "# Validation Contract",
                "",
                "A model can update formal output only after holdout validation passes.",
                "",
                "Required checks:",
                "",
                "- Same-day market baseline comparison.",
                "- Current benchmark comparison.",
                "- Candidate-region Top 3 success lift and return lift.",
                "- Score band direction as calibration diagnostics, not a hard promotion blocker.",
                "- Failure-risk band direction.",
                "- Monthly stability.",
                "- Stock and industry concentration.",
                "- Maximum Top 3 formal candidates per day.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def read_main_model_decision() -> dict:
    if not MAIN_MODEL_DECISION_PATH.exists():
        return {}
    with MAIN_MODEL_DECISION_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def approved_main_model(decision: dict) -> bool:
    return bool(
        decision
        and decision.get("formal_approved") is True
        and decision.get("status") == "passed_holdout_validation"
        and decision.get("candidate_region_validation_ok") is True
    )


def latest_model_candidates(decision: dict, latest_date: str, config: dict) -> list[dict]:
    stock_by_id = read_stock_rows(Path(config["allowed_inputs"]["stock_daily_all"]))
    score_rows = read_model_scores()
    if not score_rows:
        fail("approved main model is missing model_layer/main_model_scores.csv")
    if not any(str(row.get("日期", "")) == latest_date for row in score_rows):
        fail(
            "model_layer/main_model_scores.csv is not synced to the formal report date: "
            f"{latest_date}. Run scripts/run_main_model_training_pipeline.py before formal output."
        )
    gate = parse_float(decision.get("selected_gate"))
    replay_candidates = select_replay_candidates(score_rows, stock_by_id, latest_date, gate)
    return [row for row in replay_candidates if str(row.get("日期", "")) == latest_date]


def read_stock_rows(stock_path: Path) -> dict[str, list[dict]]:
    with stock_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    by_stock: dict[str, list[dict]] = {}
    for row in rows:
        stock_id = str(row.get("股票代號", "")).strip()
        if not stock_id:
            continue
        by_stock.setdefault(stock_id, []).append(row)
    for stock_rows in by_stock.values():
        stock_rows.sort(key=sort_key_date)
        for index, row in enumerate(stock_rows):
            row["_trading_index"] = index
    return by_stock


def read_model_scores() -> list[dict]:
    if not MAIN_MODEL_SCORES_PATH.exists():
        return []
    with MAIN_MODEL_SCORES_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["_score"] = parse_float(row.get("integrated_research_score")) or float("-inf")
    return rows


def read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path: Path, columns: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def signal_key(row: dict) -> tuple[str, str]:
    return str(row.get("signal_date", "")), str(row.get("stock_id", ""))


def filter_ledger_as_of(ledger_rows: list[dict], as_of_date: str) -> list[dict]:
    return [
        row
        for row in ledger_rows
        if not row.get("signal_date") or parse_date(str(row.get("signal_date"))) <= parse_date(as_of_date)
    ]


def candidate_to_ledger_row(candidate: dict, candidate_type: str, generated: str) -> dict:
    return {
        "signal_date": candidate.get("日期", ""),
        "stock_id": candidate.get("股票代號", ""),
        "stock_name": candidate.get("股票名稱", ""),
        "original_rank": fmt_float(candidate.get("daily_rank"), digits=0),
        "research_score": fmt_float(candidate.get("integrated_research_score")),
        "consecutive_recommendation_count": str(candidate.get("consecutive_recommendation_count", "")),
        "candidate_type": candidate_type,
        "created_at": generated,
        "last_updated_at": generated,
        "as_of_date": "",
        "buy_date": "",
        "buy_open": "",
        "target_close_plus3": "",
        "risk_low_minus3": "",
        "observed_trading_days": "0",
        "days_remaining": "10",
        "latest_close_return": "",
        "max_close_return_so_far": "",
        "max_close_return_date": "",
        "min_low_return_so_far": "",
        "hit_plus3_close_date": "",
        "hit_minus3_low_date": "",
        "tracking_status": "not_started",
    }


def select_replay_candidates(score_rows: list[dict], stock_by_id: dict[str, list[dict]], latest_date: str, gate: float | None) -> list[dict]:
    score_rows = [row for row in score_rows if str(row.get("日期", "")) <= latest_date]
    signal_dates = sorted({str(row["日期"]) for row in score_rows})
    replay_dates = set(signal_dates[-10:])
    rows_by_date: dict[str, list[dict]] = {}
    for row in score_rows:
        rows_by_date.setdefault(str(row["日期"]), []).append(row)

    selected: list[dict] = []
    last_pick_index: dict[str, int] = {}
    for signal_date in signal_dates:
        day_rows = sorted(rows_by_date.get(signal_date, []), key=lambda row: row["_score"], reverse=True)
        if not day_rows:
            continue
        if gate is not None and day_rows[0]["_score"] < gate:
            continue
        selected_today = 0
        for row in day_rows:
            stock_id = str(row.get("股票代號", "")).strip()
            stock_rows = stock_by_id.get(stock_id, [])
            price_row = next((item for item in stock_rows if item.get("日期") == signal_date), None)
            if not price_row:
                continue
            stock_index = int(price_row["_trading_index"])
            if stock_id in last_pick_index and stock_index - last_pick_index[stock_id] <= 10:
                continue
            last_pick_index[stock_id] = stock_index
            selected_today += 1
            if signal_date in replay_dates:
                selected.append(row)
            if selected_today >= 3:
                break
    return selected


def track_signal_record(record: dict, stock_by_id: dict[str, list[dict]], as_of_date: str, generated: str) -> dict:
    stock_id = str(record.get("stock_id", "")).strip()
    signal_date = str(record.get("signal_date", ""))
    stock_rows = stock_by_id.get(stock_id, [])
    signal_row = next((row for row in stock_rows if row.get("日期") == signal_date), None)
    out = {col: record.get(col, "") for col in LEDGER_COLUMNS}
    out["as_of_date"] = as_of_date
    out["last_updated_at"] = generated
    if not signal_row:
        out["tracking_status"] = "missing_signal_price"
        return out
    next_index = int(signal_row["_trading_index"]) + 1
    if next_index >= len(stock_rows):
        status = "not_started" if as_of_date <= signal_date else "missing_buy_price"
        out.update({"buy_date": "", "buy_open": "", "observed_trading_days": "0", "days_remaining": "10", "tracking_status": status})
        return out
    buy_row = stock_rows[next_index]
    buy_open = parse_float(buy_row.get("開盤價"))
    if buy_open is None or buy_open <= 0:
        buy_date = str(buy_row.get("日期", ""))
        status = "not_started" if buy_date and buy_date > as_of_date else "missing_buy_price"
        out.update({"buy_date": buy_date, "buy_open": "", "observed_trading_days": "0", "days_remaining": "10", "tracking_status": status})
        return out
    future_rows = [
        row
        for row in stock_rows[next_index : next_index + 10]
        if str(row.get("日期", "")) <= as_of_date
    ]
    target_close = buy_open * 1.03
    risk_low = buy_open * 0.97
    success_date = ""
    drawdown_date = ""
    max_close_return = None
    max_close_return_date = ""
    min_low_return = None
    latest_close_return = None
    for row in future_rows:
        close = parse_float(row.get("收盤價"))
        low = parse_float(row.get("最低價"))
        if close is not None:
            close_return = close / buy_open - 1.0
            if max_close_return is None or close_return > max_close_return:
                max_close_return = close_return
                max_close_return_date = str(row.get("日期", ""))
            latest_close_return = close_return
            if not success_date and close >= target_close:
                success_date = str(row.get("日期", ""))
        if low is not None:
            low_return = low / buy_open - 1.0
            min_low_return = low_return if min_low_return is None else min(min_low_return, low_return)
            if not drawdown_date and low <= risk_low:
                drawdown_date = str(row.get("日期", ""))
    observed_days = len(future_rows)
    if success_date:
        status = "success"
    elif observed_days >= 10:
        status = "failure"
    else:
        status = "tracking"
    out.update(
        {
            "buy_date": buy_row.get("日期", ""),
            "buy_open": fmt_float(buy_open),
            "target_close_plus3": fmt_float(target_close),
            "risk_low_minus3": fmt_float(risk_low),
            "observed_trading_days": str(observed_days),
            "days_remaining": str(max(0, 10 - observed_days)),
            "latest_close_return": fmt_float(latest_close_return),
            "max_close_return_so_far": fmt_float(max_close_return),
            "max_close_return_date": max_close_return_date,
            "min_low_return_so_far": fmt_float(min_low_return),
            "hit_plus3_close_date": success_date,
            "hit_minus3_low_date": drawdown_date,
            "tracking_status": status,
        }
    )
    return out


def tracking_view_from_ledger(ledger_rows: list[dict]) -> list[dict]:
    return [
        {
            "as_of_date": row.get("as_of_date", ""),
            "signal_date": row.get("signal_date", ""),
            "stock_id": row.get("stock_id", ""),
            "stock_name": row.get("stock_name", ""),
            "research_score": row.get("research_score", ""),
            "daily_rank": row.get("original_rank", ""),
            "consecutive_recommendation_count": row.get("consecutive_recommendation_count", ""),
            "buy_date": row.get("buy_date", ""),
            "buy_open": row.get("buy_open", ""),
            "target_close_plus3": row.get("target_close_plus3", ""),
            "risk_low_minus3": row.get("risk_low_minus3", ""),
            "observed_trading_days": row.get("observed_trading_days", ""),
            "days_remaining": row.get("days_remaining", ""),
            "latest_close_return": row.get("latest_close_return", ""),
            "max_close_return_so_far": row.get("max_close_return_so_far", ""),
            "max_close_return_date": row.get("max_close_return_date", ""),
            "min_low_return_so_far": row.get("min_low_return_so_far", ""),
            "hit_plus3_close_date": row.get("hit_plus3_close_date", ""),
            "hit_minus3_low_date": row.get("hit_minus3_low_date", ""),
            "tracking_status": row.get("tracking_status", ""),
        }
        for row in ledger_rows
    ]


def merge_signal_ledger(
    config: dict,
    latest_date: str,
    decision: dict,
    today_candidates: list[dict],
    generated: str,
) -> tuple[list[dict], list[dict]]:
    stock_by_id = read_stock_rows(Path(config["allowed_inputs"]["stock_daily_all"]))
    score_rows = read_model_scores()
    top_sets = raw_top_stock_sets(limit=10)
    gate = parse_float(decision.get("selected_gate"))
    ledger_rows = filter_ledger_as_of(read_csv_rows(FORMAL_SIGNAL_LEDGER_PATH), latest_date)
    existing = {signal_key(row) for row in ledger_rows}
    additions: list[dict] = []

    if not ledger_rows:
        for candidate in select_replay_candidates(score_rows, stock_by_id, latest_date, gate):
            candidate_type = "new_formal" if candidate.get("日期") == latest_date else "replay_seed"
            row = candidate_to_ledger_row(candidate, candidate_type, generated)
            if signal_key(row) not in existing:
                additions.append(row)
                existing.add(signal_key(row))
    else:
        for candidate in today_candidates:
            row = candidate_to_ledger_row(candidate, "new_formal", generated)
            if signal_key(row) not in existing:
                additions.append(row)
                existing.add(signal_key(row))

    ledger_rows.extend(additions)
    update_ledger_consecutive_counts(ledger_rows, latest_date, top_sets)
    updated = [track_signal_record(row, stock_by_id, latest_date, generated) for row in ledger_rows]
    updated.sort(key=lambda row: (row.get("signal_date", ""), int(float(row.get("original_rank") or 999999)), row.get("stock_id", "")))
    write_csv_rows(FORMAL_SIGNAL_LEDGER_PATH, LEDGER_COLUMNS, updated)
    return updated, additions


def write_tracking_outputs(latest_date: str, ledger_rows: list[dict]) -> None:
    tracking_rows = tracking_view_from_ledger(ledger_rows)
    write_csv_rows(FORMAL_TRACKING_CSV_PATH, TRACKING_COLUMNS, tracking_rows)
    status_counts: dict[str, int] = {}
    for row in tracking_rows:
        status_counts[row.get("tracking_status", "unknown")] = status_counts.get(row.get("tracking_status", "unknown"), 0) + 1
    lines = [
        "# Formal Candidate Tracking Replay",
        "",
        f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- As-of date: {latest_date}",
        "- Scope: formal signal ledger.",
        "- No-lookahead rule: signal rows are locked when created; only tracking fields are updated afterward.",
        "- Buy assumption: next trading day open.",
        "- Success rule: within the next 10 trading days, any close reaches buy open +3%.",
        "- Drawdown: -3% low is tracked as risk context, not automatic failure.",
        "- research_score is not a calibrated probability.",
        "",
        "## Status Counts",
        "",
    ]
    if status_counts:
        for status, count in sorted(status_counts.items()):
            lines.append(f"- {status}: {count}")
    else:
        lines.append("- no replay candidates")
    lines.extend(["", "## Files", "", f"- `{FORMAL_TRACKING_CSV_PATH.relative_to(PROJECT_ROOT)}`", ""])
    FORMAL_TRACKING_MD_PATH.write_text("\n".join(lines), encoding="utf-8")


def write_empty_tracking(latest_date: str, reason: str) -> None:
    with FORMAL_TRACKING_CSV_PATH.open("w", encoding="utf-8", newline="") as f:
        csv.DictWriter(f, fieldnames=TRACKING_COLUMNS).writeheader()
    lines = [
        "# Formal Candidate Tracking Replay",
        "",
        f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- As-of date: {latest_date}",
        f"- Status: empty",
        f"- Reason: {reason}",
        "",
    ]
    FORMAL_TRACKING_MD_PATH.write_text("\n".join(lines), encoding="utf-8")


def latest_raw_top_rows(latest_date: str, limit: int = 10) -> list[dict]:
    rows = [row for row in read_model_scores() if str(row.get("日期", "")) == latest_date]
    rows.sort(key=lambda row: row["_score"], reverse=True)
    return rows[:limit]


def raw_top_stock_sets(limit: int = 10) -> dict[str, set[str]]:
    by_date: dict[str, list[dict]] = {}
    for row in read_model_scores():
        by_date.setdefault(str(row.get("日期", "")), []).append(row)
    top_sets: dict[str, set[str]] = {}
    for date, rows in by_date.items():
        rows.sort(key=lambda row: row["_score"], reverse=True)
        top_sets[date] = {str(row.get("股票代號", "")) for row in rows[:limit]}
    return top_sets


def score_dates(top_sets: dict[str, set[str]]) -> list[str]:
    return sorted(top_sets, key=parse_date)


def consecutive_recommendation_count(stock_id: str, signal_date: str, as_of_date: str, top_sets: dict[str, set[str]]) -> int:
    dates = score_dates(top_sets)
    if signal_date not in top_sets:
        return 1
    signal_index = dates.index(signal_date)
    count = 0
    for date in dates[signal_index:]:
        if date > as_of_date:
            break
        if dates.index(date) - signal_index > 10:
            break
        if stock_id in top_sets.get(date, set()):
            count += 1
    return max(1, count)


def attach_consecutive_counts(rows: list[dict], signal_date: str, top_sets: dict[str, set[str]]) -> None:
    for row in rows:
        row["consecutive_recommendation_count"] = 1


def update_ledger_consecutive_counts(ledger_rows: list[dict], as_of_date: str, top_sets: dict[str, set[str]]) -> None:
    for row in ledger_rows:
        stock_id = str(row.get("stock_id", ""))
        signal_date = str(row.get("signal_date", ""))
        row["consecutive_recommendation_count"] = str(consecutive_recommendation_count(stock_id, signal_date, as_of_date, top_sets))


def find_recent_ledger_row(raw_row: dict, ledger_rows: list[dict], stock_by_id: dict[str, list[dict]], latest_date: str) -> dict | None:
    stock_id = str(raw_row.get("股票代號", ""))
    stock_rows = stock_by_id.get(stock_id, [])
    latest_price_row = next((row for row in stock_rows if row.get("日期") == latest_date), None)
    if not latest_price_row:
        return None
    latest_index = int(latest_price_row["_trading_index"])
    matches = []
    for row in ledger_rows:
        if row.get("stock_id") != stock_id or row.get("signal_date") == latest_date:
            continue
        signal_price_row = next((item for item in stock_rows if item.get("日期") == row.get("signal_date")), None)
        if not signal_price_row:
            continue
        distance = latest_index - int(signal_price_row["_trading_index"])
        if 0 < distance <= 10:
            matches.append((distance, row))
    if not matches:
        return None
    matches.sort(key=lambda item: item[0])
    return matches[0][1]


def continuation_rows(config: dict, latest_date: str, ledger_rows: list[dict], today_candidates: list[dict]) -> list[dict]:
    stock_by_id = read_stock_rows(Path(config["allowed_inputs"]["stock_daily_all"]))
    today_keys = {str(row.get("股票代號", "")) for row in today_candidates}
    top_sets = raw_top_stock_sets(limit=10)
    rows: list[dict] = []
    for raw in latest_raw_top_rows(latest_date, limit=10):
        stock_id = str(raw.get("股票代號", ""))
        if stock_id in today_keys:
            continue
        prior = find_recent_ledger_row(raw, ledger_rows, stock_by_id, latest_date)
        if not prior:
            continue
        rows.append(
            {
                "stock_id": stock_id,
                "stock_name": raw.get("股票名稱", ""),
                "raw_rank": fmt_float(raw.get("daily_rank"), digits=0),
                "research_score": fmt_float(raw.get("integrated_research_score")),
                "consecutive_recommendation_count": prior.get("consecutive_recommendation_count", ""),
                "previous_signal_date": prior.get("signal_date", ""),
                "previous_tracking_status": prior.get("tracking_status", ""),
                "max_close_return_so_far": prior.get("max_close_return_so_far", ""),
                "max_close_return_date": prior.get("max_close_return_date", ""),
            }
        )
    return rows


def pct_text(value: str) -> str:
    parsed = parse_float(value)
    if parsed is None:
        return ""
    return f"{parsed:.2%}"


def md_table(columns: list[str], rows: list[dict]) -> list[str]:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    if not rows:
        lines.append("| " + " | ".join([""] * len(columns)) + " |")
        return lines
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return lines


def write_daily_report(latest_date: str, today_candidates: list[dict], continuations: list[dict], ledger_rows: list[dict]) -> None:
    new_rows = [
        {
            "股票": f"{row.get('股票代號', '')} {row.get('股票名稱', '')}",
            "原始排名": fmt_float(row.get("daily_rank"), digits=0),
            "research_score": fmt_float(row.get("integrated_research_score")),
            "連續被推薦次數": row.get("consecutive_recommendation_count", ""),
            "類型": "新進正式候選",
        }
        for row in today_candidates
    ]
    continuation_display = [
        {
            "股票": f"{row.get('stock_id', '')} {row.get('stock_name', '')}",
            "原始排名": row.get("raw_rank", ""),
            "research_score": row.get("research_score", ""),
            "連續被推薦次數": row.get("consecutive_recommendation_count", ""),
            "前次訊號日": row.get("previous_signal_date", ""),
            "追蹤狀態": row.get("previous_tracking_status", ""),
            "目前最高收盤報酬": pct_text(row.get("max_close_return_so_far", "")),
            "最高收盤報酬日期": row.get("max_close_return_date", ""),
        }
        for row in continuations
    ]
    tracking_display = [
        {
            "訊號日": row.get("signal_date", ""),
            "股票": f"{row.get('stock_id', '')} {row.get('stock_name', '')}",
            "連續被推薦次數": row.get("consecutive_recommendation_count", ""),
            "買入日": row.get("buy_date", ""),
            "已追蹤日": row.get("observed_trading_days", ""),
            "最高收盤報酬": pct_text(row.get("max_close_return_so_far", "")),
            "最高收盤報酬日期": row.get("max_close_return_date", ""),
            "+3%達成日": row.get("hit_plus3_close_date", ""),
            "-3%風險日": row.get("hit_minus3_low_date", ""),
            "狀態": row.get("tracking_status", ""),
        }
        for row in ledger_rows
    ]
    lines = [
        "# Formal Daily Report",
        "",
        f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Report date: {latest_date}",
        "- research_score 是排序分數，不是機率。",
        "- strategy_backtest_hit_rate 是策略歷史回測成功率，不是個股成功率。",
        "",
        "## 今日新進正式候選",
        "",
        *md_table(["股票", "原始排名", "research_score", "連續被推薦次數", "類型"], new_rows),
        "",
        "## 高分續強但已追蹤",
        "",
        *md_table(["股票", "原始排名", "research_score", "連續被推薦次數", "前次訊號日", "追蹤狀態", "目前最高收盤報酬", "最高收盤報酬日期"], continuation_display),
        "",
        "## 正式候選追蹤",
        "",
        *md_table(["訊號日", "股票", "連續被推薦次數", "買入日", "已追蹤日", "最高收盤報酬", "最高收盤報酬日期", "+3%達成日", "-3%風險日", "狀態"], tracking_display),
        "",
    ]
    FORMAL_DAILY_REPORT_MD_PATH.write_text("\n".join(lines), encoding="utf-8")


def refresh_existing_ledger(config: dict, latest_date: str, generated: str) -> list[dict]:
    ledger_rows = filter_ledger_as_of(read_csv_rows(FORMAL_SIGNAL_LEDGER_PATH), latest_date)
    if not ledger_rows:
        return []
    stock_by_id = read_stock_rows(Path(config["allowed_inputs"]["stock_daily_all"]))
    update_ledger_consecutive_counts(ledger_rows, latest_date, raw_top_stock_sets(limit=10))
    updated = [track_signal_record(row, stock_by_id, latest_date, generated) for row in ledger_rows]
    updated.sort(key=lambda row: (row.get("signal_date", ""), int(float(row.get("original_rank") or 999999)), row.get("stock_id", "")))
    write_csv_rows(FORMAL_SIGNAL_LEDGER_PATH, LEDGER_COLUMNS, updated)
    return updated


def main_model_holdout_summary() -> dict:
    if not MAIN_MODEL_VALIDATION_SUMMARY_PATH.exists():
        return {}
    with MAIN_MODEL_VALIDATION_SUMMARY_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("split") == "holdout" and row.get("strategy") == "integrated_main_top3":
                return row
    return {}


def write_formal_files(
    config: dict,
    latest_date: str,
    reason: str,
    decision: dict | None = None,
    candidates: list[dict] | None = None,
    raw_stock_latest_date: str | None = None,
    raw_market_latest_date: str | None = None,
) -> None:
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    active = bool(candidates)
    if candidates:
        attach_consecutive_counts(candidates, latest_date, raw_top_stock_sets(limit=10))
    current_strategy = "single integrated main model Top3" if active else "retained benchmark"
    result = "正式候選已產生" if active else config["formal_candidate_default"]
    FORMAL_STATUS_PATH.write_text(
        "\n".join(
            [
                "# Formal Status",
                "",
                f"- Generated: {generated}",
                "- Status: active",
                "- Formal source: `scripts/run_main_pipeline.py`",
                f"- Current strategy: {current_strategy}",
                f"- Raw data latest date: stock={raw_stock_latest_date or latest_date}; market={raw_market_latest_date or latest_date}",
                f"- Formal report as-of date: {latest_date}",
                f"- Result: {result}",
                f"- Reason: {reason}",
                "- Rule: training outputs cannot update formal candidates directly.",
                "- Score note: research_score is a ranking score, not a calibrated probability.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    with FORMAL_CANDIDATES_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "date",
                "stock_id",
                "stock_name",
                "output_type",
                "research_score",
                "consecutive_recommendation_count",
                "calibrated_success_rate",
                "calibration_sample_count",
                "strategy_backtest_hit_rate",
                "avg_10d_high_close_return",
                "main_basis",
                "main_risk",
                "tracking_status",
            ]
        )
        if active:
            assert decision is not None
            holdout_summary = main_model_holdout_summary()
            basis = (
                "candidate-region Top3 passed holdout: "
                f"success_lift={fmt_pct(decision.get('holdout_success_lift'))}, "
                f"return_lift={fmt_pct(decision.get('holdout_return_lift'))}"
            )
            risk = (
                "research score is not calibrated probability; latest signal is tracking-only"
            )
            if decision.get("score_order_ok") is False:
                risk += "; all-row score-band ordering remains diagnostic warning"
            for row in candidates or []:
                writer.writerow(
                    [
                        latest_date,
                        row.get("股票代號", ""),
                        row.get("股票名稱", ""),
                        "formal_candidate",
                        fmt_float(row.get("integrated_research_score")),
                        row.get("consecutive_recommendation_count", ""),
                        "",
                        "",
                        fmt_float(decision.get("holdout_success_rate")),
                        fmt_float(holdout_summary.get("avg_10d_high_close_return")),
                        basis,
                        risk,
                        "tracking",
                    ]
                )
    if approved_main_model(decision or {}):
        ledger_rows, _ = merge_signal_ledger(config, latest_date, decision or {}, candidates or [], generated)
        continuations = continuation_rows(config, latest_date, ledger_rows, candidates or [])
        write_tracking_outputs(latest_date, ledger_rows)
        write_daily_report(latest_date, candidates or [], continuations, ledger_rows)
    else:
        ledger_rows = refresh_existing_ledger(config, latest_date, generated)
        if ledger_rows:
            write_tracking_outputs(latest_date, ledger_rows)
            write_daily_report(latest_date, [], [], ledger_rows)
        else:
            write_empty_tracking(latest_date, "main model is not approved for formal tracking replay")
            write_daily_report(latest_date, [], [], [])


def main() -> None:
    args = parse_args()
    config = read_config()
    paths = validate_inputs(config)
    stock_stats = csv_stats(paths["stock_daily_all"])
    market_stats = csv_stats(paths["market_daily"])
    latest_date = resolve_as_of_date(args.as_of_date, stock_stats, market_stats, paths)
    write_layer_contracts(stock_stats, market_stats, paths["theme_group"], latest_date)
    main_model_decision = read_main_model_decision()
    if approved_main_model(main_model_decision):
        candidates = latest_model_candidates(main_model_decision, latest_date, config)
        if candidates:
            reason = "single main model passed candidate-region holdout validation"
        else:
            reason = "main model passed validation, but latest date did not pass the selected score gate"
        write_formal_files(
            config,
            latest_date,
            reason,
            main_model_decision,
            candidates,
            stock_stats["latest_date"],
            market_stats["latest_date"],
        )
        decision_line = "promote validated single main model."
        formal_result = "formal candidates written" if candidates else config["formal_candidate_default"]
    else:
        reason = (
            "architecture reset completed; no new integrated model is promoted until "
            "the single main model passes validation"
        )
        write_formal_files(
            config,
            latest_date,
            reason,
            raw_stock_latest_date=stock_stats["latest_date"],
            raw_market_latest_date=market_stats["latest_date"],
        )
        decision_line = "keep current formal benchmark."
        formal_result = config["formal_candidate_default"]
    DECISION_PATH.write_text(
        "\n".join(
            [
                "# Main Pipeline Decision",
                "",
                f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"- Raw data latest date: stock={stock_stats['latest_date']}; market={market_stats['latest_date']}",
                f"- Formal report as-of date: {latest_date}",
                f"- Decision: {decision_line}",
                f"- Formal result: {formal_result}",
                "- Next allowed work: track formal candidates; retrain only through the single main model pipeline.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print("OK: main pipeline completed")
    print(f"RAW_STOCK_LATEST_DATE: {stock_stats['latest_date']}")
    print(f"RAW_MARKET_LATEST_DATE: {market_stats['latest_date']}")
    print(f"AS_OF_DATE: {latest_date}")
    print(f"FORMAL_STATUS: {FORMAL_STATUS_PATH}")
    print(f"FORMAL_CANDIDATES: {FORMAL_CANDIDATES_PATH}")


if __name__ == "__main__":
    main()
