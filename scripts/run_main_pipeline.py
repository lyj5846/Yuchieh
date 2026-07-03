from __future__ import annotations

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
FORMAL_TRACKING_CSV_PATH = PROJECT_ROOT / "formal_layer" / "formal_candidate_tracking.csv"
FORMAL_TRACKING_MD_PATH = PROJECT_ROOT / "formal_layer" / "formal_candidate_tracking.md"
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
    "buy_date",
    "buy_open",
    "target_close_plus3",
    "risk_low_minus3",
    "observed_trading_days",
    "days_remaining",
    "latest_close_return",
    "max_close_return_so_far",
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


def write_layer_contracts(stock_stats: dict, market_stats: dict, theme_path: Path) -> None:
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    DATA_STATUS_PATH.write_text(
        "\n".join(
            [
                "# Main Pipeline Data Status",
                "",
                f"- Generated: {generated}",
                f"- Stock latest date: {stock_stats['latest_date']}",
                f"- Market latest date: {market_stats['latest_date']}",
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


def track_one_candidate(candidate: dict, stock_by_id: dict[str, list[dict]], as_of_date: str) -> dict:
    stock_id = str(candidate.get("股票代號", "")).strip()
    signal_date = str(candidate.get("日期", ""))
    stock_rows = stock_by_id.get(stock_id, [])
    signal_row = next((row for row in stock_rows if row.get("日期") == signal_date), None)
    base = {
        "as_of_date": as_of_date,
        "signal_date": signal_date,
        "stock_id": stock_id,
        "stock_name": candidate.get("股票名稱", ""),
        "research_score": fmt_float(candidate.get("integrated_research_score")),
        "daily_rank": fmt_float(candidate.get("daily_rank"), digits=0),
    }
    if not signal_row:
        return {**base, "tracking_status": "missing_signal_price"}
    next_index = int(signal_row["_trading_index"]) + 1
    if next_index >= len(stock_rows):
        return {
            **base,
            "buy_date": "",
            "buy_open": "",
            "observed_trading_days": 0,
            "days_remaining": 10,
            "tracking_status": "not_started",
        }
    buy_row = stock_rows[next_index]
    buy_open = parse_float(buy_row.get("開盤價"))
    if buy_open is None or buy_open <= 0:
        return {
            **base,
            "buy_date": buy_row.get("日期", ""),
            "buy_open": "",
            "observed_trading_days": 0,
            "days_remaining": 10,
            "tracking_status": "not_started",
        }
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
    min_low_return = None
    latest_close_return = None
    for row in future_rows:
        close = parse_float(row.get("收盤價"))
        low = parse_float(row.get("最低價"))
        if close is not None:
            close_return = close / buy_open - 1.0
            max_close_return = close_return if max_close_return is None else max(max_close_return, close_return)
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
    return {
        **base,
        "buy_date": buy_row.get("日期", ""),
        "buy_open": fmt_float(buy_open),
        "target_close_plus3": fmt_float(target_close),
        "risk_low_minus3": fmt_float(risk_low),
        "observed_trading_days": observed_days,
        "days_remaining": max(0, 10 - observed_days),
        "latest_close_return": fmt_float(latest_close_return),
        "max_close_return_so_far": fmt_float(max_close_return),
        "min_low_return_so_far": fmt_float(min_low_return),
        "hit_plus3_close_date": success_date,
        "hit_minus3_low_date": drawdown_date,
        "tracking_status": status,
    }


def write_tracking_replay(config: dict, latest_date: str, decision: dict) -> None:
    stock_by_id = read_stock_rows(Path(config["allowed_inputs"]["stock_daily_all"]))
    score_rows = read_model_scores()
    gate = parse_float(decision.get("selected_gate"))
    replay_candidates = select_replay_candidates(score_rows, stock_by_id, latest_date, gate)
    tracking_rows = [track_one_candidate(row, stock_by_id, latest_date) for row in replay_candidates]
    with FORMAL_TRACKING_CSV_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TRACKING_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(tracking_rows)

    status_counts: dict[str, int] = {}
    for row in tracking_rows:
        status_counts[row.get("tracking_status", "unknown")] = status_counts.get(row.get("tracking_status", "unknown"), 0) + 1
    lines = [
        "# Formal Candidate Tracking Replay",
        "",
        f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- As-of date: {latest_date}",
        "- Scope: last 10 signal dates from the formal main-model score file.",
        "- No-lookahead rule: candidates are selected by signal-day research score and the selected gate; outcomes are checked only after selection.",
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


def main_model_holdout_summary() -> dict:
    if not MAIN_MODEL_VALIDATION_SUMMARY_PATH.exists():
        return {}
    with MAIN_MODEL_VALIDATION_SUMMARY_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("split") == "holdout" and row.get("strategy") == "integrated_main_top3":
                return row
    return {}


def write_formal_files(config: dict, latest_date: str, reason: str, decision: dict | None = None, candidates: list[dict] | None = None) -> None:
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    active = bool(candidates)
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
                f"- Raw data latest date: {latest_date}",
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
        write_tracking_replay(config, latest_date, decision or {})
    else:
        write_empty_tracking(latest_date, "main model is not approved for formal tracking replay")


def main() -> None:
    config = read_config()
    paths = validate_inputs(config)
    stock_stats = csv_stats(paths["stock_daily_all"])
    market_stats = csv_stats(paths["market_daily"])
    if stock_stats["latest_date"] != market_stats["latest_date"]:
        fail(
            "stock_daily_all and market_daily latest dates do not match: "
            f"{stock_stats['latest_date']} vs {market_stats['latest_date']}"
        )

    latest_date = stock_stats["latest_date"]
    write_layer_contracts(stock_stats, market_stats, paths["theme_group"])
    main_model_decision = read_main_model_decision()
    if approved_main_model(main_model_decision):
        candidates = latest_model_candidates(main_model_decision, latest_date, config)
        if candidates:
            reason = "single main model passed candidate-region holdout validation"
        else:
            reason = "main model passed validation, but latest date did not pass the selected score gate"
        write_formal_files(config, latest_date, reason, main_model_decision, candidates)
        decision_line = "promote validated single main model."
        formal_result = "formal candidates written" if candidates else config["formal_candidate_default"]
    else:
        reason = (
            "architecture reset completed; no new integrated model is promoted until "
            "the single main model passes validation"
        )
        write_formal_files(config, latest_date, reason)
        decision_line = "keep current formal benchmark."
        formal_result = config["formal_candidate_default"]
    DECISION_PATH.write_text(
        "\n".join(
            [
                "# Main Pipeline Decision",
                "",
                f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"- Raw data latest date: {latest_date}",
                f"- Decision: {decision_line}",
                f"- Formal result: {formal_result}",
                "- Next allowed work: track formal candidates; retrain only through the single main model pipeline.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print("OK: main pipeline completed")
    print(f"LATEST_DATE: {latest_date}")
    print(f"FORMAL_STATUS: {FORMAL_STATUS_PATH}")
    print(f"FORMAL_CANDIDATES: {FORMAL_CANDIDATES_PATH}")


if __name__ == "__main__":
    main()
