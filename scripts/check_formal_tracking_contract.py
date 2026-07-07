from __future__ import annotations

import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FORMAL_DIR = PROJECT_ROOT / ("formal" + "_layer")

FORMAL_CANDIDATES_PATH = FORMAL_DIR / "formal_candidates.csv"
LEDGER_PATH = FORMAL_DIR / "formal_signal_ledger.csv"
TRACKING_CSV_PATH = FORMAL_DIR / "formal_candidate_tracking.csv"
TRACKING_MD_PATH = FORMAL_DIR / "formal_candidate_tracking.md"
DAILY_REPORT_PATH = FORMAL_DIR / "formal_daily_report.md"

REQUIRED_LEDGER_COLUMNS = {
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
}

REQUIRED_TRACKING_COLUMNS = {
    "as_of_date",
    "signal_date",
    "stock_id",
    "stock_name",
    "research_score",
    "daily_rank",
    "consecutive_recommendation_count",
    "buy_date",
    "buy_open",
    "max_close_return_so_far",
    "max_close_return_date",
    "tracking_status",
}

ALLOWED_TRACKING_STATUS = {
    "not_started",
    "tracking",
    "success",
    "failure",
    "missing_signal_price",
    "missing_buy_price",
}


def fail(message: str) -> None:
    raise SystemExit(f"FAIL: {message}")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        return next(reader, [])


def row_key(row: dict[str, str]) -> tuple[str, str]:
    return row.get("signal_date", ""), row.get("stock_id", "")


def report_stock_text(row: dict[str, str]) -> str:
    return f"{row.get('stock_id', '')} {row.get('stock_name', '')}".strip()


def require_files() -> list[str]:
    issues: list[str] = []
    for path in [
        FORMAL_CANDIDATES_PATH,
        LEDGER_PATH,
        TRACKING_CSV_PATH,
        TRACKING_MD_PATH,
        DAILY_REPORT_PATH,
    ]:
        if not path.exists():
            issues.append(f"missing file: {path.relative_to(PROJECT_ROOT)}")
    return issues


def check_formal_candidates(ledger_rows: list[dict[str, str]]) -> list[str]:
    issues: list[str] = []
    header = read_header(FORMAL_CANDIDATES_PATH)
    rows = read_csv(FORMAL_CANDIDATES_PATH)
    columns = set(header)
    if "strategy_backtest_hit_rate" not in columns:
        issues.append("formal_candidates.csv must include strategy_backtest_hit_rate")
    if "consecutive_recommendation_count" not in columns:
        issues.append("formal_candidates.csv must include consecutive_recommendation_count")
    if "max_close_return_date" in columns:
        issues.append("formal_candidates.csv should not include max_close_return_date; it is a tracking-only field")
    if "actual_hit_rate" in columns:
        issues.append("formal_candidates.csv must not use actual_hit_rate")
    if "research_score" in header and "consecutive_recommendation_count" in header:
        if header.index("consecutive_recommendation_count") != header.index("research_score") + 1:
            issues.append("consecutive_recommendation_count must appear immediately after research_score")

    if not rows:
        return issues

    latest_candidate_date = max(row.get("date", "") for row in rows if row.get("date"))
    candidate_stocks = {row.get("stock_id", "") for row in rows if row.get("date") == latest_candidate_date}
    ledger_latest_stocks = {
        row.get("stock_id", "")
        for row in ledger_rows
        if row.get("signal_date") == latest_candidate_date and row.get("candidate_type") == "new_formal"
    }
    if candidate_stocks != ledger_latest_stocks:
        issues.append(
            f"formal candidates do not match latest new_formal ledger rows: "
            f"candidates={sorted(candidate_stocks)} ledger={sorted(ledger_latest_stocks)}"
        )
    for row in rows:
        if row.get("tracking_status") != "tracking":
            issues.append(f"formal candidate {row.get('date')} {row.get('stock_id')} should be tracking")
        count = row.get("consecutive_recommendation_count", "")
        if not count.isdigit() or int(count) < 1:
            issues.append(f"formal candidate {row.get('stock_id')} must have positive consecutive_recommendation_count")
    return issues


def check_ledger() -> tuple[list[str], list[dict[str, str]]]:
    issues: list[str] = []
    rows = read_csv(LEDGER_PATH)
    if not rows:
        issues.append("formal_signal_ledger.csv is empty")
        return issues, rows

    columns = set(rows[0])
    missing = REQUIRED_LEDGER_COLUMNS - columns
    if missing:
        issues.append(f"ledger missing columns: {sorted(missing)}")

    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = row_key(row)
        if key in seen:
            issues.append(f"duplicate ledger signal: {key[0]} {key[1]}")
        seen.add(key)

        status = row.get("tracking_status", "")
        if status not in ALLOWED_TRACKING_STATUS:
            issues.append(f"invalid tracking_status for {key[0]} {key[1]}: {status}")
        count = row.get("consecutive_recommendation_count", "")
        if not count.isdigit() or int(count) < 1:
            issues.append(f"invalid consecutive_recommendation_count for {key[0]} {key[1]}: {count}")
        try:
            original_rank = float(row.get("original_rank", ""))
        except ValueError:
            original_rank = 999999.0
        if original_rank > 10:
            issues.append(f"formal ledger row must not use raw rank > 10: {key[0]} {key[1]} rank={row.get('original_rank')}")
        if row.get("max_close_return_so_far") and not row.get("max_close_return_date"):
            issues.append(f"ledger row {key[0]} {key[1]} has max return but no max_close_return_date")
        if row.get("tracking_status") == "not_started" and row.get("buy_date"):
            issues.append(f"not_started ledger row should not have buy_date: {key[0]} {key[1]}")
    return issues, rows


def check_tracking_csv(ledger_rows: list[dict[str, str]]) -> list[str]:
    issues: list[str] = []
    rows = read_csv(TRACKING_CSV_PATH)
    if not rows:
        issues.append("formal_candidate_tracking.csv is empty")
        return issues
    columns = set(rows[0])
    missing = REQUIRED_TRACKING_COLUMNS - columns
    if missing:
        issues.append(f"formal_candidate_tracking.csv missing columns: {sorted(missing)}")
    if {row_key(row) for row in rows} != {row_key(row) for row in ledger_rows}:
        issues.append("tracking CSV signal set must match ledger signal set")
    for row in rows:
        count = row.get("consecutive_recommendation_count", "")
        if not count.isdigit() or int(count) < 1:
            issues.append(f"tracking row {row.get('signal_date')} {row.get('stock_id')} must have positive consecutive_recommendation_count")
        if row.get("max_close_return_so_far") and not row.get("max_close_return_date"):
            issues.append(f"tracking row {row.get('signal_date')} {row.get('stock_id')} has max return but no max_close_return_date")
    return issues


def check_daily_report(ledger_rows: list[dict[str, str]]) -> list[str]:
    issues: list[str] = []
    text = DAILY_REPORT_PATH.read_text(encoding="utf-8")
    required_text = [
        "今日新進正式候選",
        "高分續強但已追蹤",
        "正式候選追蹤",
        "已結案失敗學習表",
        "research_score 是排序分數，不是機率",
        "strategy_backtest_hit_rate 是策略歷史回測成功率，不是個股成功率",
        "連續被推薦次數",
        "最高收盤報酬日期",
        "客觀失敗特徵",
        "是否納入後續學習",
    ]
    for needle in required_text:
        if needle not in text:
            issues.append(f"daily report missing text: {needle}")
    if text.count("連續被推薦次數") < 3:
        issues.append("daily report must show 連續被推薦次數 in new candidates, continuations, and tracking sections")
    latest_date = max((row.get("signal_date", "") for row in ledger_rows), default="")
    latest_new_rows = [
        row for row in ledger_rows if row.get("signal_date") == latest_date and row.get("candidate_type") == "new_formal"
    ]
    for row in latest_new_rows:
        stock_text = report_stock_text(row)
        if stock_text and stock_text not in text:
            issues.append(f"daily report missing latest candidate: {stock_text}")
    if not latest_new_rows and "raw Top10 內無未追蹤候選，raw 11+ 不補正式候選" not in text:
        issues.append("daily report must explain why no raw 11+ candidates are used when latest formal candidates are empty")
    for row in ledger_rows:
        snippet = f"| {row.get('signal_date')} | {report_stock_text(row)} |"
        if snippet not in text:
            issues.append(f"daily report missing tracked signal row: {snippet}")
    failure_rows = [
        row
        for row in ledger_rows
        if row.get("tracking_status") == "failure" and int(row.get("observed_trading_days") or 0) >= 10
    ]
    for row in failure_rows:
        stock_text = report_stock_text(row)
        if stock_text and stock_text not in text:
            issues.append(f"daily report missing failure learning row: {stock_text}")
    if "個股成功率" in text and "不是個股成功率" not in text:
        issues.append("daily report appears to describe strategy rate as individual probability")
    return issues


def main() -> None:
    issues: list[str] = []
    issues.extend(require_files())
    ledger_rows: list[dict[str, str]] = []
    if not issues:
        ledger_issues, ledger_rows = check_ledger()
        issues.extend(ledger_issues)
        issues.extend(check_formal_candidates(ledger_rows))
        issues.extend(check_tracking_csv(ledger_rows))
        issues.extend(check_daily_report(ledger_rows))
    if issues:
        fail("formal tracking contract violations:\n" + "\n".join(issues))
    print("OK: formal tracking contract passed")


if __name__ == "__main__":
    main()
