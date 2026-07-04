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

ALLOWED_TRACKING_STATUS = {
    "not_started",
    "tracking",
    "success",
    "failure",
    "missing_signal_price",
    "missing_buy_price",
}

EXPECTED_0630_MAX_CLOSE_RETURN_DATES = {
    ("2026-06-18", "2409"): "2026-06-30",
    ("2026-06-22", "2610"): "2026-06-25",
    ("2026-06-25", "2618"): "2026-06-30",
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


def check_formal_candidates() -> list[str]:
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

    rows_0630 = [row for row in rows if row.get("date") == "2026-06-30"]
    stocks_0630 = {row.get("stock_id") for row in rows_0630}
    expected_0630 = {"6515", "2404", "6669"}
    if rows_0630 and stocks_0630 != expected_0630:
        issues.append(f"2026-06-30 formal candidates mismatch: {sorted(stocks_0630)}")
    for row in rows_0630:
        if row.get("tracking_status") != "tracking":
            issues.append(f"2026-06-30 formal candidate {row.get('stock_id')} should be tracking in formal_candidates.csv")
        count = row.get("consecutive_recommendation_count", "")
        if not count.isdigit() or int(count) < 1:
            issues.append(f"2026-06-30 formal candidate {row.get('stock_id')} must have positive consecutive_recommendation_count")
    return issues


def check_ledger() -> list[str]:
    issues: list[str] = []
    rows = read_csv(LEDGER_PATH)
    if not rows:
        issues.append("formal_signal_ledger.csv is empty")
        return issues

    columns = set(rows[0])
    missing = REQUIRED_LEDGER_COLUMNS - columns
    if missing:
        issues.append(f"ledger missing columns: {sorted(missing)}")

    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (row.get("signal_date", ""), row.get("stock_id", ""))
        if key in seen:
            issues.append(f"duplicate ledger signal: {key[0]} {key[1]}")
        seen.add(key)

        status = row.get("tracking_status", "")
        if status not in ALLOWED_TRACKING_STATUS:
            issues.append(f"invalid tracking_status for {key[0]} {key[1]}: {status}")
        count = row.get("consecutive_recommendation_count", "")
        if count and (not count.isdigit() or int(count) < 1):
            issues.append(f"invalid consecutive_recommendation_count for {key[0]} {key[1]}: {count}")
        expected_max_date = EXPECTED_0630_MAX_CLOSE_RETURN_DATES.get(key)
        if expected_max_date is not None and row.get("max_close_return_date", "") != expected_max_date:
            issues.append(f"max close return date mismatch for {key[0]} {key[1]}: {row.get('max_close_return_date', '')} != {expected_max_date}")

    rows_0630 = [row for row in rows if row.get("signal_date") == "2026-06-30"]
    stocks_0630 = {row.get("stock_id") for row in rows_0630}
    expected_0630 = {"6515", "2404", "6669"}
    if stocks_0630 != expected_0630:
        issues.append(f"2026-06-30 ledger candidates mismatch: {sorted(stocks_0630)}")
    for row in rows_0630:
        if row.get("candidate_type") != "new_formal":
            issues.append(f"2026-06-30 ledger candidate {row.get('stock_id')} must be new_formal")
        count = row.get("consecutive_recommendation_count", "")
        if not count.isdigit() or int(count) < 1:
            issues.append(f"2026-06-30 ledger candidate {row.get('stock_id')} must have positive consecutive_recommendation_count")

    for stock in ["2409", "2610", "2618"]:
        if stock not in {row.get("stock_id") for row in rows}:
            issues.append(f"ledger missing previously tracked high-score stock: {stock}")
    return issues


def check_tracking_csv() -> list[str]:
    issues: list[str] = []
    rows = read_csv(TRACKING_CSV_PATH)
    if not rows:
        issues.append("formal_candidate_tracking.csv is empty")
        return issues
    columns = set(rows[0])
    if "consecutive_recommendation_count" not in columns:
        issues.append("formal_candidate_tracking.csv must include consecutive_recommendation_count")
    if "max_close_return_date" not in columns:
        issues.append("formal_candidate_tracking.csv must include max_close_return_date")
    for row in rows:
        count = row.get("consecutive_recommendation_count", "")
        if not count.isdigit() or int(count) < 1:
            issues.append(f"tracking row {row.get('signal_date')} {row.get('stock_id')} must have positive consecutive_recommendation_count")
        expected_max_date = EXPECTED_0630_MAX_CLOSE_RETURN_DATES.get((row.get("signal_date", ""), row.get("stock_id", "")))
        if expected_max_date is not None and row.get("max_close_return_date", "") != expected_max_date:
            issues.append(f"tracking max close return date mismatch for {row.get('signal_date')} {row.get('stock_id')}: {row.get('max_close_return_date', '')} != {expected_max_date}")
    return issues


def check_daily_report() -> list[str]:
    issues: list[str] = []
    text = DAILY_REPORT_PATH.read_text(encoding="utf-8")
    required_text = [
        "今日新進正式候選",
        "高分續強但已追蹤",
        "正式候選追蹤",
        "research_score 是排序分數，不是機率",
        "strategy_backtest_hit_rate 是策略歷史回測成功率，不是個股成功率",
        "連續被推薦次數",
        "最高收盤報酬日期",
        "6515 穎崴",
        "2404 漢唐",
        "6669 緯穎",
        "2409 友達",
        "2610 華航",
        "2618 長榮航",
    ]
    for needle in required_text:
        if needle not in text:
            issues.append(f"daily report missing text: {needle}")
    if text.count("連續被推薦次數") < 3:
        issues.append("daily report must show 連續被推薦次數 in new candidates, continuations, and tracking sections")
    for expected_snippet in [
        "| 2026-06-18 | 2409 友達 |",
        "| 2026-06-22 | 2610 華航 |",
        "| 2026-06-25 | 2618 長榮航 |",
    ]:
        if expected_snippet not in text:
            issues.append(f"daily report missing tracked signal row: {expected_snippet}")
    if "個股成功率" in text and "不是個股成功率" not in text:
        issues.append("daily report appears to describe strategy rate as individual probability")
    return issues


def main() -> None:
    issues: list[str] = []
    issues.extend(require_files())
    if not issues:
        issues.extend(check_formal_candidates())
        issues.extend(check_ledger())
        issues.extend(check_tracking_csv())
        issues.extend(check_daily_report())
    if issues:
        fail("formal tracking contract violations:\n" + "\n".join(issues))
    print("OK: formal tracking contract passed")


if __name__ == "__main__":
    main()
