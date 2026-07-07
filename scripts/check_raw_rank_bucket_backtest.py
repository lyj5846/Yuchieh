from __future__ import annotations

import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DETAIL_PATH = PROJECT_ROOT / "validation_layer" / "raw_rank_bucket_backtest.csv"
SUMMARY_MD_PATH = PROJECT_ROOT / "validation_layer" / "raw_rank_bucket_backtest_summary.md"
DECISION_MD_PATH = PROJECT_ROOT / "decision_layer" / "raw_rank_bucket_selection_policy_decision.md"

REQUIRED_DETAIL_COLUMNS = {
    "日期",
    "股票代號",
    "股票名稱",
    "主分類",
    "split",
    "formal_pick_rank",
    "daily_rank",
    "raw_rank_bucket",
    "integrated_research_score",
    "label_status",
    "target_success",
    "future_10d_high_close_return",
    "daily_market_success_rate",
    "daily_market_avg_return",
}
REQUIRED_BUCKETS = {"raw_top3", "raw_top4_10", "raw_11_plus"}
ALLOWED_STATUS = {
    "maintain_current_fill_policy",
    "restrict_formal_candidates_to_raw_top10",
    "insufficient_evidence",
}


def fail(message: str) -> None:
    raise SystemExit(f"FAIL: {message}")


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        fail(f"missing output: {path.relative_to(PROJECT_ROOT)}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        fail(f"{path.relative_to(PROJECT_ROOT)} is empty")
    return rows


def main() -> None:
    detail = read_rows(DETAIL_PATH)
    missing = REQUIRED_DETAIL_COLUMNS - set(detail[0])
    if missing:
        fail("raw_rank_bucket_backtest.csv missing columns: " + ", ".join(sorted(missing)))

    buckets = {row["raw_rank_bucket"] for row in detail}
    if not REQUIRED_BUCKETS.issubset(buckets):
        fail("raw rank backtest missing buckets: " + ", ".join(sorted(REQUIRED_BUCKETS - buckets)))

    labels = {row["label_status"] for row in detail}
    if not {"completed", "tracking"}.intersection(labels):
        fail("raw rank backtest must contain completed or tracking labels")

    rows_0703 = {row["股票代號"]: row for row in detail if row["日期"] == "2026-07-03"}
    expected_0703 = {
        "7769": "raw_top4_10",
        "8046": "raw_11_plus",
        "6669": "raw_11_plus",
    }
    for stock_id, expected_bucket in expected_0703.items():
        row = rows_0703.get(stock_id)
        if not row:
            fail(f"2026-07-03 expected replay candidate missing: {stock_id}")
        if row.get("raw_rank_bucket") != expected_bucket:
            fail(
                f"2026-07-03 {stock_id} bucket mismatch: "
                f"{row.get('raw_rank_bucket')} != {expected_bucket}"
            )

    summary = SUMMARY_MD_PATH.read_text(encoding="utf-8") if SUMMARY_MD_PATH.exists() else ""
    for phrase in [
        "Raw Rank Bucket Backtest",
        "制度審查；不重訓模型、不改正式候選",
        "research_score is a ranking score, not a probability.",
        "historical bucket performance, not individual stock probability",
        "raw_top3",
        "raw_top4_10",
        "raw_11_plus",
    ]:
        if phrase not in summary:
            fail(f"raw rank bucket summary missing phrase: {phrase}")

    decision = DECISION_MD_PATH.read_text(encoding="utf-8") if DECISION_MD_PATH.exists() else ""
    if not decision:
        fail("missing raw rank bucket decision")
    if "Formal output is unchanged by this review." not in decision:
        fail("decision must state formal output is unchanged")
    if "research_score remains a ranking score, not a probability." not in decision:
        fail("decision must state research_score is not a probability")
    if not any(f"Status: `{status}`" in decision for status in ALLOWED_STATUS):
        fail("decision status is not one of the allowed values")

    print("OK: raw rank bucket backtest contract passed")
    print(f"SUMMARY: {SUMMARY_MD_PATH}")


if __name__ == "__main__":
    main()
