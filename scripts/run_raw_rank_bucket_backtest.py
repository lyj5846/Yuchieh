from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import pandas as pd

from run_main_model_training_pipeline import (
    CONFIG_PATH,
    PROJECT_ROOT,
    load_json,
    normalize_stock,
    normalize_stock_id,
    read_csv,
    validate_inputs,
)


VALIDATION_DIR = PROJECT_ROOT / "validation_layer"
DECISION_DIR = PROJECT_ROOT / "decision_layer"
SCORES_PATH = PROJECT_ROOT / "model_layer" / "main_model_scores.csv"
MAIN_DECISION_PATH = DECISION_DIR / "main_model_decision.json"

DETAIL_PATH = VALIDATION_DIR / "raw_rank_bucket_backtest.csv"
SUMMARY_MD_PATH = VALIDATION_DIR / "raw_rank_bucket_backtest_summary.md"
DECISION_MD_PATH = DECISION_DIR / "raw_rank_bucket_selection_policy_decision.md"

LOOKAHEAD_DAYS = 10
MAX_DAILY_CANDIDATES = 3
MIN_HOLDOUT_COMPLETED = 20
MIN_OVERALL_COMPLETED = 50
MAX_SUCCESS_DROP = 0.05
MAX_RETURN_DROP = 0.02


def fail(message: str) -> None:
    raise SystemExit(f"FAIL: {message}")


def fmt_pct(value: object) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.2%}"


def safe_float(value: object) -> float:
    try:
        if value is None or pd.isna(value):
            return math.nan
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def selected_gate() -> float | None:
    if not MAIN_DECISION_PATH.exists():
        fail("missing main_model_decision.json")
    decision = json.loads(MAIN_DECISION_PATH.read_text(encoding="utf-8"))
    value = decision.get("selected_gate")
    parsed = safe_float(value)
    return None if pd.isna(parsed) else parsed


def load_stock_index(config: dict) -> pd.DataFrame:
    paths = validate_inputs(config)
    stock = normalize_stock(read_csv(paths["stock_daily_all"]))
    stock["股票代號"] = normalize_stock_id(stock["股票代號"])
    stock = stock.sort_values(["股票代號", "日期"]).reset_index(drop=True)
    stock["stock_trading_index"] = stock.groupby("股票代號", sort=False).cumcount()
    return stock[["日期", "股票代號", "stock_trading_index"]].copy()


def load_scores(stock_index: pd.DataFrame) -> pd.DataFrame:
    if not SCORES_PATH.exists():
        fail("missing main_model_scores.csv")
    scores = pd.read_csv(SCORES_PATH, encoding="utf-8-sig")
    required = {
        "日期",
        "股票代號",
        "股票名稱",
        "主分類",
        "split",
        "daily_rank",
        "integrated_research_score",
        "target_success",
        "future_10d_high_close_return",
        "daily_market_success_rate",
        "daily_market_avg_return",
    }
    missing = required - set(scores.columns)
    if missing:
        fail("main_model_scores.csv missing columns: " + ", ".join(sorted(missing)))

    scores = scores.copy()
    scores["日期"] = pd.to_datetime(scores["日期"])
    scores["股票代號"] = normalize_stock_id(scores["股票代號"])
    for col in [
        "daily_rank",
        "integrated_research_score",
        "target_success",
        "future_10d_high_close_return",
        "daily_market_success_rate",
        "daily_market_avg_return",
    ]:
        scores[col] = pd.to_numeric(scores[col], errors="coerce")
    scores = scores.merge(stock_index, on=["日期", "股票代號"], how="left")
    scores = scores.dropna(subset=["stock_trading_index", "integrated_research_score", "daily_rank"]).copy()
    scores["stock_trading_index"] = scores["stock_trading_index"].astype(int)
    scores["raw_rank_bucket"] = scores["daily_rank"].map(rank_bucket)
    scores["label_status"] = "tracking"
    completed = scores["target_success"].notna() & scores["future_10d_high_close_return"].notna()
    scores.loc[completed, "label_status"] = "completed"
    return scores.sort_values(["日期", "integrated_research_score"], ascending=[True, False]).reset_index(drop=True)


def rank_bucket(rank: object) -> str:
    value = safe_float(rank)
    if pd.isna(value):
        return "unknown"
    if value <= 3:
        return "raw_top3"
    if value <= 10:
        return "raw_top4_10"
    return "raw_11_plus"


def replay_current_formal_logic(scores: pd.DataFrame, gate: float | None) -> pd.DataFrame:
    selected_indices: list[int] = []
    last_pick_index: dict[str, int] = {}
    for signal_date, day_rows in scores.groupby("日期", sort=True):
        day_rows = day_rows.sort_values(
            ["integrated_research_score", "股票代號"],
            ascending=[False, True],
        )
        if day_rows.empty:
            continue
        if gate is not None and float(day_rows.iloc[0]["integrated_research_score"]) < gate:
            continue
        selected_today = 0
        for idx, row in day_rows.iterrows():
            stock_id = str(row["股票代號"])
            stock_index = int(row["stock_trading_index"])
            if stock_id in last_pick_index and stock_index - last_pick_index[stock_id] <= LOOKAHEAD_DAYS:
                continue
            selected_indices.append(idx)
            last_pick_index[stock_id] = stock_index
            selected_today += 1
            if selected_today >= MAX_DAILY_CANDIDATES:
                break
    if not selected_indices:
        return pd.DataFrame(columns=list(scores.columns) + ["formal_pick_rank"])
    formal = scores.loc[selected_indices].copy()
    formal = formal.sort_values(["日期", "integrated_research_score"], ascending=[True, False])
    formal["formal_pick_rank"] = formal.groupby("日期").cumcount() + 1
    return formal.reset_index(drop=True)


def summarize(formal: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for bucket in ["raw_top3", "raw_top4_10", "raw_11_plus"]:
        bucket_rows = formal[formal["raw_rank_bucket"] == bucket].copy()
        for split in ["overall", "train", "development", "holdout"]:
            part = bucket_rows if split == "overall" else bucket_rows[bucket_rows["split"] == split]
            completed = part[part["label_status"] == "completed"].copy()
            tracking = part[part["label_status"] != "completed"].copy()
            if completed.empty:
                success_rate = math.nan
                avg_return = math.nan
                market_success = math.nan
                market_return = math.nan
                success_lift = math.nan
                return_lift = math.nan
                max_stock_share = math.nan
                max_industry_share = math.nan
            else:
                success_rate = float(completed["target_success"].mean())
                avg_return = float(completed["future_10d_high_close_return"].mean())
                market_success = float(completed["daily_market_success_rate"].mean())
                market_return = float(completed["daily_market_avg_return"].mean())
                success_lift = success_rate - market_success
                return_lift = avg_return - market_return
                max_stock_share = float(completed["股票代號"].value_counts(normalize=True).max())
                max_industry_share = float(completed["主分類"].fillna("未分類").value_counts(normalize=True).max())
            rows.append(
                {
                    "raw_rank_bucket": bucket,
                    "split": split,
                    "signals": int(len(part)),
                    "completed_signals": int(len(completed)),
                    "tracking_signals": int(len(tracking)),
                    "active_days": int(part["日期"].nunique()) if not part.empty else 0,
                    "unique_stocks": int(completed["股票代號"].nunique()) if not completed.empty else 0,
                    "success_rate": success_rate,
                    "avg_10d_high_close_return": avg_return,
                    "daily_market_success_rate": market_success,
                    "daily_market_avg_return": market_return,
                    "success_lift": success_lift,
                    "return_lift": return_lift,
                    "max_stock_share": max_stock_share,
                    "max_industry_share": max_industry_share,
                }
            )
    return pd.DataFrame(rows)


def metric(summary: pd.DataFrame, bucket: str, split: str) -> dict:
    row = summary[(summary["raw_rank_bucket"] == bucket) & (summary["split"] == split)]
    if row.empty:
        return {}
    return row.iloc[0].to_dict()


def decide(summary: pd.DataFrame) -> dict:
    holdout_4_10 = metric(summary, "raw_top4_10", "holdout")
    holdout_11 = metric(summary, "raw_11_plus", "holdout")
    overall_4_10 = metric(summary, "raw_top4_10", "overall")
    overall_11 = metric(summary, "raw_11_plus", "overall")

    if (
        int(holdout_4_10.get("completed_signals", 0)) >= MIN_HOLDOUT_COMPLETED
        and int(holdout_11.get("completed_signals", 0)) >= MIN_HOLDOUT_COMPLETED
    ):
        basis = "holdout"
        base = holdout_4_10
        test = holdout_11
    elif (
        int(overall_4_10.get("completed_signals", 0)) >= MIN_OVERALL_COMPLETED
        and int(overall_11.get("completed_signals", 0)) >= MIN_OVERALL_COMPLETED
    ):
        basis = "overall"
        base = overall_4_10
        test = overall_11
    else:
        return {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": "insufficient_evidence",
            "basis": "none",
            "recommended_policy": "keep_current_policy_but_mark_raw_11_plus_as_candidate_observation",
            "reason": "raw_11_plus or raw_top4_10 does not have enough completed samples for a stable policy decision.",
        }

    success_drop = safe_float(base.get("success_rate")) - safe_float(test.get("success_rate"))
    return_drop = safe_float(base.get("avg_10d_high_close_return")) - safe_float(test.get("avg_10d_high_close_return"))
    test_success_lift = safe_float(test.get("success_lift"))
    test_return_lift = safe_float(test.get("return_lift"))
    stable = (
        success_drop <= MAX_SUCCESS_DROP
        and return_drop <= MAX_RETURN_DROP
        and test_success_lift >= 0
        and test_return_lift >= 0
    )
    status = "maintain_current_fill_policy" if stable else "restrict_formal_candidates_to_raw_top10"
    policy = (
        "current_fill_policy_allowed"
        if stable
        else "raw_top10_only_do_not_fill_from_raw_11_plus"
    )
    reason = (
        "raw_11_plus stayed close to raw_top4_10 and kept positive market lift."
        if stable
        else "raw_11_plus underperformed raw_top4_10 or failed to keep positive market lift."
    )
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": status,
        "basis": basis,
        "recommended_policy": policy,
        "reason": reason,
        "success_drop_vs_raw_top4_10": success_drop,
        "return_drop_vs_raw_top4_10": return_drop,
        "raw_11_plus_success_lift": test_success_lift,
        "raw_11_plus_return_lift": test_return_lift,
    }


def write_detail(formal: pd.DataFrame) -> None:
    output_cols = [
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
    ]
    out = formal[output_cols].copy()
    out["日期"] = pd.to_datetime(out["日期"]).dt.strftime("%Y-%m-%d")
    out.to_csv(DETAIL_PATH, index=False, encoding="utf-8-sig")


def md_table(df: pd.DataFrame) -> list[str]:
    columns = [
        "raw_rank_bucket",
        "split",
        "completed_signals",
        "tracking_signals",
        "success_rate",
        "avg_10d_high_close_return",
        "success_lift",
        "return_lift",
    ]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in df[columns].iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["raw_rank_bucket"]),
                    str(row["split"]),
                    str(int(row["completed_signals"])),
                    str(int(row["tracking_signals"])),
                    fmt_pct(row["success_rate"]),
                    fmt_pct(row["avg_10d_high_close_return"]),
                    fmt_pct(row["success_lift"]),
                    fmt_pct(row["return_lift"]),
                ]
            )
            + " |"
        )
    return lines


def write_summary(summary: pd.DataFrame, decision: dict, gate: float | None, formal: pd.DataFrame) -> None:
    latest = formal["日期"].max().strftime("%Y-%m-%d") if not formal.empty else ""
    rows_0703 = formal[pd.to_datetime(formal["日期"]).dt.strftime("%Y-%m-%d") == "2026-07-03"].copy()
    row_0703_text = []
    if not rows_0703.empty:
        for _, row in rows_0703.sort_values("formal_pick_rank").iterrows():
            row_0703_text.append(
                f"- {row['股票代號']} {row['股票名稱']}: raw rank {int(row['daily_rank'])}, bucket `{row['raw_rank_bucket']}`"
            )
    else:
        row_0703_text.append("- 2026-07-03 has no replayed formal candidates in this backtest output.")

    lines = [
        "# Raw Rank Bucket Backtest",
        "",
        f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Latest replayed date: {latest}",
        f"- Selected gate: {gate if gate is not None else 'none'}",
        "- Purpose:制度審查；不重訓模型、不改正式候選。",
        "- research_score is a ranking score, not a probability.",
        "- 10日+3% success rate in this report is historical bucket performance, not individual stock probability.",
        "",
        "## Decision",
        "",
        f"- Status: `{decision.get('status')}`",
        f"- Basis: `{decision.get('basis')}`",
        f"- Recommended policy: `{decision.get('recommended_policy')}`",
        f"- Reason: {decision.get('reason')}",
        "",
        "## 2026-07-03 Bucket Check",
        "",
        *row_0703_text,
        "",
        "## Bucket Summary",
        "",
        *md_table(summary),
        "",
    ]
    SUMMARY_MD_PATH.write_text("\n".join(lines), encoding="utf-8")


def write_decision(decision: dict) -> None:
    lines = [
        "# Raw Rank Bucket Selection Policy Decision",
        "",
        f"- Generated: {decision.get('generated_at')}",
        f"- Status: `{decision.get('status')}`",
        f"- Basis: `{decision.get('basis')}`",
        f"- Recommended policy: `{decision.get('recommended_policy')}`",
        f"- Reason: {decision.get('reason')}",
        "- Formal output is unchanged by this review.",
        "- This does not train or promote a model.",
        "- research_score remains a ranking score, not a probability.",
        "",
    ]
    for key in [
        "success_drop_vs_raw_top4_10",
        "return_drop_vs_raw_top4_10",
        "raw_11_plus_success_lift",
        "raw_11_plus_return_lift",
    ]:
        if key in decision:
            lines.append(f"- {key}: {fmt_pct(decision.get(key))}")
    DECISION_MD_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    config = load_json(CONFIG_PATH)
    stock_index = load_stock_index(config)
    scores = load_scores(stock_index)
    gate = selected_gate()
    formal = replay_current_formal_logic(scores, gate)
    if formal.empty:
        fail("current replay logic produced no candidates")
    summary = summarize(formal)
    decision = decide(summary)
    write_detail(formal)
    write_summary(summary, decision, gate, formal)
    write_decision(decision)
    print("OK: raw rank bucket backtest completed")
    print(f"STATUS: {decision.get('status')}")
    print(f"POLICY: {decision.get('recommended_policy')}")
    print(f"DETAIL: {DETAIL_PATH}")
    print(f"SUMMARY: {SUMMARY_MD_PATH}")


if __name__ == "__main__":
    main()
