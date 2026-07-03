from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from run_main_model_training_pipeline import (
    CONFIG_PATH,
    PROJECT_ROOT,
    load_json,
    normalize_market,
    normalize_stock,
    normalize_theme,
    read_csv,
    split_name,
    validate_inputs,
)


VALIDATION_DIR = PROJECT_ROOT / "validation_layer"
DECISION_DIR = PROJECT_ROOT / "decision_layer"

REVIEW_MD_PATH = VALIDATION_DIR / "target_sensitivity_review.md"
SUMMARY_PATH = VALIDATION_DIR / "target_sensitivity_summary.csv"
MONTHLY_PATH = VALIDATION_DIR / "target_sensitivity_monthly.csv"
DECISION_JSON_PATH = DECISION_DIR / "target_sensitivity_decision.json"

LOOKAHEAD_DAYS = 10
TARGET_CANDIDATES = [
    {
        "target_id": "old_touch_3pct_10d",
        "description": "目前主目標：10 日內任一天收盤 +3%，中途 -3% 只作風險旁支標籤。",
        "profit_threshold": 0.03,
        "adverse_threshold": None,
        "lookahead_days": 10,
        "is_current": True,
        "is_old_baseline": True,
    },
    {
        "target_id": "risk_adjusted_3pct_before_minus3pct_10d",
        "description": "硬風險比較：10 日內先收盤 +3%，且不能先最低價 -3%。",
        "profit_threshold": 0.03,
        "adverse_threshold": -0.03,
        "lookahead_days": 10,
        "is_current": False,
        "is_old_baseline": False,
    },
    {
        "target_id": "risk_adjusted_3pct_before_minus5pct_10d",
        "description": "放寬停損風險：10 日內先收盤 +3%，且不能先最低價 -5%。",
        "profit_threshold": 0.03,
        "adverse_threshold": -0.05,
        "lookahead_days": 10,
        "is_current": False,
        "is_old_baseline": False,
    },
    {
        "target_id": "risk_adjusted_2pct_before_minus3pct_10d",
        "description": "降低獲利門檻：10 日內先收盤 +2%，且不能先最低價 -3%。",
        "profit_threshold": 0.02,
        "adverse_threshold": -0.03,
        "lookahead_days": 10,
        "is_current": False,
        "is_old_baseline": False,
    },
    {
        "target_id": "risk_adjusted_3pct_before_minus3pct_5d",
        "description": "縮短時間：5 日內先收盤 +3%，且不能先最低價 -3%。",
        "profit_threshold": 0.03,
        "adverse_threshold": -0.03,
        "lookahead_days": 5,
        "is_current": False,
        "is_old_baseline": False,
    },
]


def fail(message: str) -> None:
    raise SystemExit(f"FAIL: {message}")


def first_hit_day(values: pd.DataFrame, threshold: float, direction: str) -> pd.Series:
    if direction == "ge":
        hits = values.ge(threshold)
    elif direction == "le":
        hits = values.le(threshold)
    else:
        fail(f"unsupported hit direction: {direction}")
    hit_matrix = hits.to_numpy(dtype=bool)
    first = np.full(hit_matrix.shape[0], np.nan, dtype=float)
    has_hit = hit_matrix.any(axis=1)
    first[has_hit] = np.argmax(hit_matrix[has_hit], axis=1) + 1
    return pd.Series(first, index=values.index)


def load_base_frame(config: dict) -> tuple[pd.DataFrame, str]:
    paths = validate_inputs(config)
    stock = normalize_stock(read_csv(paths["stock_daily_all"]))
    market = normalize_market(read_csv(paths["market_daily"]))
    _theme = normalize_theme(read_csv(paths["theme_group"]))
    stock_latest = stock["日期"].max()
    market_latest = market["日期"].max()
    if stock_latest != market_latest:
        fail(f"stock and market latest dates differ: {stock_latest.date()} vs {market_latest.date()}")
    out = stock.sort_values(["股票代號", "日期"]).reset_index(drop=True).copy()
    group = out.groupby("股票代號", sort=False)
    out["stock_trading_index"] = group.cumcount()
    out["buy_open_next"] = group["開盤價"].shift(-1)
    for horizon in range(1, LOOKAHEAD_DAYS + 1):
        out[f"future_close_{horizon}"] = group["收盤價"].shift(-horizon)
        out[f"future_low_{horizon}"] = group["最低價"].shift(-horizon)
    out["future_window_count"] = out[[f"future_close_{i}" for i in range(1, LOOKAHEAD_DAYS + 1)]].notna().sum(axis=1)
    out["future_low_window_count"] = out[[f"future_low_{i}" for i in range(1, LOOKAHEAD_DAYS + 1)]].notna().sum(axis=1)
    out["max_label_complete"] = (
        (out["future_window_count"] == LOOKAHEAD_DAYS)
        & (out["future_low_window_count"] == LOOKAHEAD_DAYS)
        & out["buy_open_next"].notna()
        & (out["buy_open_next"] > 0)
    )
    out["split"] = split_name(out["日期"], out["max_label_complete"], config)
    return out, stock_latest.strftime("%Y-%m-%d")


def evaluate_target(base: pd.DataFrame, candidate: dict) -> pd.DataFrame:
    lookahead = int(candidate["lookahead_days"])
    profit_threshold = float(candidate["profit_threshold"])
    adverse_threshold = candidate["adverse_threshold"]
    close_cols = [f"future_close_{i}" for i in range(1, lookahead + 1)]
    low_cols = [f"future_low_{i}" for i in range(1, lookahead + 1)]
    close_returns = base[close_cols].div(base["buy_open_next"], axis=0) - 1.0
    low_returns = base[low_cols].div(base["buy_open_next"], axis=0) - 1.0
    close_returns = close_returns.replace([np.inf, -np.inf], np.nan)
    low_returns = low_returns.replace([np.inf, -np.inf], np.nan)

    out = pd.DataFrame(
        {
            "日期": base["日期"],
            "股票代號": base["股票代號"],
            "split": base["split"],
            "label_complete": (
                (base[[f"future_close_{i}" for i in range(1, lookahead + 1)]].notna().sum(axis=1) == lookahead)
                & (base[[f"future_low_{i}" for i in range(1, lookahead + 1)]].notna().sum(axis=1) == lookahead)
                & base["buy_open_next"].notna()
                & (base["buy_open_next"] > 0)
            ),
        }
    )
    profit_day = first_hit_day(close_returns, profit_threshold, "ge")
    out["profit_event_day"] = profit_day
    out["future_high_close_return"] = close_returns.max(axis=1)
    out["max_adverse_return"] = low_returns.min(axis=1)
    out["day_end_return"] = close_returns.iloc[:, -1]

    if adverse_threshold is None:
        out["adverse_event_day"] = np.nan
        out["target_success"] = (out["label_complete"] & out["profit_event_day"].notna()).astype(int)
        out["same_day_both_event"] = 0
        out["adverse_first_event"] = 0
        out["realized_rule_return"] = out["day_end_return"]
        out.loc[out["target_success"].eq(1), "realized_rule_return"] = profit_threshold
    else:
        adverse_day = first_hit_day(low_returns, float(adverse_threshold), "le")
        has_profit = profit_day.notna()
        has_adverse = adverse_day.notna()
        profit_first = has_profit & (~has_adverse | (profit_day < adverse_day))
        adverse_first = has_adverse & (~has_profit | (adverse_day <= profit_day))
        out["adverse_event_day"] = adverse_day
        out["target_success"] = (out["label_complete"] & profit_first).astype(int)
        out["same_day_both_event"] = (
            out["label_complete"] & has_profit & has_adverse & profit_day.eq(adverse_day)
        ).astype(int)
        out["adverse_first_event"] = (out["label_complete"] & adverse_first).astype(int)
        out["realized_rule_return"] = out["day_end_return"]
        out.loc[profit_first, "realized_rule_return"] = profit_threshold
        out.loc[adverse_first, "realized_rule_return"] = adverse_threshold
    out.loc[~out["label_complete"], "realized_rule_return"] = np.nan
    out["target_id"] = str(candidate["target_id"])
    out["description"] = str(candidate["description"])
    out["profit_threshold"] = profit_threshold
    out["adverse_threshold"] = adverse_threshold if adverse_threshold is not None else np.nan
    out["lookahead_days"] = lookahead
    out["is_current"] = bool(candidate["is_current"])
    out["is_old_baseline"] = bool(candidate["is_old_baseline"])
    return out


def safe_mean(series: pd.Series) -> float:
    value = series.mean()
    return float(value) if pd.notna(value) else math.nan


def safe_std(series: pd.Series) -> float:
    value = series.std()
    return float(value) if pd.notna(value) else math.nan


def summarize_target(target: pd.DataFrame, section: str, group: str, split: str) -> dict:
    part = target[target["label_complete"]].copy()
    return {
        "target_id": str(target["target_id"].iloc[0]) if not target.empty else "",
        "section": section,
        "group": group,
        "split": split,
        "description": str(target["description"].iloc[0]) if not target.empty else "",
        "profit_threshold": safe_mean(target["profit_threshold"]) if not target.empty else math.nan,
        "adverse_threshold": safe_mean(target["adverse_threshold"]) if not target.empty else math.nan,
        "lookahead_days": int(target["lookahead_days"].iloc[0]) if not target.empty else 0,
        "is_current": bool(target["is_current"].iloc[0]) if not target.empty else False,
        "is_old_baseline": bool(target["is_old_baseline"].iloc[0]) if not target.empty else False,
        "rows": int(len(part)),
        "days": int(part["日期"].nunique()) if not part.empty else 0,
        "stocks": int(part["股票代號"].nunique()) if not part.empty else 0,
        "success_rate": safe_mean(part["target_success"]) if not part.empty else math.nan,
        "success_count": int(part["target_success"].sum()) if not part.empty else 0,
        "adverse_first_rate": safe_mean(part["adverse_first_event"]) if not part.empty else math.nan,
        "same_day_both_event_rate": safe_mean(part["same_day_both_event"]) if not part.empty else math.nan,
        "avg_high_close_return": safe_mean(part["future_high_close_return"]) if not part.empty else math.nan,
        "avg_max_adverse_return": safe_mean(part["max_adverse_return"]) if not part.empty else math.nan,
        "avg_realized_rule_return": safe_mean(part["realized_rule_return"]) if not part.empty else math.nan,
    }


def build_summaries(targets: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict] = []
    monthly_rows: list[dict] = []
    for target_id, target in targets.groupby("target_id", sort=False):
        for split in ["train", "development", "holdout"]:
            split_part = target[target["split"].eq(split)]
            summary_rows.append(summarize_target(split_part, "overall", split, split))
            month_frame = split_part.copy()
            month_frame["month"] = month_frame["日期"].dt.strftime("%Y-%m")
            for month, part in month_frame.groupby("month", sort=True):
                monthly_rows.append(summarize_target(part, "monthly", month, split))
    return pd.DataFrame(summary_rows), pd.DataFrame(monthly_rows)


def candidate_decision_score(summary: pd.DataFrame, monthly: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for target_id, overall in summary[summary["section"].eq("overall")].groupby("target_id", sort=False):
        by_split = {row["split"]: row for _, row in overall.iterrows()}
        if not {"train", "development", "holdout"}.issubset(by_split):
            continue
        train = by_split["train"]
        dev = by_split["development"]
        holdout = by_split["holdout"]
        monthly_target = monthly[monthly["target_id"].eq(target_id)].copy()
        monthly_success_std = safe_std(monthly_target["success_rate"])
        positive_month_share = safe_mean(monthly_target["success_rate"].gt(0.25).astype(float))
        split_gap = max(
            abs(float(train["success_rate"]) - float(dev["success_rate"])),
            abs(float(dev["success_rate"]) - float(holdout["success_rate"])),
            abs(float(train["success_rate"]) - float(holdout["success_rate"])),
        )
        enough_samples = min(int(train["success_count"]), int(dev["success_count"]), int(holdout["success_count"])) >= 100
        base_rate_usable = 0.25 <= float(holdout["success_rate"]) <= 0.65
        realized_positive = float(holdout["avg_realized_rule_return"]) > -0.005
        stable = split_gap <= 0.18 and (pd.isna(monthly_success_std) or monthly_success_std <= 0.18)
        score = 0.0
        score += 2.0 if enough_samples else -2.0
        score += 1.5 if base_rate_usable else -1.0
        score += 1.5 if stable else -1.5
        score += 1.0 if realized_positive else -1.0
        score += float(holdout["avg_realized_rule_return"]) * 10 if pd.notna(holdout["avg_realized_rule_return"]) else 0.0
        score += (1.0 - split_gap)
        rows.append(
            {
                "target_id": target_id,
                "description": holdout["description"],
                "is_current": bool(holdout["is_current"]),
                "is_old_baseline": bool(holdout["is_old_baseline"]),
                "train_success_rate": train["success_rate"],
                "development_success_rate": dev["success_rate"],
                "holdout_success_rate": holdout["success_rate"],
                "holdout_success_count": holdout["success_count"],
                "holdout_adverse_first_rate": holdout["adverse_first_rate"],
                "holdout_avg_realized_rule_return": holdout["avg_realized_rule_return"],
                "split_success_rate_max_gap": split_gap,
                "monthly_success_rate_std": monthly_success_std,
                "positive_month_share": positive_month_share,
                "enough_samples": enough_samples,
                "base_rate_usable": base_rate_usable,
                "realized_positive": realized_positive,
                "stable_across_splits": stable,
                "decision_score": score,
            }
        )
    return pd.DataFrame(rows).sort_values("decision_score", ascending=False)


def decide(summary: pd.DataFrame, monthly: pd.DataFrame) -> dict:
    scored = candidate_decision_score(summary, monthly)
    current = scored[scored["is_current"]].iloc[0].to_dict()
    non_old = scored[~scored["is_old_baseline"]].copy()
    best = non_old.iloc[0].to_dict()
    if str(current["target_id"]) == "old_touch_3pct_10d":
        status = "current_target_label_viable_but_model_failed"
        recommended = "review_data_enrichment"
        reason = "目前主目標已回到 10 日內 +3%；-3% 只作風險旁支標籤。此審查保留硬風險目標作比較，但不建議把 -3% 再改回自動失敗。"
    elif str(best["target_id"]) != str(current["target_id"]) and float(best["decision_score"]) >= float(current["decision_score"]) + 1.0:
        status = "current_target_not_best"
        recommended = "review_target_contract_change"
        reason = "目前風險調整目標不是敏感度審查中最穩的目標；下一步應討論是否改 label，而不是再重訓目前目標。"
    elif bool(current["enough_samples"]) and bool(current["base_rate_usable"]) and bool(current["stable_across_splits"]):
        status = "current_target_label_viable_but_model_failed"
        recommended = "review_data_enrichment"
        reason = "目前目標本身樣本量與分布可用，但模型仍學不到選股優勢；下一步應看資料缺口，不是再調模型。"
    else:
        status = "current_target_too_unstable"
        recommended = "review_target_contract_change"
        reason = "目前目標的樣本分布或穩定性不足；下一步應改目標定義。"
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": status,
        "recommended_next_step": recommended,
        "reason": reason,
        "current_target_id": current["target_id"],
        "best_non_old_target_id": best["target_id"],
        "current_target": current,
        "best_non_old_target": best,
    }


def fmt_pct(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.2%}"


def write_review_md(data_latest: str, decision: dict, scored: pd.DataFrame) -> None:
    current = decision["current_target"]
    best = decision["best_non_old_target"]
    lines = [
        "# Target Sensitivity Review",
        "",
        f"- Generated: {decision['generated_at']}",
        f"- Data latest date: {data_latest}",
        "- Scope: label-only target sensitivity; no model training; no stock candidates.",
        "- Formal output: unchanged by this review.",
        "",
        "## 白話結論",
        "",
        decision["reason"],
        "",
        f"- Review status: `{decision['status']}`",
        f"- Recommended next step: `{decision['recommended_next_step']}`",
        f"- Current target: `{decision['current_target_id']}`",
        f"- Best non-old target by this review: `{decision['best_non_old_target_id']}`",
        "",
        "## Current Target Snapshot",
        "",
        f"- Holdout success rate: {fmt_pct(current.get('holdout_success_rate'))}",
        f"- Holdout adverse-first rate: {fmt_pct(current.get('holdout_adverse_first_rate'))}",
        f"- Holdout realized rule return: {fmt_pct(current.get('holdout_avg_realized_rule_return'))}",
        f"- Split success-rate max gap: {fmt_pct(current.get('split_success_rate_max_gap'))}",
        "",
        "## Best Alternative Snapshot",
        "",
        f"- Target: `{best.get('target_id')}`",
        f"- Description: {best.get('description')}",
        f"- Holdout success rate: {fmt_pct(best.get('holdout_success_rate'))}",
        f"- Holdout adverse-first rate: {fmt_pct(best.get('holdout_adverse_first_rate'))}",
        f"- Holdout realized rule return: {fmt_pct(best.get('holdout_avg_realized_rule_return'))}",
        f"- Split success-rate max gap: {fmt_pct(best.get('split_success_rate_max_gap'))}",
        "",
        "## Candidate Ranking",
        "",
        "| target | holdout success | realized return | split gap | decision score |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, row in scored.iterrows():
        lines.append(
            "| "
            + str(row["target_id"])
            + " | "
            + fmt_pct(row["holdout_success_rate"])
            + " | "
            + fmt_pct(row["holdout_avg_realized_rule_return"])
            + " | "
            + fmt_pct(row["split_success_rate_max_gap"])
            + " | "
            + f"{float(row['decision_score']):.3f}"
            + " |"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- This does not choose stocks.",
            "- This does not train or promote a model.",
            "- This does not use old report scores or artificial labels.",
            "- This review compares target definitions only; it is not a probability report.",
            "",
            "## Outputs",
            "",
            f"- `{SUMMARY_PATH.relative_to(PROJECT_ROOT)}`",
            f"- `{MONTHLY_PATH.relative_to(PROJECT_ROOT)}`",
            f"- `{DECISION_JSON_PATH.relative_to(PROJECT_ROOT)}`",
            "",
        ]
    )
    REVIEW_MD_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    config = load_json(CONFIG_PATH)
    base, data_latest = load_base_frame(config)
    target_frames = [evaluate_target(base, candidate) for candidate in TARGET_CANDIDATES]
    targets = pd.concat(target_frames, ignore_index=True)
    summary, monthly = build_summaries(targets)
    scored = candidate_decision_score(summary, monthly)
    decision = decide(summary, monthly)
    summary = summary.merge(
        scored[[
            "target_id",
            "decision_score",
            "split_success_rate_max_gap",
            "monthly_success_rate_std",
            "positive_month_share",
            "enough_samples",
            "base_rate_usable",
            "realized_positive",
            "stable_across_splits",
        ]],
        on="target_id",
        how="left",
    )
    SUMMARY_PATH.write_text(summary.to_csv(index=False), encoding="utf-8-sig")
    MONTHLY_PATH.write_text(monthly.to_csv(index=False), encoding="utf-8-sig")
    DECISION_JSON_PATH.write_text(json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_review_md(data_latest, decision, scored)
    print("OK: target sensitivity review completed")
    print(f"STATUS: {decision['status']}")
    print(f"NEXT_STEP: {decision['recommended_next_step']}")
    print(f"REVIEW: {REVIEW_MD_PATH}")


if __name__ == "__main__":
    main()
