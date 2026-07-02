from __future__ import annotations

import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from run_deep_learning_frequency_expansion_experiment import (  # noqa: E402
    add_day_ranks,
    score_datasets,
    strategy_candidates,
    strategy_summary,
)
from run_deep_learning_selection_experiment import (  # noqa: E402
    fmt_pct,
    sequence_features,
    standardize,
    train_mlp,
    validate_inputs,
    build_dataset,
)
from run_deep_learning_calibrated_candidate_experiment import (  # noqa: E402
    load_config,
    read_csv,
    build_scoring_dataset,
    split_masks,
)


EXPERIMENT_DIR = PROJECT_ROOT / "research_layer"
FORMAL_DIR = PROJECT_ROOT / "research_layer" / "formal_write_disabled"

DECISION_PATH = EXPERIMENT_DIR / "rolling_deep_learning_strategy_decision.md"
BACKTEST_PATH = EXPERIMENT_DIR / "rolling_deep_learning_strategy_backtest.csv"
MONTHLY_PATH = EXPERIMENT_DIR / "rolling_deep_learning_strategy_monthly.csv"
ADDED_DAYS_PATH = EXPERIMENT_DIR / "rolling_deep_learning_strategy_added_days.csv"
LATEST_PATH = EXPERIMENT_DIR / "rolling_deep_learning_strategy_latest.csv"
FORMAL_STATUS_PATH = FORMAL_DIR / "formal_status.md"
FORMAL_CANDIDATES_PATH = FORMAL_DIR / "formal_candidates.csv"


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def month_starts(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    months = pd.date_range(start=start.replace(day=1), end=end, freq="MS")
    return [pd.Timestamp(m) for m in months]


def score_with_window(data: pd.DataFrame, features: pd.DataFrame, eval_month: pd.Timestamp, months: int) -> pd.DataFrame:
    eval_start = eval_month
    eval_end = eval_month + pd.offsets.MonthEnd(0)
    train_end = eval_start - pd.Timedelta(days=15)
    train_start = train_end - pd.DateOffset(months=months)
    train_mask = (data["日期"] >= train_start) & (data["日期"] <= train_end)
    eval_mask = (data["日期"] >= eval_start) & (data["日期"] <= eval_end)
    if train_mask.sum() < 5000 or eval_mask.sum() == 0:
        return pd.DataFrame()

    X, _, _ = standardize(features.to_numpy(dtype=np.float32), train_mask.to_numpy())
    y = data["target_success"].to_numpy(dtype=np.float32).reshape(-1, 1)
    model, _ = train_mlp(X[train_mask.to_numpy()], y[train_mask.to_numpy()], hidden=32, epochs=45, seed=100 + months + eval_month.month)
    scored = data.loc[eval_mask].copy()
    eval_x = X[eval_mask.to_numpy()]
    scored["dl_score"] = sigmoid(
        (np.maximum(eval_x @ model["w1"] + model["b1"], 0.0) @ model["w2"] + model["b2"])[:, 0]
    )
    scored["rolling_window_months"] = months
    scored["eval_month"] = eval_start.strftime("%Y-%m")
    return scored


def summarize_strategy(full_universe: pd.DataFrame, picked: pd.DataFrame, strategy: str) -> dict:
    if picked.empty:
        return {
            "strategy": strategy,
            "rows": 0,
            "days": 0,
            "success_rate": float("nan"),
            "same_day_baseline_success_rate": float("nan"),
            "success_lift": float("nan"),
            "avg_10d_high_close_return": float("nan"),
            "same_day_baseline_avg_return": float("nan"),
            "return_lift": float("nan"),
            "top_stock_share": float("nan"),
            "top_industry_share": float("nan"),
        }
    selected_days = picked["日期"].drop_duplicates()
    full_on_selected_days = full_universe[full_universe["日期"].isin(selected_days)].copy()
    base = full_on_selected_days.groupby("日期").agg(
        day_success=("target_success", "mean"),
        day_return=("future_10d_high_close_return", "mean"),
    )
    chosen = picked.groupby("日期").agg(
        pick_success=("target_success", "mean"),
        pick_return=("future_10d_high_close_return", "mean"),
    )
    joined = chosen.join(base, how="inner")
    return {
        "strategy": strategy,
        "rows": len(picked),
        "days": picked["日期"].nunique(),
        "success_rate": picked["target_success"].mean(),
        "same_day_baseline_success_rate": joined["day_success"].mean(),
        "success_lift": joined["pick_success"].mean() - joined["day_success"].mean(),
        "avg_10d_high_close_return": picked["future_10d_high_close_return"].mean(),
        "same_day_baseline_avg_return": joined["day_return"].mean(),
        "return_lift": joined["pick_return"].mean() - joined["day_return"].mean(),
        "top_stock_share": picked["股票代號"].value_counts(normalize=True).iloc[0],
        "top_industry_share": picked["主分類"].fillna("unknown").value_counts(normalize=True).iloc[0],
    }


def pick_strategy(scored: pd.DataFrame, strategy: str) -> pd.DataFrame:
    work = add_day_ranks(scored)
    if strategy == "rolling_top3":
        return work.sort_values(["日期", "dl_score"], ascending=[True, False]).groupby("日期").head(3)
    if strategy == "rolling_top5":
        return work.sort_values(["日期", "dl_score"], ascending=[True, False]).groupby("日期").head(5)
    if strategy == "rolling_market_gate_top3":
        return work[work["market_strength_score"] >= 0.55].sort_values(["日期", "dl_score"], ascending=[True, False]).groupby("日期").head(3)
    if strategy == "rolling_exception_top1":
        threshold = work["dl_score"].quantile(0.90)
        return work[(work["market_strength_score"] < 0.55) & (work["dl_score"] >= threshold)].sort_values(["日期", "dl_score"], ascending=[True, False]).groupby("日期").head(1)
    if strategy == "rolling_adaptive":
        high = work[work["market_strength_score"] >= 0.65].sort_values(["日期", "dl_score"], ascending=[True, False]).groupby("日期").head(5)
        normal = work[(work["market_strength_score"] >= 0.55) & (work["market_strength_score"] < 0.65)].sort_values(["日期", "dl_score"], ascending=[True, False]).groupby("日期").head(3)
        threshold = work["dl_score"].quantile(0.90)
        exception = work[(work["market_strength_score"] < 0.55) & (work["dl_score"] >= threshold)].sort_values(["日期", "dl_score"], ascending=[True, False]).groupby("日期").head(1)
        return pd.concat([high, normal, exception], ignore_index=False)
    raise ValueError(strategy)


def added_days_summary(full_universe: pd.DataFrame, candidate: pd.DataFrame, baseline: pd.DataFrame, label: str) -> dict:
    base_days = set(baseline["日期"].unique())
    added = candidate[~candidate["日期"].isin(base_days)].copy()
    out = summarize_strategy(full_universe, added, f"{label}_added_days")
    return out


def latest_candidates(scored: pd.DataFrame, strategy: str, passed: bool) -> pd.DataFrame:
    latest_date = scored["日期"].max()
    latest = scored[scored["日期"] == latest_date].copy()
    picked = pick_strategy(latest, strategy)
    if picked.empty:
        return pd.DataFrame(
            columns=[
                "signal_date",
                "stock_id",
                "stock_name",
                "strategy",
                "research_score",
                "market_strength_score",
                "status",
                "formal_candidate",
            ]
        )
    picked["signal_date"] = picked["日期"].dt.strftime("%Y-%m-%d")
    picked["stock_id"] = picked["股票代號"]
    picked["stock_name"] = picked["股票名稱"]
    picked["strategy"] = strategy
    picked["research_score"] = picked["dl_score"]
    picked["status"] = "追蹤中"
    picked["formal_candidate"] = bool(passed)
    return picked[
        [
            "signal_date",
            "stock_id",
            "stock_name",
            "strategy",
            "research_score",
            "market_strength_score",
            "status",
            "formal_candidate",
        ]
    ]


def write_formal(config: dict, active: bool, reason: str, tracking: pd.DataFrame, strategy: str) -> None:
    if active:
        result = f"rolling deep learning strategy enabled: {strategy}"
        if tracking.empty:
            result += "; current latest date has no candidate"
    else:
        result = "deep learning Top 3 formal strategy remains enabled"
        if tracking.empty:
            result += "; current latest date has no candidate"
    FORMAL_STATUS_PATH.write_text(
        "\n".join(
            [
                "# Formal Status",
                "",
                "- Status: active",
                f"- Result: {result}",
                f"- Reason: {reason}",
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
                "estimated_success_probability",
                "calibration_sample_count",
                "actual_hit_rate",
                "avg_10d_high_close_return",
                "main_basis",
                "main_risk",
            ]
        )
        if not active:
            return
        for _, row in tracking.iterrows():
            writer.writerow(
                [
                    row["signal_date"],
                    row["stock_id"],
                    row["stock_name"],
                    "",
                    "",
                    "",
                    "",
                    row["strategy"],
                    "tracking only until 10 trading days complete",
                ]
            )


def main() -> None:
    EXPERIMENT_DIR.mkdir(exist_ok=True)
    FORMAL_DIR.mkdir(exist_ok=True)
    config = load_config()
    paths = validate_inputs(config)
    stock = read_csv(paths["stock_daily_all"])
    market = read_csv(paths["market_daily"])
    theme = read_csv(paths["theme_group"])
    data = build_dataset(stock, market, theme)
    features, _ = sequence_features(data, lookback=20)
    data = data[features["has_full_20d_history"] == 1].copy()
    features = features.loc[data.index].drop(columns=["has_full_20d_history"])

    holdout_start = pd.Timestamp(config["time_split"]["holdout_start"])
    latest_full_date = data["日期"].max()
    eval_months = month_starts(holdout_start, latest_full_date)
    windows = [6, 9, 12]
    scored_parts = []
    for month in eval_months:
        for window in windows:
            scored = score_with_window(data, features, month, window)
            if not scored.empty:
                scored_parts.append(scored)
    rolling_scores = pd.concat(scored_parts, ignore_index=True) if scored_parts else pd.DataFrame()

    strategies = [
        "rolling_top3",
        "rolling_top5",
        "rolling_market_gate_top3",
        "rolling_exception_top1",
        "rolling_adaptive",
    ]
    backtest_rows = []
    monthly_rows = []
    added_rows = []
    selected_picks = {}

    # Rebuild fourth-round baseline on the same holdout dates as comparison.
    baseline_train, _, _, _ = score_datasets(config)
    baseline_holdout = baseline_train[baseline_train["日期"] >= holdout_start].copy()
    baseline_strict = strategy_candidates(baseline_holdout, "strict_gate_top3", np.inf, np.inf)
    baseline_summary = summarize_strategy(baseline_holdout, baseline_strict, "fourth_round_strict_gate_top3")
    baseline_summary["window_months"] = "baseline"
    backtest_rows.append(baseline_summary)

    for window in windows:
        scored_w = rolling_scores[rolling_scores["rolling_window_months"] == window].copy()
        if scored_w.empty:
            continue
        for strategy in strategies:
            picked = pick_strategy(scored_w, strategy)
            selected_picks[(window, strategy)] = picked
            row = summarize_strategy(scored_w, picked, strategy)
            row["window_months"] = window
            backtest_rows.append(row)
            added = added_days_summary(scored_w, picked, baseline_strict, f"{window}m_{strategy}")
            added["window_months"] = window
            added_rows.append(added)
            for month, part in picked.groupby("eval_month"):
                month_universe = scored_w[scored_w["eval_month"] == month]
                m = summarize_strategy(month_universe, part, strategy)
                m["window_months"] = window
                m["eval_month"] = month
                monthly_rows.append(m)

    backtest = pd.DataFrame(backtest_rows)
    monthly = pd.DataFrame(monthly_rows)
    added_days = pd.DataFrame(added_rows)
    backtest.to_csv(BACKTEST_PATH, index=False, encoding="utf-8-sig")
    monthly.to_csv(MONTHLY_PATH, index=False, encoding="utf-8-sig")
    added_days.to_csv(ADDED_DAYS_PATH, index=False, encoding="utf-8-sig")

    base = backtest[backtest["strategy"] == "fourth_round_strict_gate_top3"].iloc[0]
    candidates = backtest[backtest["window_months"] != "baseline"].copy()
    passed_rows = []
    for _, row in candidates.iterrows():
        added = added_days[
            (added_days["window_months"].astype(str) == str(row["window_months"]))
            & (added_days["strategy"] == f"{int(row['window_months'])}m_{row['strategy']}_added_days")
        ]
        added_ok = True
        if not added.empty and int(added.iloc[0]["rows"]) > 0:
            added_ok = bool(added.iloc[0]["success_lift"] >= 0 and added.iloc[0]["return_lift"] >= 0)
        month_rows = monthly[
            (monthly["window_months"].astype(str) == str(row["window_months"]))
            & (monthly["strategy"] == row["strategy"])
        ]
        weak_months = 0
        if not month_rows.empty:
            weak_months = int(((month_rows["success_lift"] < 0) | (month_rows["return_lift"] < 0)).sum())
        passes = bool(
            row["days"] > base["days"]
            and row["success_rate"] >= base["success_rate"] - 0.03
            and row["success_lift"] >= 0.05
            and row["return_lift"] > 0
            and row["top_stock_share"] <= 0.20
            and row["top_industry_share"] <= 0.50
            and added_ok
            and weak_months <= 1
        )
        if passes:
            out = row.to_dict()
            out["weak_months"] = weak_months
            passed_rows.append(out)

    if passed_rows:
        chosen = pd.DataFrame(passed_rows).sort_values(["days", "success_lift", "return_lift"], ascending=False).iloc[0]
        active = True
        selected_strategy = str(chosen["strategy"])
        selected_window = int(chosen["window_months"])
        reason = f"{selected_window}m {selected_strategy} passed rolling gates and replaces fourth-round Top 3"
    else:
        active = False
        selected_strategy = "fourth_round_strict_gate_top3"
        selected_window = 0
        reason = "no rolling strategy passed all gates; keeping fourth-round Top 3"

    # Latest research scoring uses the most recent rolling window if available; formal only if active.
    latest_scored = rolling_scores[rolling_scores["rolling_window_months"] == (selected_window or 12)].copy()
    tracking = latest_candidates(latest_scored, selected_strategy if active else "rolling_market_gate_top3", active)
    tracking.to_csv(LATEST_PATH, index=False, encoding="utf-8-sig")
    write_formal(config, active, reason, tracking, selected_strategy)

    best = backtest.sort_values(["days", "success_lift", "return_lift"], ascending=False).head(10)
    lines = [
        "# Rolling Deep Learning Strategy Decision",
        "",
        f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "- Data sources: three allowed CSV inputs only",
        "- Goal: test whether recent rolling learning improves frequency without lowering quality",
        f"- Status: {'replace_fourth_round' if active else 'keep_fourth_round'}",
        f"- Selected strategy: {selected_strategy}",
        f"- Selected window months: {selected_window if selected_window else '-'}",
        f"- Latest candidate count: {len(tracking) if active else 0}",
        "",
        "## Fourth Round Baseline",
        "",
        f"- Days: {int(base['days'])}",
        f"- Success rate: {fmt_pct(base['success_rate'])}",
        f"- Success lift: {fmt_pct(base['success_lift'])}",
        f"- Return lift: {fmt_pct(base['return_lift'])}",
        "",
        "## Best Rolling Strategies",
        "",
    ]
    for _, row in best.iterrows():
        lines.append(
            f"- {row['window_months']} {row['strategy']}: days {int(row['days'])}, "
            f"success {fmt_pct(row['success_rate'])}, lift {fmt_pct(row['success_lift'])}, "
            f"return lift {fmt_pct(row['return_lift'])}, industry share {fmt_pct(row['top_industry_share'])}"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- {reason}.",
            "",
            "## Output Files",
            "",
            f"- Backtest: `{BACKTEST_PATH}`",
            f"- Monthly: `{MONTHLY_PATH}`",
            f"- Added days: `{ADDED_DAYS_PATH}`",
            f"- Latest: `{LATEST_PATH}`",
            f"- Formal status: `{FORMAL_STATUS_PATH}`",
        ]
    )
    DECISION_PATH.write_text("\n".join(lines), encoding="utf-8")
    print("OK: rolling deep learning strategy search completed")
    print(f"STATUS: {'replace_fourth_round' if active else 'keep_fourth_round'}")
    print(f"DECISION: {DECISION_PATH}")


if __name__ == "__main__":
    main()

