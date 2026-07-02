from __future__ import annotations

import csv
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from run_deep_learning_calibrated_candidate_experiment import (  # noqa: E402
    build_scoring_dataset,
    load_config,
    read_csv,
)
from run_deep_learning_frequency_expansion_experiment import (  # noqa: E402
    add_day_ranks,
    score_datasets,
    strategy_candidates,
)
from run_deep_learning_selection_experiment import (  # noqa: E402
    build_dataset,
    fmt_pct,
    sequence_features,
    standardize,
    train_mlp,
    validate_inputs,
)


EXPERIMENT_DIR = PROJECT_ROOT / "research_layer"
FORMAL_DIR = PROJECT_ROOT / "research_layer" / "formal_write_disabled"

DECISION_PATH = EXPERIMENT_DIR / "deep_learning_exception_gate_decision.md"
BACKTEST_PATH = EXPERIMENT_DIR / "deep_learning_exception_gate_backtest.csv"
MONTHLY_PATH = EXPERIMENT_DIR / "deep_learning_exception_gate_monthly.csv"
CANDIDATES_PATH = EXPERIMENT_DIR / "deep_learning_exception_gate_candidates.csv"
LATEST_PATH = EXPERIMENT_DIR / "deep_learning_exception_gate_latest.csv"
FORMAL_STATUS_PATH = FORMAL_DIR / "formal_status.md"
FORMAL_CANDIDATES_PATH = FORMAL_DIR / "formal_candidates.csv"


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def month_starts(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    months = pd.date_range(start=start.replace(day=1), end=end, freq="MS")
    return [pd.Timestamp(m) for m in months]


def load_clean_data() -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    config = load_config()
    paths = validate_inputs(config)
    stock = read_csv(paths["stock_daily_all"])
    market = read_csv(paths["market_daily"])
    theme = read_csv(paths["theme_group"])

    train_data = build_dataset(stock, market, theme)
    scoring_data = build_scoring_dataset(stock, market, theme)
    features, _ = sequence_features(train_data, lookback=20)
    train_data = train_data[features["has_full_20d_history"] == 1].copy()
    features = features.loc[train_data.index].drop(columns=["has_full_20d_history"])

    scoring_features, _ = sequence_features(scoring_data, lookback=20)
    scoring_data = scoring_data[scoring_features["has_full_20d_history"] == 1].copy()
    scoring_features = scoring_features.loc[scoring_data.index].drop(columns=["has_full_20d_history"])
    scoring_features = scoring_features.reindex(columns=features.columns, fill_value=0.0)
    return config, train_data, features, scoring_data, scoring_features


def score_window(
    data: pd.DataFrame,
    features: pd.DataFrame,
    eval_month: pd.Timestamp,
    months: int,
) -> pd.DataFrame:
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
    model, _ = train_mlp(
        X[train_mask.to_numpy()],
        y[train_mask.to_numpy()],
        hidden=32,
        epochs=45,
        seed=700 + months + eval_month.month,
    )
    scored = data.loc[eval_mask].copy()
    eval_x = X[eval_mask.to_numpy()]
    scored[f"score_{months}m"] = sigmoid(
        (np.maximum(eval_x @ model["w1"] + model["b1"], 0.0) @ model["w2"] + model["b2"])[:, 0]
    )
    scored["eval_month"] = eval_start.strftime("%Y-%m")
    return scored


def build_rolling_scores(data: pd.DataFrame, features: pd.DataFrame, holdout_start: pd.Timestamp) -> pd.DataFrame:
    latest_full_date = data["日期"].max()
    months = month_starts(holdout_start, latest_full_date)
    by_window = []
    for window in [6, 9, 12]:
        parts = []
        for month in months:
            scored = score_window(data, features, month, window)
            if not scored.empty:
                parts.append(scored)
        if parts:
            cols = ["日期", "股票代號", f"score_{window}m", "eval_month"]
            by_window.append(pd.concat(parts, ignore_index=True)[cols])
    if not by_window:
        return pd.DataFrame()
    out = by_window[0]
    for part in by_window[1:]:
        out = out.merge(part, on=["日期", "股票代號", "eval_month"], how="outer")
    score_cols = [c for c in out.columns if c.startswith("score_")]
    out["ensemble_score_mean"] = out[score_cols].mean(axis=1)
    out["ensemble_score_min"] = out[score_cols].min(axis=1)
    out["ensemble_agree_count"] = out[score_cols].notna().sum(axis=1)
    return out


def fourth_round_baseline(config: dict, holdout_start: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    scored, _, _, _ = score_datasets(config)
    holdout = scored[scored["日期"] >= holdout_start].copy()
    strict = strategy_candidates(holdout, "strict_gate_top3", np.inf, np.inf).copy()
    return holdout, strict


def summarize(full: pd.DataFrame, picked: pd.DataFrame, strategy: str) -> dict:
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
    base_full = full[full["日期"].isin(selected_days)]
    base = base_full.groupby("日期").agg(
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


def pick_exception(work: pd.DataFrame, strategy: str) -> pd.DataFrame:
    if strategy == "exception_6m_top1":
        eligible = work[work["score_6m"].notna()].copy()
        eligible["exception_score"] = eligible["score_6m"]
    elif strategy == "exception_9m_top1":
        eligible = work[work["score_9m"].notna()].copy()
        eligible["exception_score"] = eligible["score_9m"]
    elif strategy == "exception_12m_top1":
        eligible = work[work["score_12m"].notna()].copy()
        eligible["exception_score"] = eligible["score_12m"]
    elif strategy == "exception_ensemble_mean_top1":
        eligible = work[work["ensemble_agree_count"] == 3].copy()
        eligible["exception_score"] = eligible["ensemble_score_mean"]
    elif strategy == "exception_ensemble_all_high_top1":
        eligible = work[
            (work["ensemble_agree_count"] == 3)
            & (work["ensemble_score_min"] >= work["ensemble_score_min"].quantile(0.80))
        ].copy()
        eligible["exception_score"] = eligible["ensemble_score_min"]
    else:
        raise ValueError(strategy)
    if eligible.empty:
        return eligible
    eligible = add_day_ranks(eligible, "exception_score")
    return eligible.sort_values(["日期", "exception_score"], ascending=[True, False]).groupby("日期").head(1)


def monthly_summary(full: pd.DataFrame, picked: pd.DataFrame, strategy: str) -> pd.DataFrame:
    rows = []
    if picked.empty:
        return pd.DataFrame()
    for month, part in picked.groupby("eval_month"):
        universe = full[full["eval_month"] == month]
        row = summarize(universe, part, strategy)
        row["eval_month"] = month
        rows.append(row)
    return pd.DataFrame(rows)


def latest_candidates(full: pd.DataFrame, picked_strategy: str, passed: bool) -> pd.DataFrame:
    latest_date = full["日期"].max()
    latest = full[full["日期"] == latest_date].copy()
    picked = pick_exception(latest, picked_strategy).copy()
    if picked.empty:
        return pd.DataFrame(
            columns=[
                "signal_date",
                "stock_id",
                "stock_name",
                "candidate_type",
                "research_score",
                "market_strength_score",
                "status",
                "formal_candidate",
            ]
        )
    picked["signal_date"] = picked["日期"].dt.strftime("%Y-%m-%d")
    picked["stock_id"] = picked["股票代號"]
    picked["stock_name"] = picked["股票名稱"]
    picked["candidate_type"] = "exception_top1"
    picked["research_score"] = picked["exception_score"]
    picked["status"] = "追蹤中"
    picked["formal_candidate"] = bool(passed)
    return picked[
        [
            "signal_date",
            "stock_id",
            "stock_name",
            "candidate_type",
            "research_score",
            "market_strength_score",
            "status",
            "formal_candidate",
        ]
    ]


def write_formal(config: dict, passed: bool, reason: str, latest: pd.DataFrame) -> None:
    if passed:
        result = "deep learning Top 3 plus exception Top 1 formal strategy enabled"
        if latest.empty:
            result += "; current latest date has no exception candidate"
    else:
        result = "deep learning Top 3 formal strategy remains enabled"
        if latest.empty:
            result += "; current latest date has no exception candidate"
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
        if not passed:
            return
        for _, row in latest.iterrows():
            writer.writerow(
                [
                    row["signal_date"],
                    row["stock_id"],
                    row["stock_name"],
                    "",
                    "",
                    "",
                    "",
                    row["candidate_type"],
                    "tracking only until 10 trading days complete",
                ]
            )


def main() -> None:
    EXPERIMENT_DIR.mkdir(exist_ok=True)
    FORMAL_DIR.mkdir(exist_ok=True)
    config, data, features, scoring_data, _ = load_clean_data()
    holdout_start = pd.Timestamp(config["time_split"]["holdout_start"])

    rolling_scores = build_rolling_scores(data, features, holdout_start)
    if rolling_scores.empty:
        raise RuntimeError("rolling scoring produced no rows")

    fourth_full, fourth_picked = fourth_round_baseline(config, holdout_start)
    fourth_days = set(fourth_picked["日期"].unique())
    full = scoring_data[scoring_data["日期"] >= holdout_start].merge(
        rolling_scores,
        on=["日期", "股票代號"],
        how="inner",
    )
    exception_universe = full[~full["日期"].isin(fourth_days)].copy()

    strategies = [
        "exception_6m_top1",
        "exception_9m_top1",
        "exception_12m_top1",
        "exception_ensemble_mean_top1",
        "exception_ensemble_all_high_top1",
    ]
    backtest_rows = []
    monthly_parts = []
    candidate_parts = []
    selected = {}
    for strategy in strategies:
        picked = pick_exception(exception_universe, strategy).copy()
        if not picked.empty:
            picked["strategy"] = strategy
            candidate_parts.append(
                picked[
                    [
                        "日期",
                        "股票代號",
                        "股票名稱",
                        "主分類",
                        "strategy",
                        "exception_score",
                        "market_strength_score",
                        "target_success",
                        "future_10d_high_close_return",
                        "daily_market_success_rate",
                        "daily_market_avg_return",
                        "eval_month",
                    ]
                ]
            )
        selected[strategy] = picked
        backtest_rows.append(summarize(exception_universe, picked, strategy))
        monthly = monthly_summary(exception_universe, picked, strategy)
        if not monthly.empty:
            monthly_parts.append(monthly)

    backtest = pd.DataFrame(backtest_rows)
    monthly_table = pd.concat(monthly_parts, ignore_index=True) if monthly_parts else pd.DataFrame()
    candidates = pd.concat(candidate_parts, ignore_index=True) if candidate_parts else pd.DataFrame()
    backtest.to_csv(BACKTEST_PATH, index=False, encoding="utf-8-sig")
    monthly_table.to_csv(MONTHLY_PATH, index=False, encoding="utf-8-sig")
    candidates.to_csv(CANDIDATES_PATH, index=False, encoding="utf-8-sig")

    passed_rows = []
    for _, row in backtest.iterrows():
        month_rows = monthly_table[monthly_table["strategy"] == row["strategy"]]
        weak_months = 0
        if not month_rows.empty:
            weak_months = int(((month_rows["success_lift"] < 0) | (month_rows["return_lift"] < 0)).sum())
        passes = bool(
            row["rows"] >= 20
            and row["success_lift"] >= 0.08
            and row["return_lift"] > 0
            and row["top_stock_share"] <= 0.20
            and row["top_industry_share"] <= 0.50
            and weak_months <= 1
        )
        if passes:
            item = row.to_dict()
            item["weak_months"] = weak_months
            passed_rows.append(item)

    if passed_rows:
        chosen = pd.DataFrame(passed_rows).sort_values(
            ["success_lift", "return_lift", "rows"],
            ascending=False,
        ).iloc[0]
        passed = True
        selected_strategy = str(chosen["strategy"])
        reason = f"{selected_strategy} passed exception gates and can be added to fourth-round Top 3"
    else:
        passed = False
        selected_strategy = "exception_ensemble_mean_top1"
        reason = "no exception Top 1 strategy passed all gates; keeping fourth-round Top 3"

    latest = latest_candidates(exception_universe, selected_strategy, passed)
    latest.to_csv(LATEST_PATH, index=False, encoding="utf-8-sig")
    write_formal(config, passed, reason, latest)

    best = backtest.sort_values(["success_lift", "return_lift", "rows"], ascending=False).head(8)
    lines = [
        "# Deep Learning Exception Gate Decision",
        "",
        f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "- Data sources: three allowed CSV inputs only",
        "- Goal: test exception Top 1 only on dates where fourth-round Top 3 does not act",
        f"- Status: {'exception_gate_pass' if passed else 'keep_fourth_round'}",
        f"- Selected strategy: {selected_strategy if passed else '-'}",
        f"- Latest exception candidate count: {len(latest) if passed else 0}",
        "",
        "## Best Exception Strategies",
        "",
    ]
    for _, row in best.iterrows():
        lines.append(
            f"- {row['strategy']}: rows {int(row['rows'])}, days {int(row['days'])}, "
            f"success {fmt_pct(row['success_rate'])}, lift {fmt_pct(row['success_lift'])}, "
            f"return lift {fmt_pct(row['return_lift'])}, stock share {fmt_pct(row['top_stock_share'])}, "
            f"industry share {fmt_pct(row['top_industry_share'])}"
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
            f"- Candidates: `{CANDIDATES_PATH}`",
            f"- Latest: `{LATEST_PATH}`",
            f"- Formal status: `{FORMAL_STATUS_PATH}`",
        ]
    )
    DECISION_PATH.write_text("\n".join(lines), encoding="utf-8")
    print("OK: deep learning exception gate experiment completed")
    print(f"STATUS: {'exception_gate_pass' if passed else 'keep_fourth_round'}")
    print(f"DECISION: {DECISION_PATH}")


if __name__ == "__main__":
    main()

