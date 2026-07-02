from __future__ import annotations

import csv
import itertools
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
    split_masks,
)
from run_deep_learning_frequency_expansion_experiment import (  # noqa: E402
    score_datasets,
    strategy_candidates,
)
from run_deep_learning_selection_experiment import (  # noqa: E402
    build_dataset,
    fmt_pct,
    predict_mlp,
    sequence_features,
    standardize,
    train_mlp,
    validate_inputs,
)


EXPERIMENT_DIR = PROJECT_ROOT / "research_layer"
FORMAL_DIR = PROJECT_ROOT / "research_layer" / "formal_write_disabled"

DECISION_PATH = EXPERIMENT_DIR / "integrated_main_model_decision.md"
BACKTEST_PATH = EXPERIMENT_DIR / "integrated_main_model_backtest.csv"
CALIBRATION_PATH = EXPERIMENT_DIR / "integrated_main_model_calibration.csv"
FAILURE_PATH = EXPERIMENT_DIR / "integrated_main_model_failure_learning.csv"
EPISODE_PATH = EXPERIMENT_DIR / "integrated_main_model_episode_tracking.csv"
LATEST_PATH = EXPERIMENT_DIR / "integrated_main_model_latest.csv"
FORMAL_STATUS_PATH = FORMAL_DIR / "formal_status.md"
FORMAL_CANDIDATES_PATH = FORMAL_DIR / "formal_candidates.csv"

FOURTH_ROUND_DAYS = 36
FOURTH_ROUND_SUCCESS_RATE = 0.7962962962962963
EPISODE_GAP_TRADING_DAYS = 10


def add_risk_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["股票代號", "日期"]).copy()
    g = out.groupby("股票代號", sort=False)
    future_close = pd.concat([g["收盤價"].shift(-i) for i in range(1, 11)], axis=1)
    out["future_10d_low_close"] = future_close.min(axis=1)
    out["future_10d_low_close_return"] = out["future_10d_low_close"] / out["buy_open_next"] - 1.0
    out["stock_trading_index"] = g.cumcount()
    out["underperform_market_label"] = (
        out["future_10d_high_close_return"] < out["daily_market_avg_return"]
    ).astype(int)
    out["risk_label"] = (
        (out["target_success"] == 0)
        | (out["future_10d_low_close_return"] <= -0.03)
        | (out["underperform_market_label"] == 1)
    ).astype(int)

    prev_success_idx = g["target_success"].transform(
        lambda s: s.shift(1).rolling(EPISODE_GAP_TRADING_DAYS, min_periods=1).max()
    )
    out["episode_start_label"] = ((out["target_success"] == 1) & (prev_success_idx.fillna(0) == 0)).astype(int)
    return out


def prepare_data() -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    config = load_config()
    paths = validate_inputs(config)
    stock = read_csv(paths["stock_daily_all"])
    market = read_csv(paths["market_daily"])
    theme = read_csv(paths["theme_group"])

    train_data = add_risk_labels(build_dataset(stock, market, theme))
    scoring_data = add_risk_labels(build_scoring_dataset(stock, market, theme))

    train_features, _ = sequence_features(train_data, lookback=20)
    train_data = train_data[train_features["has_full_20d_history"] == 1].copy()
    train_features = train_features.loc[train_data.index].drop(columns=["has_full_20d_history"])

    scoring_features, _ = sequence_features(scoring_data, lookback=20)
    scoring_data = scoring_data[scoring_features["has_full_20d_history"] == 1].copy()
    scoring_features = scoring_features.loc[scoring_data.index].drop(columns=["has_full_20d_history"])
    scoring_features = scoring_features.reindex(columns=train_features.columns, fill_value=0.0)
    return config, train_data, train_features, scoring_data, scoring_features


def score_multitask(
    train_data: pd.DataFrame,
    train_features: pd.DataFrame,
    scoring_data: pd.DataFrame,
    scoring_features: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, list[float]]:
    masks = split_masks(train_data, config)
    train_mask = masks["train"].to_numpy()
    X_train_all, mean, std = standardize(train_features.to_numpy(dtype=np.float32), train_mask)
    y = train_data[
        ["target_success", "risk_label", "episode_start_label", "relative_top20_label"]
    ].to_numpy(dtype=np.float32)
    model, loss = train_mlp(X_train_all[train_mask], y[train_mask], hidden=48, epochs=90, seed=901)

    train_scored = train_data.copy()
    train_pred = predict_mlp(X_train_all, model)
    train_scored["success_head"] = train_pred[:, 0]
    train_scored["risk_head"] = train_pred[:, 1]
    train_scored["episode_head"] = train_pred[:, 2]
    train_scored["rank_head"] = train_pred[:, 3]

    scoring_x = ((scoring_features.to_numpy(dtype=np.float32) - mean) / std).astype(np.float32)
    scoring_pred = predict_mlp(scoring_x, model)
    scoring_scored = scoring_data.copy()
    scoring_scored["success_head"] = scoring_pred[:, 0]
    scoring_scored["risk_head"] = scoring_pred[:, 1]
    scoring_scored["episode_head"] = scoring_pred[:, 2]
    scoring_scored["rank_head"] = scoring_pred[:, 3]
    return train_scored, scoring_scored, loss


def apply_score(df: pd.DataFrame, weights: tuple[float, float, float, float]) -> pd.DataFrame:
    out = df.copy()
    sw, rw, ew, fw = weights
    out["integrated_decision_score"] = (
        sw * out["success_head"]
        + rw * out["rank_head"]
        + ew * out["episode_head"]
        - fw * out["risk_head"]
    )
    return out


def select_top3_episode_dedup(df: pd.DataFrame, score_col: str = "integrated_decision_score") -> pd.DataFrame:
    if df.empty:
        return df.copy()
    work = df.sort_values(["日期", score_col], ascending=[True, False]).copy()
    selected = []
    last_pick_idx: dict[str, int] = {}
    for date, day in work.groupby("日期", sort=True):
        day_selected = 0
        for _, row in day.iterrows():
            stock_id = str(row["股票代號"])
            idx = int(row["stock_trading_index"])
            if stock_id in last_pick_idx and idx - last_pick_idx[stock_id] <= EPISODE_GAP_TRADING_DAYS:
                continue
            selected.append(row)
            last_pick_idx[stock_id] = idx
            day_selected += 1
            if day_selected >= 3:
                break
    return pd.DataFrame(selected)


def summary_row(full: pd.DataFrame, picked: pd.DataFrame, split: str, strategy: str, weights: str) -> dict:
    if picked.empty:
        return {
            "split": split,
            "strategy": strategy,
            "weights": weights,
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
        "split": split,
        "strategy": strategy,
        "weights": weights,
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


def fourth_round_summary(config: dict, holdout_start: pd.Timestamp) -> dict:
    scored, _, _, _ = score_datasets(config)
    holdout = scored[scored["日期"] >= holdout_start].copy()
    strict = strategy_candidates(holdout, "strict_gate_top3", np.inf, np.inf)
    return summary_row(holdout, strict, "holdout", "fourth_round_top3", "baseline")


def tune_weights(dev: pd.DataFrame) -> tuple[tuple[float, float, float, float], pd.DataFrame]:
    candidates = []
    grid = list(itertools.product([0.8, 1.0, 1.2], [0.4, 0.7, 1.0], [0.2, 0.5, 0.8], [0.4, 0.7, 1.0]))
    for weights in grid:
        scored = apply_score(dev, weights)
        picked = select_top3_episode_dedup(scored)
        row = summary_row(scored, picked, "development", "integrated_top3", ",".join(map(str, weights)))
        row["weight_tuple"] = weights
        candidates.append(row)
    table = pd.DataFrame(candidates)
    feasible = table[
        (table["rows"] > 0)
        & (table["success_lift"] >= 0)
        & (table["return_lift"] >= 0)
        & (table["top_stock_share"] <= 0.25)
        & (table["top_industry_share"] <= 0.60)
    ].copy()
    if feasible.empty:
        feasible = table.copy()
    feasible = feasible.sort_values(["success_lift", "return_lift", "success_rate", "days"], ascending=False)
    return feasible.iloc[0]["weight_tuple"], table


def band_table(df: pd.DataFrame, score_col: str, target_col: str, split: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    work = df.copy()
    work["band"] = pd.qcut(work[score_col].rank(method="first"), q=min(4, len(work)), labels=False, duplicates="drop")
    rows = []
    for band, part in work.groupby("band"):
        rows.append(
            {
                "split": split,
                "score_col": score_col,
                "band": int(band),
                "rows": len(part),
                "avg_score": part[score_col].mean(),
                "actual_rate": part[target_col].mean(),
                "avg_10d_high_close_return": part["future_10d_high_close_return"].mean(),
            }
        )
    return pd.DataFrame(rows)


def latest_candidates(scored: pd.DataFrame, passed: bool) -> pd.DataFrame:
    latest_date = scored["日期"].max()
    latest = scored[scored["日期"] == latest_date].copy()
    picked = select_top3_episode_dedup(latest).copy()
    if picked.empty:
        return pd.DataFrame(
            columns=[
                "signal_date",
                "stock_id",
                "stock_name",
                "research_score",
                "success_head",
                "risk_head",
                "episode_head",
                "rank_head",
                "status",
                "formal_candidate",
            ]
        )
    picked["signal_date"] = picked["日期"].dt.strftime("%Y-%m-%d")
    picked["stock_id"] = picked["股票代號"]
    picked["stock_name"] = picked["股票名稱"]
    picked["research_score"] = picked["integrated_decision_score"]
    picked["status"] = "追蹤中"
    picked["formal_candidate"] = bool(passed)
    return picked[
        [
            "signal_date",
            "stock_id",
            "stock_name",
            "research_score",
            "success_head",
            "risk_head",
            "episode_head",
            "rank_head",
            "status",
            "formal_candidate",
        ]
    ]


def write_formal(config: dict, passed: bool, reason: str, latest: pd.DataFrame) -> None:
    result = "deep learning Top 3 formal strategy remains enabled"
    if passed:
        result = "integrated deep learning Top 3 formal strategy enabled"
    if latest.empty:
        result += "; current latest date has no candidate"
    FORMAL_STATUS_PATH.write_text(
        "\n".join(["# Formal Status", "", "- Status: active", f"- Result: {result}", f"- Reason: {reason}", ""]),
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
                    "integrated_deep_learning_score",
                    "tracking only until 10 trading days complete",
                ]
            )


def main() -> None:
    EXPERIMENT_DIR.mkdir(exist_ok=True)
    FORMAL_DIR.mkdir(exist_ok=True)
    config, train_data, train_features, scoring_data, scoring_features = prepare_data()
    train_scored, scoring_scored, loss = score_multitask(train_data, train_features, scoring_data, scoring_features, config)

    masks = split_masks(train_scored, config)
    dev = train_scored[masks["development"]].copy()
    holdout = train_scored[masks["holdout"]].copy()
    holdout_start = pd.Timestamp(config["time_split"]["holdout_start"])

    weights, tuning_table = tune_weights(dev)
    weights_text = ",".join(map(str, weights))
    tuning_table.drop(columns=["weight_tuple"]).to_csv(
        EXPERIMENT_DIR / "integrated_main_model_weight_search.csv",
        index=False,
        encoding="utf-8-sig",
    )

    dev_scored = apply_score(dev, weights)
    holdout_scored = apply_score(holdout, weights)
    scoring_scored = apply_score(scoring_scored, weights)
    dev_picked = select_top3_episode_dedup(dev_scored)
    holdout_picked = select_top3_episode_dedup(holdout_scored)

    fourth = fourth_round_summary(config, holdout_start)
    rows = [
        summary_row(dev_scored, dev_picked, "development", "integrated_top3", weights_text),
        summary_row(holdout_scored, holdout_picked, "holdout", "integrated_top3", weights_text),
        fourth,
    ]
    backtest = pd.DataFrame(rows)
    backtest.to_csv(BACKTEST_PATH, index=False, encoding="utf-8-sig")

    calibration = pd.concat(
        [
            band_table(holdout_scored, "integrated_decision_score", "target_success", "holdout"),
            band_table(holdout_scored, "success_head", "target_success", "holdout"),
        ],
        ignore_index=True,
    )
    calibration.to_csv(CALIBRATION_PATH, index=False, encoding="utf-8-sig")
    failure = band_table(holdout_scored, "risk_head", "risk_label", "holdout")
    failure.to_csv(FAILURE_PATH, index=False, encoding="utf-8-sig")

    episode_tracking = holdout_picked[
        [
            "日期",
            "股票代號",
            "股票名稱",
            "主分類",
            "integrated_decision_score",
            "success_head",
            "risk_head",
            "episode_head",
            "rank_head",
            "target_success",
            "risk_label",
            "future_10d_high_close_return",
            "daily_market_avg_return",
        ]
    ].copy()
    episode_tracking.to_csv(EPISODE_PATH, index=False, encoding="utf-8-sig")

    integrated = backtest[(backtest["split"] == "holdout") & (backtest["strategy"] == "integrated_top3")].iloc[0]
    fourth_row = backtest[backtest["strategy"] == "fourth_round_top3"].iloc[0]
    cal_sorted = calibration[calibration["score_col"] == "integrated_decision_score"].sort_values("band")
    fail_sorted = failure.sort_values("band")
    calibration_ok = bool(len(cal_sorted) >= 2 and cal_sorted.iloc[-1]["actual_rate"] >= cal_sorted.iloc[0]["actual_rate"])
    risk_ok = bool(len(fail_sorted) >= 2 and fail_sorted.iloc[-1]["actual_rate"] >= fail_sorted.iloc[0]["actual_rate"])
    month_count = holdout_picked["日期"].dt.strftime("%Y-%m").nunique() if not holdout_picked.empty else 0
    passed = bool(
        integrated["success_rate"] >= FOURTH_ROUND_SUCCESS_RATE - 0.03
        and integrated["success_lift"] >= 0.08
        and integrated["return_lift"] > 0
        and calibration_ok
        and risk_ok
        and integrated["top_stock_share"] <= 0.20
        and integrated["top_industry_share"] <= 0.50
        and month_count >= 2
    )
    reason = "integrated model passed all gates and replaces fourth-round Top 3"
    if not passed:
        reason = "integrated model did not pass all gates; keeping fourth-round Top 3"

    latest = latest_candidates(scoring_scored, passed)
    latest.to_csv(LATEST_PATH, index=False, encoding="utf-8-sig")
    write_formal(config, passed, reason, latest)

    lines = [
        "# Integrated Main Model Decision",
        "",
        f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "- Data sources: three allowed CSV inputs only",
        "- Goal: replace branch experiments with one integrated deep learning decision layer",
        f"- Status: {'integrated_model_pass' if passed else 'keep_fourth_round'}",
        f"- Selected weights: {weights_text}",
        f"- Training loss: {loss[0]:.6f} -> {loss[-1]:.6f}",
        "",
        "## Holdout Comparison",
        "",
        f"- Integrated days: {int(integrated['days'])}",
        f"- Integrated success rate: {fmt_pct(integrated['success_rate'])}",
        f"- Integrated success lift: {fmt_pct(integrated['success_lift'])}",
        f"- Integrated return lift: {fmt_pct(integrated['return_lift'])}",
        f"- Fourth-round days: {int(fourth_row['days'])}",
        f"- Fourth-round success rate: {fmt_pct(fourth_row['success_rate'])}",
        f"- Fourth-round success lift: {fmt_pct(fourth_row['success_lift'])}",
        f"- Fourth-round return lift: {fmt_pct(fourth_row['return_lift'])}",
        "",
        "## Gates",
        "",
        f"- Calibration ordering valid: {calibration_ok}",
        f"- Failure risk ordering valid: {risk_ok}",
        f"- Top stock share: {fmt_pct(integrated['top_stock_share'])}",
        f"- Top industry share: {fmt_pct(integrated['top_industry_share'])}",
        f"- Holdout active months: {month_count}",
        "",
        "## Decision",
        "",
        f"- {reason}.",
        "",
        "## Output Files",
        "",
        f"- Backtest: `{BACKTEST_PATH}`",
        f"- Calibration: `{CALIBRATION_PATH}`",
        f"- Failure learning: `{FAILURE_PATH}`",
        f"- Episode tracking: `{EPISODE_PATH}`",
        f"- Latest: `{LATEST_PATH}`",
        f"- Formal status: `{FORMAL_STATUS_PATH}`",
    ]
    DECISION_PATH.write_text("\n".join(lines), encoding="utf-8")
    print("OK: integrated deep learning main model completed")
    print(f"STATUS: {'integrated_model_pass' if passed else 'keep_fourth_round'}")
    print(f"DECISION: {DECISION_PATH}")


if __name__ == "__main__":
    main()

