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

from run_deep_learning_selection_experiment import (  # noqa: E402
    build_dataset,
    fmt_pct,
    predict_mlp,
    sequence_features,
    standardize,
    train_mlp,
    validate_inputs,
)


CONFIG_PATH = PROJECT_ROOT / "project_config.json"
EXPERIMENT_DIR = PROJECT_ROOT / "research_layer"
FORMAL_DIR = PROJECT_ROOT / "research_layer" / "formal_write_disabled"

DECISION_PATH = EXPERIMENT_DIR / "deep_learning_calibrated_decision.md"
BANDS_PATH = EXPERIMENT_DIR / "deep_learning_calibrated_bands.csv"
TOP3_BACKTEST_PATH = EXPERIMENT_DIR / "deep_learning_calibrated_top3_backtest.csv"
TRACKING_PATH = EXPERIMENT_DIR / "deep_learning_candidate_tracking.csv"
LATEST_RESEARCH_PATH = EXPERIMENT_DIR / "deep_learning_calibrated_latest_research_candidates.csv"
FORMAL_STATUS_PATH = FORMAL_DIR / "formal_status.md"
FORMAL_CANDIDATES_PATH = FORMAL_DIR / "formal_candidates.csv"


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def split_masks(df: pd.DataFrame, config: dict) -> dict[str, pd.Series]:
    cfg = config["time_split"]
    return {
        "train": df["日期"] <= pd.Timestamp(cfg["train_end"]),
        "development": (df["日期"] >= pd.Timestamp(cfg["dev_start"]))
        & (df["日期"] <= pd.Timestamp(cfg["dev_end"])),
        "holdout": df["日期"] >= pd.Timestamp(cfg["holdout_start"]),
    }


def build_scoring_dataset(stock: pd.DataFrame, market: pd.DataFrame, theme: pd.DataFrame) -> pd.DataFrame:
    stock = stock.copy()
    market = market.copy()
    theme = theme.copy()
    stock["日期"] = pd.to_datetime(stock["日期"])
    market["日期"] = pd.to_datetime(market["日期"])
    stock["股票代號"] = stock["股票代號"].astype(str)
    theme["股票代號"] = theme["股票代號"].astype(str)

    stock = stock.sort_values(["股票代號", "日期"]).reset_index(drop=True)
    g = stock.groupby("股票代號", sort=False)
    stock["buy_open_next"] = g["開盤價"].shift(-1)
    future_close = pd.concat([g["收盤價"].shift(-i) for i in range(1, 11)], axis=1)
    stock["future_10d_high_close"] = future_close.max(axis=1)
    stock["future_10d_high_close_return"] = stock["future_10d_high_close"] / stock["buy_open_next"] - 1.0
    stock["has_full_10d_window"] = (
        future_close.notna().all(axis=1)
        & stock["buy_open_next"].notna()
        & (stock["buy_open_next"] > 0)
        & np.isfinite(stock["future_10d_high_close_return"])
    )
    stock["target_success"] = (
        stock["future_10d_high_close"] >= stock["buy_open_next"] * 1.03
    ).astype(int)

    stock["ret_1d"] = g["收盤價"].pct_change(1)
    stock["open_gap"] = stock["開盤價"] / g["收盤價"].shift(1) - 1.0
    stock["intraday_return"] = stock["收盤價"] / stock["開盤價"].replace(0, np.nan) - 1.0
    stock["intraday_range"] = (stock["最高價"] - stock["最低價"]) / stock["收盤價"].replace(0, np.nan)
    stock["close_position"] = (stock["收盤價"] - stock["最低價"]) / (
        stock["最高價"] - stock["最低價"]
    ).replace(0, np.nan)

    for window in [5, 10, 20]:
        stock[f"ret_{window}d"] = g["收盤價"].pct_change(window)
        stock[f"vol_ratio_{window}d"] = stock["成交量(張)"] / g["成交量(張)"].transform(
            lambda s: s.rolling(window, min_periods=max(2, window // 2)).mean()
        )
        stock[f"ma_gap_{window}d"] = stock["收盤價"] / g["收盤價"].transform(
            lambda s: s.rolling(window, min_periods=max(2, window // 2)).mean()
        ) - 1.0

    for col in [
        "外資買賣超(張)",
        "投信買賣超(張)",
        "自營商買賣超(張)",
        "三大法人合計買賣超(張)",
        "融資買進",
        "融資賣出",
        "融資餘額",
    ]:
        if col in stock.columns:
            stock[f"{col}_5d_sum"] = g[col].transform(lambda s: s.rolling(5, min_periods=2).sum())

    market = market.sort_values("日期").reset_index(drop=True)
    for col in ["加權指數收盤", "電子指數收盤", "櫃買指數收盤"]:
        if col in market.columns:
            market[f"{col}_ret_5d"] = market[col].pct_change(5)
            market[f"{col}_ma_gap_20d"] = market[col] / market[col].rolling(20, min_periods=10).mean() - 1.0
    if {"上漲家數", "下跌家數"}.issubset(market.columns):
        market["market_breadth"] = market["上漲家數"] / (
            market["上漲家數"] + market["下跌家數"]
        ).replace(0, np.nan)
    if "大盤成交值(億元)" in market.columns:
        market["market_value_ratio_20d"] = market["大盤成交值(億元)"] / market[
            "大盤成交值(億元)"
        ].rolling(20, min_periods=10).mean()

    strength_cols = [
        c
        for c in [
            "market_breadth",
            "加權指數收盤_ret_5d",
            "電子指數收盤_ret_5d",
            "櫃買指數收盤_ret_5d",
            "加權指數收盤_ma_gap_20d",
            "market_value_ratio_20d",
        ]
        if c in market.columns
    ]
    if strength_cols:
        market["market_strength_score"] = market[strength_cols].replace([np.inf, -np.inf], np.nan).rank(pct=True).mean(axis=1)
    else:
        market["market_strength_score"] = 0.5

    merged = stock.merge(market, on="日期", how="left", suffixes=("", "_market"))
    merged = merged.merge(
        theme[["股票代號", "股票名稱", "主分類", "子分類"]],
        on="股票代號",
        how="left",
    )

    full = merged[merged["has_full_10d_window"]].copy()
    day_success = full.groupby("日期")["target_success"].mean()
    day_return = full.groupby("日期")["future_10d_high_close_return"].mean()
    merged["daily_market_success_rate"] = merged["日期"].map(day_success)
    merged["daily_market_avg_return"] = merged["日期"].map(day_return)
    merged["relative_top20_label"] = 0
    full_rank = full.groupby("日期")["future_10d_high_close_return"].rank(pct=True, method="average") >= 0.80
    merged.loc[full.index, "relative_top20_label"] = full_rank.astype(int)
    return merged


def dev_band_edges(dev_scores: pd.Series, bins: int = 4) -> np.ndarray:
    quantiles = np.linspace(0, 1, bins + 1)
    edges = dev_scores.quantile(quantiles).to_numpy(dtype=float).copy()
    edges[0] = -np.inf
    edges[-1] = np.inf
    return np.unique(edges)


def assign_bands(scores: pd.Series, edges: np.ndarray) -> pd.Series:
    labels = [f"band_{i+1}" for i in range(len(edges) - 1)]
    return pd.cut(scores, bins=edges, labels=labels, include_lowest=True)


def band_table(df: pd.DataFrame, split: str, score_col: str, edges: np.ndarray) -> pd.DataFrame:
    work = df.copy()
    work["calibration_band"] = assign_bands(work[score_col], edges)
    rows = []
    for band, part in work.groupby("calibration_band", observed=False):
        if part.empty:
            continue
        rows.append(
            {
                "split": split,
                "band": str(band),
                "rows": len(part),
                "avg_score": part[score_col].mean(),
                "actual_success_rate": part["target_success"].mean(),
                "avg_10d_high_close_return": part["future_10d_high_close_return"].mean(),
                "relative_top20_rate": part["relative_top20_label"].mean(),
            }
        )
    return pd.DataFrame(rows)


def top3_backtest(df: pd.DataFrame, split: str, score_col: str, gate: float) -> pd.DataFrame:
    work = df[df["market_strength_score"] >= gate].copy()
    if work.empty:
        return pd.DataFrame()
    picked = work.sort_values(["日期", score_col], ascending=[True, False]).groupby("日期").head(3)
    base = work.groupby("日期").agg(
        same_day_baseline_success_rate=("target_success", "mean"),
        same_day_baseline_avg_return=("future_10d_high_close_return", "mean"),
    )
    chosen = picked.groupby("日期").agg(
        top3_success_rate=("target_success", "mean"),
        top3_avg_return=("future_10d_high_close_return", "mean"),
        top3_count=("股票代號", "count"),
    )
    out = chosen.join(base, how="inner").reset_index()
    out["success_lift"] = out["top3_success_rate"] - out["same_day_baseline_success_rate"]
    out["return_lift"] = out["top3_avg_return"] - out["same_day_baseline_avg_return"]
    return out


def top3_summary(backtest: pd.DataFrame, picked: pd.DataFrame) -> dict:
    if backtest.empty or picked.empty:
        return {
            "days": 0,
            "rows": 0,
            "top3_success_rate": float("nan"),
            "same_day_baseline_success_rate": float("nan"),
            "success_lift": float("nan"),
            "top3_avg_return": float("nan"),
            "same_day_baseline_avg_return": float("nan"),
            "return_lift": float("nan"),
            "top_stock_share": float("nan"),
            "top_industry_share": float("nan"),
        }
    return {
        "days": backtest["日期"].nunique(),
        "rows": len(picked),
        "top3_success_rate": picked["target_success"].mean(),
        "same_day_baseline_success_rate": backtest["same_day_baseline_success_rate"].mean(),
        "success_lift": backtest["success_lift"].mean(),
        "top3_avg_return": picked["future_10d_high_close_return"].mean(),
        "same_day_baseline_avg_return": backtest["same_day_baseline_avg_return"].mean(),
        "return_lift": backtest["return_lift"].mean(),
        "top_stock_share": picked["股票代號"].value_counts(normalize=True).iloc[0],
        "top_industry_share": picked["主分類"].fillna("unknown").value_counts(normalize=True).iloc[0],
    }


def make_tracking(scored: pd.DataFrame, score_col: str, gate: float, config: dict, passed: bool, edges: np.ndarray, band_rates: dict) -> pd.DataFrame:
    latest_date = scored["日期"].max()
    latest = scored[(scored["日期"] == latest_date) & (scored["market_strength_score"] >= gate)].copy()
    if latest.empty:
        return pd.DataFrame(
            columns=[
                "signal_date",
                "stock_id",
                "stock_name",
                "research_score",
                "calibrated_success_rate",
                "market_strength_score",
                "status",
                "formal_candidate",
            ]
        )
    latest["calibration_band"] = assign_bands(latest[score_col], edges)
    latest["calibrated_success_rate"] = latest["calibration_band"].astype(str).map(band_rates)
    picked = latest.sort_values(score_col, ascending=False).head(3).copy()
    picked["signal_date"] = picked["日期"].dt.strftime("%Y-%m-%d")
    picked["stock_id"] = picked["股票代號"]
    picked["stock_name"] = picked["股票名稱"]
    picked["research_score"] = picked[score_col]
    picked["status"] = "追蹤中"
    picked["formal_candidate"] = bool(passed)
    return picked[
        [
            "signal_date",
            "stock_id",
            "stock_name",
            "research_score",
            "calibrated_success_rate",
            "market_strength_score",
            "status",
            "formal_candidate",
        ]
    ]


def write_formal(config: dict, passed: bool, reason: str, tracking: pd.DataFrame) -> None:
    if not passed:
        FORMAL_STATUS_PATH.write_text(
            "\n".join(
                [
                    "# Formal Status",
                    "",
                    "- Status: not active",
                    f"- Result: {config['formal_candidate_default']}",
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
        return

    result = "deep learning Top 3 formal strategy enabled"
    if tracking.empty:
        result = "deep learning Top 3 formal strategy enabled; current latest date has no candidate"
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
        for _, row in tracking.iterrows():
            writer.writerow(
                [
                    row["signal_date"],
                    row["stock_id"],
                    row["stock_name"],
                    row["calibrated_success_rate"],
                    "",
                    row["calibrated_success_rate"],
                    "",
                    "deep learning calibrated Top 3",
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

    train_data = build_dataset(stock, market, theme)
    scoring_data = build_scoring_dataset(stock, market, theme)

    train_features, _ = sequence_features(train_data, lookback=20)
    train_data = train_data[train_features["has_full_20d_history"] == 1].copy()
    train_features = train_features.loc[train_data.index].drop(columns=["has_full_20d_history"])
    scoring_features, _ = sequence_features(scoring_data, lookback=20)
    scoring_data = scoring_data[scoring_features["has_full_20d_history"] == 1].copy()
    scoring_features = scoring_features.loc[scoring_data.index].drop(columns=["has_full_20d_history"])
    scoring_features = scoring_features.reindex(columns=train_features.columns, fill_value=0.0)

    masks = split_masks(train_data, config)
    train_mask = masks["train"].to_numpy()
    X_train_all, mean, std = standardize(train_features.to_numpy(dtype=np.float32), train_mask)
    y_abs = train_data["target_success"].to_numpy(dtype=np.float32).reshape(-1, 1)

    abs_model, abs_loss = train_mlp(X_train_all[train_mask], y_abs[train_mask], seed=11)
    train_data["mlp_absolute_success"] = (1 / (1 + np.exp(-np.clip((np.maximum(X_train_all @ abs_model["w1"] + abs_model["b1"], 0.0) @ abs_model["w2"] + abs_model["b2"])[:, 0], -30, 30))))
    scoring_x = ((scoring_features.to_numpy(dtype=np.float32) - mean) / std).astype(np.float32)
    scoring_data["mlp_absolute_success"] = (1 / (1 + np.exp(-np.clip((np.maximum(scoring_x @ abs_model["w1"] + abs_model["b1"], 0.0) @ abs_model["w2"] + abs_model["b2"])[:, 0], -30, 30))))

    dev = train_data.loc[masks["development"]].copy()
    holdout = train_data.loc[masks["holdout"]].copy()
    score_col = "mlp_absolute_success"
    gate = 0.55
    edges = dev_band_edges(dev[score_col], bins=4)
    dev_bands = band_table(dev, "development", score_col, edges)
    holdout_bands = band_table(holdout, "holdout", score_col, edges)
    bands = pd.concat([dev_bands, holdout_bands], ignore_index=True)
    bands.to_csv(BANDS_PATH, index=False, encoding="utf-8-sig")

    dev_bt = top3_backtest(dev, "development", score_col, gate)
    holdout_bt = top3_backtest(holdout, "holdout", score_col, gate)
    top3_backtest_all = pd.concat(
        [
            dev_bt.assign(split="development"),
            holdout_bt.assign(split="holdout"),
        ],
        ignore_index=True,
    )
    top3_backtest_all.to_csv(TOP3_BACKTEST_PATH, index=False, encoding="utf-8-sig")

    holdout_picked = (
        holdout[holdout["market_strength_score"] >= gate]
        .sort_values(["日期", score_col], ascending=[True, False])
        .groupby("日期")
        .head(3)
        .copy()
    )
    holdout_summary = top3_summary(holdout_bt, holdout_picked)

    holdout_band_success = holdout_bands["actual_success_rate"].to_list()
    highest_band = holdout_bands.iloc[-1] if not holdout_bands.empty else None
    lowest_band = holdout_bands.iloc[0] if not holdout_bands.empty else None
    band_not_reversed = bool(
        highest_band is not None
        and lowest_band is not None
        and highest_band["actual_success_rate"] >= lowest_band["actual_success_rate"]
        and highest_band["avg_10d_high_close_return"] >= lowest_band["avg_10d_high_close_return"]
    )
    highest_band_rows_ok = bool(highest_band is not None and highest_band["rows"] >= 50)

    passed = bool(
        holdout_summary["success_lift"] >= 0.05
        and holdout_summary["return_lift"] > 0
        and band_not_reversed
        and highest_band_rows_ok
        and holdout_summary["top_stock_share"] <= 0.20
        and holdout_summary["top_industry_share"] <= 0.50
    )

    band_rates = {
        row["band"]: row["actual_success_rate"]
        for _, row in holdout_bands.iterrows()
    }
    tracking = make_tracking(scoring_data, score_col, gate, config, passed, edges, band_rates)
    tracking.to_csv(TRACKING_PATH, index=False, encoding="utf-8-sig")
    tracking.to_csv(LATEST_RESEARCH_PATH, index=False, encoding="utf-8-sig")

    if passed:
        status = "formal_pass"
        reason = "deep learning Top 3 passed holdout lift, return, calibration, sample, and concentration gates"
    elif holdout_summary["success_lift"] >= 0.05 and holdout_summary["return_lift"] > 0:
        status = "research_signal_only"
        reason = "Top 3 lift passed, but calibration or concentration gates did not all pass"
    else:
        status = "rejected"
        reason = "Top 3 did not pass holdout lift and return gates"

    write_formal(config, passed, reason, tracking)

    lines = [
        "# Deep Learning Calibrated Candidate Decision",
        "",
        f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "- Data sources: three allowed CSV inputs only",
        "- Strategy tested: mlp_absolute_success, market_strength_gate=0.55, Top 3",
        f"- Status: {status}",
        f"- Formal candidates enabled: {passed}",
        f"- Latest candidate count: {len(tracking)}",
        "",
        "## Training Loss",
        "",
        f"- Absolute model: {abs_loss[0]:.6f} -> {abs_loss[-1]:.6f}",
        "",
        "## Holdout Top 3",
        "",
        f"- Days: {holdout_summary['days']}",
        f"- Top 3 success rate: {fmt_pct(holdout_summary['top3_success_rate'])}",
        f"- Same-day baseline success rate: {fmt_pct(holdout_summary['same_day_baseline_success_rate'])}",
        f"- Success lift: {fmt_pct(holdout_summary['success_lift'])}",
        f"- Top 3 avg 10d high close return: {fmt_pct(holdout_summary['top3_avg_return'])}",
        f"- Same-day baseline avg return: {fmt_pct(holdout_summary['same_day_baseline_avg_return'])}",
        f"- Return lift: {fmt_pct(holdout_summary['return_lift'])}",
        f"- Top stock share: {fmt_pct(holdout_summary['top_stock_share'])}",
        f"- Top industry share: {fmt_pct(holdout_summary['top_industry_share'])}",
        "",
        "## Calibration Gate",
        "",
        f"- Highest band rows >= 50: {highest_band_rows_ok}",
        f"- Holdout high band not worse than low band: {band_not_reversed}",
        f"- Holdout band success rates: {', '.join(fmt_pct(x) for x in holdout_band_success)}",
        "",
        "## Decision",
        "",
        f"- {reason}.",
        "",
        "## Output Files",
        "",
        f"- Bands: `{BANDS_PATH}`",
        f"- Top 3 backtest: `{TOP3_BACKTEST_PATH}`",
        f"- Candidate tracking: `{TRACKING_PATH}`",
        f"- Formal status: `{FORMAL_STATUS_PATH}`",
        f"- Formal candidates: `{FORMAL_CANDIDATES_PATH}`",
    ]
    DECISION_PATH.write_text("\n".join(lines), encoding="utf-8")
    print("OK: deep learning calibrated candidate experiment completed")
    print(f"STATUS: {status}")
    print(f"DECISION: {DECISION_PATH}")


if __name__ == "__main__":
    main()

