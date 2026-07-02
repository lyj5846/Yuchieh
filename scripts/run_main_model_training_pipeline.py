from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
except ModuleNotFoundError:
    bundled_python = (
        Path.home()
        / ".cache"
        / "codex-runtimes"
        / "codex-primary-runtime"
        / "dependencies"
        / "python"
        / "python.exe"
    )
    if bundled_python.exists() and Path(sys.executable).resolve() != bundled_python.resolve():
        result = subprocess.run([str(bundled_python), str(Path(__file__).resolve()), *sys.argv[1:]])
        raise SystemExit(result.returncode)
    raise


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "project_config.json"
PLAN_PATH = PROJECT_ROOT / "planning_layer" / "current_model_plan.json"

MODEL_DIR = PROJECT_ROOT / "model_layer"
VALIDATION_DIR = PROJECT_ROOT / "validation_layer"
DECISION_DIR = PROJECT_ROOT / "decision_layer"

TRAINING_SPEC_PATH = MODEL_DIR / "main_model_training_spec.md"
SCORES_PATH = MODEL_DIR / "main_model_scores.csv"
VALIDATION_SUMMARY_PATH = VALIDATION_DIR / "main_model_validation_summary.csv"
CALIBRATION_PATH = VALIDATION_DIR / "main_model_calibration.csv"
DECISION_MD_PATH = DECISION_DIR / "main_model_decision.md"
DECISION_JSON_PATH = DECISION_DIR / "main_model_decision.json"

CONFIRMED_PLAN_ID = "risk_adjusted_main_model_training_plan"
LOOKBACK_DAYS = 20
EPISODE_GAP_DAYS = 10
LOOKAHEAD_DAYS = 10
PROFIT_THRESHOLD = 0.03
ADVERSE_THRESHOLD = -0.03
RETURN_RANK_WINDOWS = [1, 3, 5, 10, 20]
MA_RANK_WINDOWS = [5, 10, 20]
SAME_DAY_ADVANTAGE_LOSS_WEIGHT = 3.0

def fail(message: str) -> None:
    raise SystemExit(f"FAIL: {message}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the single integrated main model.")
    parser.add_argument("--confirmed-plan", required=True, help=f"Must be {CONFIRMED_PLAN_ID}.")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_confirmation(args: argparse.Namespace, plan: dict) -> None:
    if args.confirmed_plan != CONFIRMED_PLAN_ID:
        fail(f"--confirmed-plan must be {CONFIRMED_PLAN_ID}")
    if plan.get("recommended_experiment_id") != CONFIRMED_PLAN_ID:
        fail("planning layer does not recommend the requested main model training plan")
    if plan.get("confirmation_required") is not True:
        fail("planning layer must require confirmation before training")


def validate_inputs(config: dict) -> dict[str, Path]:
    allowed = config.get("allowed_inputs", {})
    if set(allowed) != {"stock_daily_all", "market_daily", "theme_group"}:
        fail("allowed_inputs must contain exactly the three approved sources")
    old_marker = "stock" + "_raw_only" + "_project"
    paths = {name: Path(value) for name, value in allowed.items()}
    for name, path in paths.items():
        if not path.exists():
            fail(f"missing input {name}: {path}")
        if old_marker in str(path):
            fail(f"old project path is not allowed: {name}")
        if name == "theme_group" and PROJECT_ROOT not in path.parents:
            fail("theme_group must live inside this project")
    return paths


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def normalize_stock(stock: pd.DataFrame) -> pd.DataFrame:
    out = stock.copy()
    out["日期"] = pd.to_datetime(out["日期"])
    out["股票代號"] = out["股票代號"].astype(str).str.strip()
    for col in out.columns:
        if col in {"日期", "股票代號", "資料狀態", "價格來源", "法人來源", "融資券來源", "當沖來源"}:
            continue
        out[col] = pd.to_numeric(out[col].astype(str).str.replace(",", "", regex=False), errors="coerce")
    return out.sort_values(["股票代號", "日期"]).reset_index(drop=True)


def normalize_market(market: pd.DataFrame) -> pd.DataFrame:
    out = market.copy()
    out["日期"] = pd.to_datetime(out["日期"])
    for col in out.columns:
        if col in {"日期", "資料狀態"}:
            continue
        out[col] = pd.to_numeric(out[col].astype(str).str.replace(",", "", regex=False), errors="coerce")
    out = out.sort_values("日期").reset_index(drop=True)
    for col in ["加權指數收盤", "電子指數收盤", "櫃買指數收盤"]:
        for window in RETURN_RANK_WINDOWS:
            out[f"{col}_ret_{window}"] = out[col].pct_change(window)
    breadth_denominator = (out["上漲家數"] + out["下跌家數"]).replace(0, np.nan)
    out["market_breadth"] = (out["上漲家數"] - out["下跌家數"]) / breadth_denominator
    out["market_turnover_ret_5"] = out["大盤成交值(億元)"].pct_change(5)
    out["market_foreign_5"] = out["外資買賣超(億元)"].rolling(5, min_periods=1).sum()
    out["market_margin_5"] = out["融資增減(億元)"].rolling(5, min_periods=1).sum()
    return out


def normalize_theme(theme: pd.DataFrame) -> pd.DataFrame:
    out = theme.copy()
    out["股票代號"] = out["股票代號"].astype(str).str.strip()
    keep = [c for c in ["股票代號", "股票名稱", "主分類", "子分類"] if c in out.columns]
    return out[keep].drop_duplicates("股票代號", keep="last")


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


def add_labels(stock: pd.DataFrame) -> pd.DataFrame:
    out = stock.sort_values(["股票代號", "日期"]).copy()
    group = out.groupby("股票代號", sort=False)
    out["stock_trading_index"] = group.cumcount()
    out["buy_open_next"] = group["開盤價"].shift(-1)
    future_closes = pd.concat([group["收盤價"].shift(-i) for i in range(1, LOOKAHEAD_DAYS + 1)], axis=1)
    future_lows = pd.concat([group["最低價"].shift(-i) for i in range(1, LOOKAHEAD_DAYS + 1)], axis=1)
    future_closes.columns = [f"future_close_{i}" for i in range(1, LOOKAHEAD_DAYS + 1)]
    future_lows.columns = [f"future_low_{i}" for i in range(1, LOOKAHEAD_DAYS + 1)]
    out["future_window_count"] = future_closes.notna().sum(axis=1)
    out["future_low_window_count"] = future_lows.notna().sum(axis=1)
    out["label_complete"] = (
        (out["future_window_count"] == LOOKAHEAD_DAYS)
        & (out["future_low_window_count"] == LOOKAHEAD_DAYS)
        & out["buy_open_next"].notna()
        & (out["buy_open_next"] > 0)
    )
    close_returns = future_closes.div(out["buy_open_next"], axis=0) - 1.0
    low_returns = future_lows.div(out["buy_open_next"], axis=0) - 1.0
    close_returns = close_returns.replace([np.inf, -np.inf], np.nan)
    low_returns = low_returns.replace([np.inf, -np.inf], np.nan)
    out["future_10d_high_close"] = future_closes.max(axis=1)
    out["future_10d_low_price"] = future_lows.min(axis=1)
    out["future_10d_high_close_return"] = close_returns.max(axis=1)
    out["future_10d_low_close_return"] = low_returns.min(axis=1)
    out["max_adverse_return"] = out["future_10d_low_close_return"]
    out["future_10d_day10_close_return"] = close_returns.iloc[:, -1]
    out["label_complete"] = (
        out["label_complete"]
        & out["future_10d_high_close_return"].notna()
        & out["future_10d_low_close_return"].notna()
    )
    out["old_target_success"] = (
        out["label_complete"] & (out["future_10d_high_close_return"] >= PROFIT_THRESHOLD)
    ).astype(int)
    out["profit_event_day"] = first_hit_day(close_returns, PROFIT_THRESHOLD, "ge")
    out["adverse_event_day"] = first_hit_day(low_returns, ADVERSE_THRESHOLD, "le")
    has_profit = out["profit_event_day"].notna()
    has_adverse = out["adverse_event_day"].notna()
    profit_first = has_profit & (~has_adverse | (out["profit_event_day"] < out["adverse_event_day"]))
    adverse_first = has_adverse & (~has_profit | (out["adverse_event_day"] <= out["profit_event_day"]))
    out["first_event_day"] = np.nan
    out.loc[profit_first, "first_event_day"] = out.loc[profit_first, "profit_event_day"]
    out.loc[adverse_first, "first_event_day"] = out.loc[adverse_first, "adverse_event_day"]
    out["first_event_type"] = "none"
    out.loc[profit_first, "first_event_type"] = "profit_first"
    out.loc[adverse_first, "first_event_type"] = "adverse_first"
    out["same_day_both_event"] = (
        out["label_complete"]
        & has_profit
        & has_adverse
        & out["profit_event_day"].eq(out["adverse_event_day"])
    ).astype(int)
    out["risk_adjusted_10d_success"] = (out["label_complete"] & profit_first).astype(int)
    out["target_success"] = out["risk_adjusted_10d_success"].astype(int)
    out["old_success_but_risk_failed"] = (
        (out["old_target_success"] == 1) & (out["target_success"] == 0)
    ).astype(int)
    out["realized_10d_trade_return"] = out["future_10d_day10_close_return"]
    out.loc[profit_first, "realized_10d_trade_return"] = PROFIT_THRESHOLD
    out.loc[adverse_first, "realized_10d_trade_return"] = ADVERSE_THRESHOLD
    out.loc[~out["label_complete"], "realized_10d_trade_return"] = np.nan
    return out


def add_features(stock: pd.DataFrame, market: pd.DataFrame, theme: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = stock.merge(market, on="日期", how="left", suffixes=("", "_market"))
    out = out.merge(theme, on="股票代號", how="left")
    out["股票名稱"] = out.get("股票名稱", pd.Series(index=out.index, dtype=object)).fillna("")
    out["主分類"] = out.get("主分類", pd.Series(index=out.index, dtype=object)).fillna("unknown")
    out["子分類"] = out.get("子分類", pd.Series(index=out.index, dtype=object)).fillna("unknown")
    out = out.sort_values(["股票代號", "日期"]).reset_index(drop=True)
    group = out.groupby("股票代號", sort=False)

    out["has_full_20d_history"] = (out["stock_trading_index"] >= LOOKBACK_DAYS - 1).astype(int)
    for window in [1, 3, 5, 10, 20]:
        out[f"close_ret_{window}"] = group["收盤價"].pct_change(window)
    for window in [5, 10, 20]:
        ma = group["收盤價"].transform(lambda s: s.rolling(window, min_periods=window).mean())
        volume_ma = group["成交量(張)"].transform(lambda s: s.rolling(window, min_periods=window).mean())
        out[f"close_vs_ma_{window}"] = out["收盤價"] / ma - 1.0
        out[f"volume_vs_ma_{window}"] = out["成交量(張)"] / volume_ma - 1.0
    out["range_pct"] = (out["最高價"] - out["最低價"]) / out["收盤價"].replace(0, np.nan)
    out["volatility_10"] = group["收盤價"].transform(lambda s: s.pct_change().rolling(10, min_periods=10).std())
    out["volatility_20"] = group["收盤價"].transform(lambda s: s.pct_change().rolling(20, min_periods=20).std())

    for base in ["外資買賣超(張)", "投信買賣超(張)", "自營商買賣超(張)", "三大法人合計買賣超(張)", "融資增減", "融券增減"]:
        if base in out.columns:
            out[f"{base}_sum_5"] = group[base].transform(lambda s: s.rolling(5, min_periods=1).sum())
            out[f"{base}_sum_10"] = group[base].transform(lambda s: s.rolling(10, min_periods=1).sum())

    return_ranking_features: list[str] = []
    relative_feature_data: dict[str, pd.Series] = {}
    for window in RETURN_RANK_WINDOWS:
        return_col = f"close_ret_{window}"
        same_day_rank = f"same_day_return_rank_{window}"
        industry_rank = f"industry_return_rank_{window}"
        relative_feature_data[same_day_rank] = out.groupby("日期")[return_col].rank(pct=True)
        relative_feature_data[industry_rank] = out.groupby(["日期", "主分類"])[return_col].rank(pct=True)
        return_ranking_features.extend([same_day_rank, industry_rank])
        market_pairs = [
            ("weighted", f"加權指數收盤_ret_{window}"),
            ("electronics", f"電子指數收盤_ret_{window}"),
            ("otc", f"櫃買指數收盤_ret_{window}"),
        ]
        for market_name, market_col in market_pairs:
            if market_col in out.columns:
                relative_col = f"return_vs_{market_name}_{window}"
                relative_feature_data[relative_col] = out[return_col] - out[market_col]
                return_ranking_features.append(relative_col)
    for window in MA_RANK_WINDOWS:
        volume_col = f"volume_vs_ma_{window}"
        ma_col = f"close_vs_ma_{window}"
        volume_rank = f"industry_volume_rank_{window}"
        ma_rank = f"industry_ma_position_rank_{window}"
        relative_feature_data[volume_rank] = out.groupby(["日期", "主分類"])[volume_col].rank(pct=True)
        relative_feature_data[ma_rank] = out.groupby(["日期", "主分類"])[ma_col].rank(pct=True)
        return_ranking_features.extend([volume_rank, ma_rank])
    if relative_feature_data:
        out = pd.concat([out, pd.DataFrame(relative_feature_data, index=out.index)], axis=1)

    numeric_features = [
        "開盤價",
        "最高價",
        "最低價",
        "收盤價",
        "成交量(張)",
        "當沖比率",
        "融資餘額",
        "融券餘額",
        "close_ret_1",
        "close_ret_3",
        "close_ret_5",
        "close_ret_10",
        "close_ret_20",
        "close_vs_ma_5",
        "close_vs_ma_10",
        "close_vs_ma_20",
        "volume_vs_ma_5",
        "volume_vs_ma_10",
        "volume_vs_ma_20",
        "range_pct",
        "volatility_10",
        "volatility_20",
        "加權指數收盤_ret_1",
        "加權指數收盤_ret_3",
        "加權指數收盤_ret_5",
        "加權指數收盤_ret_10",
        "加權指數收盤_ret_20",
        "電子指數收盤_ret_1",
        "電子指數收盤_ret_3",
        "電子指數收盤_ret_5",
        "電子指數收盤_ret_10",
        "電子指數收盤_ret_20",
        "櫃買指數收盤_ret_1",
        "櫃買指數收盤_ret_3",
        "櫃買指數收盤_ret_5",
        "櫃買指數收盤_ret_10",
        "櫃買指數收盤_ret_20",
        "market_breadth",
        "market_turnover_ret_5",
        "market_foreign_5",
        "market_margin_5",
    ]
    rolling_features = [c for c in out.columns if c.endswith("_sum_5") or c.endswith("_sum_10")]
    numeric_features.extend(rolling_features)
    numeric_features.extend(return_ranking_features)
    numeric_features = [c for c in numeric_features if c in out.columns]

    theme_dummies = pd.get_dummies(out["主分類"], prefix="theme", dummy_na=False)
    out = pd.concat([out, theme_dummies], axis=1)
    feature_cols = numeric_features + list(theme_dummies.columns)
    out[feature_cols] = out[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out, feature_cols


def add_relative_and_risk_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    completed = out["label_complete"]
    out["daily_market_avg_return"] = out.groupby("日期")["future_10d_high_close_return"].transform("mean")
    out["daily_market_success_rate"] = out.groupby("日期")["target_success"].transform("mean")
    return_percentiles = out[completed].groupby("日期")["future_10d_high_close_return"].rank(pct=True)
    out["same_day_return_percentile"] = 0.0
    out.loc[return_percentiles.index, "same_day_return_percentile"] = return_percentiles.astype(float)
    out["same_day_advantage_label"] = (
        completed
        & (out["same_day_return_percentile"] >= 0.70)
    ).astype(int)
    out["selection_success_label"] = (
        completed
        & (out["target_success"] == 1)
        & (out["same_day_return_percentile"] >= 0.50)
    ).astype(int)
    out["same_day_advantage_target"] = out["same_day_return_percentile"].astype(float)
    ranks = out[completed].groupby("日期")["future_10d_high_close_return"].rank(pct=True)
    out["relative_top20_label"] = 0
    out.loc[ranks.index, "relative_top20_label"] = (ranks >= 0.80).astype(int)
    out["underperform_market_label"] = (
        out["future_10d_high_close_return"] < out["daily_market_avg_return"]
    ).fillna(False).astype(int)
    out["risk_label"] = (
        completed
        & (
            (out["target_success"] == 0)
            | (out["future_10d_low_close_return"] <= -0.03)
            | (out["underperform_market_label"] == 1)
        )
    ).astype(int)

    out = out.sort_values(["股票代號", "日期"]).copy()
    group = out.groupby("股票代號", sort=False)
    prev_success = group["target_success"].transform(
        lambda s: s.shift(1).rolling(EPISODE_GAP_DAYS, min_periods=1).max()
    )
    out["episode_start_label"] = (
        completed & (out["target_success"] == 1) & (prev_success.fillna(0) == 0)
    ).astype(int)
    return out


def split_name(dates: pd.Series, complete: pd.Series, config: dict) -> pd.Series:
    train_end = pd.Timestamp(config["time_split"]["train_end"])
    dev_start = pd.Timestamp(config["time_split"]["dev_start"])
    dev_end = pd.Timestamp(config["time_split"]["dev_end"])
    holdout_start = pd.Timestamp(config["time_split"]["holdout_start"])
    split = pd.Series("tracking", index=dates.index)
    split[(complete) & (dates <= train_end)] = "train"
    split[(complete) & (dates >= dev_start) & (dates <= dev_end)] = "development"
    split[(complete) & (dates >= holdout_start)] = "holdout"
    return split


def standardize(x: np.ndarray, train_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x[train_mask].mean(axis=0)
    std = x[train_mask].std(axis=0)
    std[std == 0] = 1.0
    return ((x - mean) / std).astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def train_mlp(
    x: np.ndarray,
    y: np.ndarray,
    hidden: int = 48,
    epochs: int = 90,
    lr: float = 0.015,
    regression_indices: set[int] | None = None,
    output_weights: np.ndarray | None = None,
) -> tuple[dict, list[float]]:
    regression_indices = regression_indices or set()
    if output_weights is None:
        output_weights = np.ones(y.shape[1], dtype=np.float32)
    output_weights = output_weights.astype(np.float32)
    if output_weights.shape[0] != y.shape[1]:
        fail("output_weights length must match output heads")
    rng = np.random.default_rng(42)
    w1 = rng.normal(0, 0.08, size=(x.shape[1], hidden)).astype(np.float32)
    b1 = np.zeros(hidden, dtype=np.float32)
    w2 = rng.normal(0, 0.08, size=(hidden, y.shape[1])).astype(np.float32)
    b2 = np.zeros(y.shape[1], dtype=np.float32)
    losses: list[float] = []
    n = max(len(x), 1)
    regression_mask = np.array([i in regression_indices for i in range(y.shape[1])])
    binary_mask = ~regression_mask
    for _ in range(epochs):
        z1 = x @ w1 + b1
        h = np.maximum(z1, 0.0)
        logits = h @ w2 + b2
        pred = sigmoid(logits)
        loss_matrix = np.zeros_like(pred)
        if binary_mask.any():
            loss_matrix[:, binary_mask] = -(
                y[:, binary_mask] * np.log(pred[:, binary_mask] + 1e-7)
                + (1 - y[:, binary_mask]) * np.log(1 - pred[:, binary_mask] + 1e-7)
            )
        if regression_mask.any():
            loss_matrix[:, regression_mask] = (pred[:, regression_mask] - y[:, regression_mask]) ** 2
        loss = np.mean(loss_matrix * output_weights)
        losses.append(float(loss))
        grad_logits = np.zeros_like(pred)
        if binary_mask.any():
            grad_logits[:, binary_mask] = (
                output_weights[binary_mask]
                * (pred[:, binary_mask] - y[:, binary_mask])
                / n
            )
        if regression_mask.any():
            grad_logits[:, regression_mask] = (
                output_weights[regression_mask]
                *
                2.0
                * (pred[:, regression_mask] - y[:, regression_mask])
                * pred[:, regression_mask]
                * (1.0 - pred[:, regression_mask])
                / n
            )
        grad_w2 = h.T @ grad_logits
        grad_b2 = grad_logits.sum(axis=0)
        grad_h = grad_logits @ w2.T
        grad_z1 = grad_h * (z1 > 0)
        grad_w1 = x.T @ grad_z1
        grad_b1 = grad_z1.sum(axis=0)
        w1 -= lr * grad_w1.astype(np.float32)
        b1 -= lr * grad_b1.astype(np.float32)
        w2 -= lr * grad_w2.astype(np.float32)
        b2 -= lr * grad_b2.astype(np.float32)
    return {"w1": w1, "b1": b1, "w2": w2, "b2": b2}, losses


def predict_mlp(x: np.ndarray, model: dict) -> np.ndarray:
    h = np.maximum(x @ model["w1"] + model["b1"], 0.0)
    return sigmoid(h @ model["w2"] + model["b2"])


def apply_integrated_score(df: pd.DataFrame, weights: tuple[float, float, float, float]) -> pd.DataFrame:
    out = df.copy()
    success_w, advantage_w, episode_w, risk_w = weights
    out["integrated_research_score"] = (
        success_w * out["success_advantage_head"]
        + advantage_w * out["same_day_advantage_head"]
        + episode_w * out["episode_head"]
        - risk_w * out["risk_head"]
    )
    return out


def select_top3(df: pd.DataFrame, gate: float | None) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    work = df.sort_values(["日期", "integrated_research_score"], ascending=[True, False]).copy()
    rows = []
    last_pick_index: dict[str, int] = {}
    for _, day in work.groupby("日期", sort=True):
        if gate is not None and day["integrated_research_score"].max() < gate:
            continue
        selected_today = 0
        for _, row in day.iterrows():
            stock_id = str(row["股票代號"])
            stock_index = int(row["stock_trading_index"])
            if stock_id in last_pick_index and stock_index - last_pick_index[stock_id] <= EPISODE_GAP_DAYS:
                continue
            rows.append(row)
            last_pick_index[stock_id] = stock_index
            selected_today += 1
            if selected_today >= 3:
                break
    if not rows:
        return pd.DataFrame(columns=work.columns)
    return pd.DataFrame(rows)


def score_probe_top3(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    work = df.copy()
    work["integrated_research_score"] = work[score_col]
    return select_top3(work, gate=None)


def summary_row(full: pd.DataFrame, picked: pd.DataFrame, split: str, strategy: str, gate: float | None) -> dict:
    if picked.empty:
        return {
            "split": split,
            "strategy": strategy,
            "gate": gate if gate is not None else "",
            "rows": 0,
            "days": 0,
            "success_rate": math.nan,
            "same_day_baseline_success_rate": math.nan,
            "success_lift": math.nan,
            "avg_10d_high_close_return": math.nan,
            "same_day_baseline_avg_return": math.nan,
            "return_lift": math.nan,
            "top_stock_share": math.nan,
            "top_industry_share": math.nan,
        }
    selected_days = picked["日期"].drop_duplicates()
    baseline = full[full["日期"].isin(selected_days)].groupby("日期").agg(
        day_success=("target_success", "mean"),
        day_return=("future_10d_high_close_return", "mean"),
    )
    chosen = picked.groupby("日期").agg(
        pick_success=("target_success", "mean"),
        pick_return=("future_10d_high_close_return", "mean"),
    )
    joined = chosen.join(baseline, how="inner")
    return {
        "split": split,
        "strategy": strategy,
        "gate": gate if gate is not None else "",
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


def monthly_lift_summary(full: pd.DataFrame, picked: pd.DataFrame) -> dict:
    if picked.empty:
        return {
            "monthly_positive_months": 0,
            "monthly_total_months": 0,
            "min_monthly_success_lift": math.nan,
            "min_monthly_return_lift": math.nan,
            "mean_monthly_success_lift": math.nan,
            "mean_monthly_return_lift": math.nan,
            "monthly_stability_passed": False,
        }
    rows = []
    full_month = full.copy()
    picked_month = picked.copy()
    full_month["tuning_month"] = full_month["日期"].dt.strftime("%Y-%m")
    picked_month["tuning_month"] = picked_month["日期"].dt.strftime("%Y-%m")
    for month, month_picked in picked_month.groupby("tuning_month", sort=True):
        month_full = full_month[full_month["tuning_month"].eq(month)]
        row = summary_row(month_full, month_picked, "development", f"month_{month}", None)
        rows.append(row)
    monthly = pd.DataFrame(rows)
    valid = monthly.dropna(subset=["success_lift", "return_lift"])
    if valid.empty:
        return {
            "monthly_positive_months": 0,
            "monthly_total_months": 0,
            "min_monthly_success_lift": math.nan,
            "min_monthly_return_lift": math.nan,
            "mean_monthly_success_lift": math.nan,
            "mean_monthly_return_lift": math.nan,
            "monthly_stability_passed": False,
        }
    positive = valid[(valid["success_lift"] > 0) & (valid["return_lift"] > 0)]
    total_months = int(len(valid))
    required_positive = max(2, math.ceil(total_months * 0.60))
    return {
        "monthly_positive_months": int(len(positive)),
        "monthly_total_months": total_months,
        "min_monthly_success_lift": float(valid["success_lift"].min()),
        "min_monthly_return_lift": float(valid["return_lift"].min()),
        "mean_monthly_success_lift": float(valid["success_lift"].mean()),
        "mean_monthly_return_lift": float(valid["return_lift"].mean()),
        "monthly_stability_passed": bool(len(positive) >= required_positive),
    }


def balanced_objective(row: pd.Series) -> float:
    success_lift = float(row["success_lift"]) if not pd.isna(row["success_lift"]) else -1.0
    return_lift = float(row["return_lift"]) if not pd.isna(row["return_lift"]) else -1.0
    monthly_positive = float(row.get("monthly_positive_months", 0) or 0)
    concentration_penalty = max(0.0, float(row["top_industry_share"]) - 0.50) if not pd.isna(row["top_industry_share"]) else 0.20
    weak_side = min(success_lift, return_lift)
    return weak_side + 0.25 * success_lift + 0.25 * return_lift + 0.02 * monthly_positive - 0.10 * concentration_penalty


def tune_strategy(dev: pd.DataFrame) -> tuple[tuple[float, float, float, float], float | None, pd.DataFrame, dict]:
    rows = []
    # This is still a single research score. The grid only decides how to combine
    # the existing four heads; it does not create another model or probability.
    weight_grid = [
        (sw, rw, ew, fw)
        for sw in [1.0, 1.2, 1.5, 1.8, 2.2]
        for rw in [0.6, 0.8, 1.0, 1.2, 1.5]
        for ew in [0.0, 0.1, 0.2]
        for fw in [0.0, 0.1, 0.2, 0.4]
    ]
    for weights in weight_grid:
        scored = apply_integrated_score(dev, weights)
        day_max = scored.groupby("日期")["integrated_research_score"].max()
        gates: list[float | None] = [None]
        gates.extend(float(day_max.quantile(q)) for q in [0.50, 0.60, 0.70, 0.80])
        for gate in gates:
            picked = select_top3(scored, gate)
            row = summary_row(scored, picked, "development", "integrated_main_top3", gate)
            row["weights"] = ",".join(str(v) for v in weights)
            row.update(monthly_lift_summary(scored, picked))
            row["balanced_objective_score"] = balanced_objective(pd.Series(row))
            rows.append(row)
    table = pd.DataFrame(rows)
    feasible = table[
        (table["days"] >= 10)
        & (table["success_lift"] > 0)
        & (table["return_lift"] > 0)
        & (table["monthly_stability_passed"])
        & (table["top_stock_share"] <= 0.25)
        & (table["top_industry_share"] <= 0.60)
    ].copy()
    if feasible.empty:
        feasible = table[table["days"] >= 5].copy()
    if feasible.empty:
        feasible = table.copy()
    feasible = feasible.sort_values(
        [
            "balanced_objective_score",
            "monthly_positive_months",
            "success_lift",
            "return_lift",
            "success_rate",
            "days",
        ],
        ascending=[False, False, False, False, False, False],
    )
    best = feasible.iloc[0]
    weights = tuple(float(v) for v in str(best["weights"]).split(","))
    gate = None if best["gate"] == "" or pd.isna(best["gate"]) else float(best["gate"])
    return weights, gate, table, best.to_dict()


def band_table(df: pd.DataFrame, score_col: str, target_col: str, band_type: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    work = df.copy()
    work["band"] = pd.qcut(work[score_col].rank(method="first"), q=min(4, len(work)), labels=False, duplicates="drop")
    rows = []
    for band, part in work.groupby("band"):
        rows.append(
            {
                "band_type": band_type,
                "score_col": score_col,
                "target_col": target_col,
                "band": int(band),
                "rows": len(part),
                "avg_score": part[score_col].mean(),
                "actual_rate": part[target_col].mean(),
                "avg_10d_high_close_return": part["future_10d_high_close_return"].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values(["band_type", "band"])


def fmt_pct(value: float) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    return f"{value:.2%}"


def build_training_frame(config: dict) -> tuple[pd.DataFrame, list[str]]:
    paths = validate_inputs(config)
    stock = normalize_stock(read_csv(paths["stock_daily_all"]))
    market = normalize_market(read_csv(paths["market_daily"]))
    theme = normalize_theme(read_csv(paths["theme_group"]))
    stock_latest = stock["日期"].max()
    market_latest = market["日期"].max()
    if stock_latest != market_latest:
        fail(f"stock and market latest dates differ: {stock_latest.date()} vs {market_latest.date()}")
    labeled = add_labels(stock)
    featured, feature_cols = add_features(labeled, market, theme)
    return add_relative_and_risk_labels(featured), feature_cols


def main() -> None:
    args = parse_args()
    config = load_json(CONFIG_PATH)
    plan = load_json(PLAN_PATH)
    validate_confirmation(args, plan)

    frame, feature_cols = build_training_frame(config)
    frame["split"] = split_name(frame["日期"], frame["label_complete"], config)
    usable = frame[(frame["label_complete"]) & (frame["has_full_20d_history"] == 1)].copy()
    scoring = frame[frame["has_full_20d_history"] == 1].copy()
    if usable.empty:
        fail("no usable completed samples after feature generation")
    train_mask = usable["split"].eq("train").to_numpy()
    if train_mask.sum() == 0:
        fail("no training samples")

    x_all, mean, std = standardize(usable[feature_cols].to_numpy(dtype=np.float32), train_mask)
    y = usable[
        [
            "selection_success_label",
            "same_day_advantage_target",
            "risk_label",
            "episode_start_label",
        ]
    ].to_numpy(dtype=np.float32)
    output_weights = np.array([1.0, SAME_DAY_ADVANTAGE_LOSS_WEIGHT, 1.0, 1.0], dtype=np.float32)
    model, losses = train_mlp(
        x_all[train_mask],
        y[train_mask],
        regression_indices={1},
        output_weights=output_weights,
    )
    pred = predict_mlp(x_all, model)
    scored = usable.copy()
    scored["success_advantage_head"] = pred[:, 0]
    scored["same_day_advantage_head"] = pred[:, 1]
    scored["risk_head"] = pred[:, 2]
    scored["episode_head"] = pred[:, 3]

    dev = scored[scored["split"].eq("development")].copy()
    holdout = scored[scored["split"].eq("holdout")].copy()
    if dev.empty or holdout.empty:
        fail("development and holdout splits must both contain completed samples")
    weights, gate, tuning_table, selected_strategy = tune_strategy(dev)
    tuning_table.to_csv(VALIDATION_DIR / "main_model_strategy_tuning.csv", index=False, encoding="utf-8-sig")

    dev_scored = apply_integrated_score(dev, weights)
    holdout_scored = apply_integrated_score(holdout, weights)
    dev_picked = select_top3(dev_scored, gate)
    holdout_picked = select_top3(holdout_scored, gate)
    dev_probe_picked = score_probe_top3(dev, "same_day_advantage_head")
    holdout_probe_picked = score_probe_top3(holdout, "same_day_advantage_head")

    validation_rows = [
        summary_row(dev_scored, dev_picked, "development", "integrated_main_top3", gate),
        summary_row(holdout_scored, holdout_picked, "holdout", "integrated_main_top3", gate),
        summary_row(dev, dev_probe_picked, "development", "return_ranking_probe_top3", None),
        summary_row(holdout, holdout_probe_picked, "holdout", "return_ranking_probe_top3", None),
        {
            "split": "holdout",
            "strategy": "risk_adjusted_market_baseline",
            "gate": "",
            "rows": len(holdout),
            "days": holdout["日期"].nunique(),
            "success_rate": holdout.groupby("日期")["target_success"].mean().mean(),
            "same_day_baseline_success_rate": "",
            "success_lift": 0.0,
            "avg_10d_high_close_return": holdout.groupby("日期")["future_10d_high_close_return"].mean().mean(),
            "same_day_baseline_avg_return": "",
            "return_lift": 0.0,
            "top_stock_share": "",
            "top_industry_share": "",
        },
    ]
    validation = pd.DataFrame(validation_rows)
    validation.to_csv(VALIDATION_SUMMARY_PATH, index=False, encoding="utf-8-sig")

    calibration = pd.concat(
        [
            band_table(holdout_scored, "integrated_research_score", "target_success", "integrated_score_success"),
            band_table(holdout_scored, "integrated_research_score", "same_day_advantage_label", "integrated_score_advantage"),
            band_table(holdout_scored, "success_advantage_head", "target_success", "success_advantage_head_success"),
            band_table(holdout_scored, "success_advantage_head", "selection_success_label", "success_advantage_head_selection"),
            band_table(holdout_scored, "same_day_advantage_head", "same_day_advantage_label", "same_day_advantage_head_advantage"),
            band_table(holdout_scored, "same_day_advantage_head", "same_day_advantage_target", "same_day_advantage_head_target"),
            band_table(holdout_scored, "risk_head", "risk_label", "risk_head_failure"),
        ],
        ignore_index=True,
    )
    calibration.to_csv(CALIBRATION_PATH, index=False, encoding="utf-8-sig")

    cal_score = calibration[calibration["band_type"].eq("integrated_score_success")].sort_values("band")
    cal_advantage = calibration[calibration["band_type"].eq("same_day_advantage_head_advantage")].sort_values("band")
    cal_advantage_target = calibration[calibration["band_type"].eq("same_day_advantage_head_target")].sort_values("band")
    cal_risk = calibration[calibration["band_type"].eq("risk_head_failure")].sort_values("band")
    score_order_ok = bool(len(cal_score) >= 2 and cal_score.iloc[-1]["actual_rate"] >= cal_score.iloc[0]["actual_rate"])
    advantage_order_ok = bool(len(cal_advantage) >= 2 and cal_advantage.iloc[-1]["actual_rate"] >= cal_advantage.iloc[0]["actual_rate"])
    return_ranking_probe_order_ok = bool(
        len(cal_advantage_target) >= 2
        and cal_advantage_target.iloc[-1]["actual_rate"] >= cal_advantage_target.iloc[0]["actual_rate"]
    )
    risk_order_ok = bool(len(cal_risk) >= 2 and cal_risk.iloc[-1]["actual_rate"] >= cal_risk.iloc[0]["actual_rate"])
    holdout_row = validation[validation["strategy"].eq("integrated_main_top3") & validation["split"].eq("holdout")].iloc[0]
    holdout_probe_row = validation[validation["strategy"].eq("return_ranking_probe_top3") & validation["split"].eq("holdout")].iloc[0]
    active_months = holdout_picked["日期"].dt.strftime("%Y-%m").nunique() if not holdout_picked.empty else 0
    passed = bool(
        holdout_row["success_lift"] > 0
        and holdout_row["return_lift"] > 0
        and score_order_ok
        and advantage_order_ok
        and return_ranking_probe_order_ok
        and holdout_probe_row["return_lift"] > 0
        and risk_order_ok
        and holdout_row["top_stock_share"] <= 0.20
        and holdout_row["top_industry_share"] <= 0.50
        and active_months >= 2
    )
    status = "passed_holdout_validation" if passed else "not_promoted"
    reason = "main model passed validation and can be considered by the formal entrypoint"
    if not passed:
        reason = "main model did not pass every validation gate; formal output must remain unchanged"

    scoring_x = ((scoring[feature_cols].to_numpy(dtype=np.float32) - mean) / std).astype(np.float32)
    scoring_pred = predict_mlp(scoring_x, model)
    scoring = scoring.copy()
    scoring["success_advantage_head"] = scoring_pred[:, 0]
    scoring["same_day_advantage_head"] = scoring_pred[:, 1]
    scoring["risk_head"] = scoring_pred[:, 2]
    scoring["episode_head"] = scoring_pred[:, 3]
    scoring = apply_integrated_score(scoring, weights)
    scoring["split"] = split_name(scoring["日期"], scoring["label_complete"], config)
    scoring["daily_rank"] = scoring.groupby("日期")["integrated_research_score"].rank(ascending=False, method="first")
    return_ranking_output_cols = [
        c
        for c in feature_cols
        if c.startswith("same_day_return_rank_")
        or c.startswith("industry_return_rank_")
        or c.startswith("industry_volume_rank_")
        or c.startswith("industry_ma_position_rank_")
        or c.startswith("return_vs_weighted_")
        or c.startswith("return_vs_electronics_")
        or c.startswith("return_vs_otc_")
    ]
    output_cols = [
        "日期",
        "股票代號",
        "股票名稱",
        "主分類",
        "split",
        "daily_rank",
        "integrated_research_score",
        "success_advantage_head",
        "same_day_advantage_head",
        "risk_head",
        "episode_head",
        "target_success",
        "risk_adjusted_10d_success",
        "old_target_success",
        "old_success_but_risk_failed",
        "selection_success_label",
        "same_day_advantage_label",
        "same_day_advantage_target",
        "same_day_return_percentile",
        "risk_label",
        "relative_top20_label",
        "episode_start_label",
        "future_10d_high_close_return",
        "future_10d_low_close_return",
        "max_adverse_return",
        "profit_event_day",
        "adverse_event_day",
        "first_event_day",
        "first_event_type",
        "same_day_both_event",
        "realized_10d_trade_return",
        "daily_market_success_rate",
        "daily_market_avg_return",
    ]
    output_cols.extend(return_ranking_output_cols)
    score_output = scoring[output_cols].sort_values(["日期", "daily_rank"])
    score_output.to_csv(SCORES_PATH, index=False, encoding="utf-8-sig")

    TRAINING_SPEC_PATH.write_text(
        "\n".join(
            [
                "# Main Model Training Spec",
                "",
                f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"- Confirmed plan: `{CONFIRMED_PLAN_ID}`",
                "- Data sources: three approved CSV inputs only.",
                "- Model: one hidden-layer numpy MLP with four outputs.",
                "- Training heads: selection_success, same_day_advantage soft target, failure_risk, episode_start.",
                "- Formal target_success is risk-adjusted_10d_success.",
                "- Risk-adjusted success: next-day open buy; +3% close must occur before any -3% low within 10 trading days.",
                "- Conservative tie rule: if +3% close and -3% low occur on the same day, target_success is failure.",
                "- Old +3% touch target is retained as old_target_success for comparison only.",
                "- Same-day advantage soft target: pure same-day return percentile.",
                "- Uses same-day relative return-ranking features against all stocks, same industry, and market indices.",
                f"- same_day_advantage loss weight: {SAME_DAY_ADVANTAGE_LOSS_WEIGHT}.",
                "- Strategy tuning: selected on development with monthly stability and a balanced success/return objective.",
                "- Development monthly stability requires most active months to have both success lift and return lift above zero.",
                f"- Feature lookback: {LOOKBACK_DAYS} trading days.",
                f"- Episode gap: {EPISODE_GAP_DAYS} trading days.",
                f"- Selected weights: {', '.join(str(v) for v in weights)}",
                f"- Selected gate: {gate if gate is not None else 'none'}",
                f"- Selected development positive months: {int(selected_strategy.get('monthly_positive_months', 0))}/{int(selected_strategy.get('monthly_total_months', 0))}",
                f"- Selected balanced objective score: {float(selected_strategy.get('balanced_objective_score', math.nan)):.6f}",
                "- Raw outputs are research ranking scores, not calibrated success rates.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    decision = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "confirmed_plan": CONFIRMED_PLAN_ID,
        "target_contract": "risk_adjusted_10d_success",
        "status": status,
        "formal_approved": passed,
        "reason": reason,
        "selected_weights": list(weights),
        "selected_gate": gate,
        "training_loss_start": losses[0],
        "training_loss_end": losses[-1],
        "holdout_success_rate": float(holdout_row["success_rate"]) if not pd.isna(holdout_row["success_rate"]) else None,
        "holdout_success_lift": float(holdout_row["success_lift"]) if not pd.isna(holdout_row["success_lift"]) else None,
        "holdout_return_lift": float(holdout_row["return_lift"]) if not pd.isna(holdout_row["return_lift"]) else None,
        "holdout_old_target_success_rate": float(holdout["old_target_success"].mean()) if not holdout.empty else None,
        "holdout_risk_adjusted_success_rate": float(holdout["target_success"].mean()) if not holdout.empty else None,
        "holdout_old_success_but_risk_failed_rate": float(holdout["old_success_but_risk_failed"].mean()) if not holdout.empty else None,
        "holdout_old_success_but_risk_failed_count": int(holdout["old_success_but_risk_failed"].sum()) if not holdout.empty else 0,
        "holdout_old_success_but_risk_failed_among_old_success": (
            float(holdout["old_success_but_risk_failed"].sum() / holdout["old_target_success"].sum())
            if not holdout.empty and int(holdout["old_target_success"].sum()) > 0
            else None
        ),
        "holdout_return_ranking_probe_success_lift": float(holdout_probe_row["success_lift"]) if not pd.isna(holdout_probe_row["success_lift"]) else None,
        "holdout_return_ranking_probe_return_lift": float(holdout_probe_row["return_lift"]) if not pd.isna(holdout_probe_row["return_lift"]) else None,
        "score_order_ok": score_order_ok,
        "advantage_order_ok": advantage_order_ok,
        "return_ranking_probe_order_ok": return_ranking_probe_order_ok,
        "risk_order_ok": risk_order_ok,
        "active_months": int(active_months),
        "same_day_advantage_loss_weight": SAME_DAY_ADVANTAGE_LOSS_WEIGHT,
        "development_monthly_positive_months": int(selected_strategy.get("monthly_positive_months", 0)),
        "development_monthly_total_months": int(selected_strategy.get("monthly_total_months", 0)),
        "development_min_monthly_success_lift": float(selected_strategy.get("min_monthly_success_lift", math.nan)),
        "development_min_monthly_return_lift": float(selected_strategy.get("min_monthly_return_lift", math.nan)),
        "development_mean_monthly_success_lift": float(selected_strategy.get("mean_monthly_success_lift", math.nan)),
        "development_mean_monthly_return_lift": float(selected_strategy.get("mean_monthly_return_lift", math.nan)),
        "selected_weight_stability_passed": bool(selected_strategy.get("monthly_stability_passed", False)),
        "selected_weight_objective_score": float(selected_strategy.get("balanced_objective_score", math.nan)),
    }
    DECISION_JSON_PATH.write_text(json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    DECISION_MD_PATH.write_text(
        "\n".join(
            [
                "# Main Model Decision",
                "",
                f"- Generated: {decision['generated_at']}",
                f"- Status: {status}",
                f"- Formal approved: {passed}",
                f"- Reason: {reason}",
                f"- Training loss: {losses[0]:.6f} -> {losses[-1]:.6f}",
                "- Target contract: risk_adjusted_10d_success",
                f"- Holdout old +3% touch success rate: {fmt_pct(decision['holdout_old_target_success_rate'])}",
                f"- Holdout risk-adjusted success rate: {fmt_pct(decision['holdout_risk_adjusted_success_rate'])}",
                f"- Holdout old successes filtered by risk rule among all rows: {fmt_pct(decision['holdout_old_success_but_risk_failed_rate'])}",
                f"- Holdout old successes filtered by risk rule among old successes: {fmt_pct(decision['holdout_old_success_but_risk_failed_among_old_success'])}",
                f"- Holdout success rate: {fmt_pct(decision['holdout_success_rate'])}",
                f"- Holdout success lift: {fmt_pct(decision['holdout_success_lift'])}",
                f"- Holdout return lift: {fmt_pct(decision['holdout_return_lift'])}",
                f"- Holdout return-ranking probe success lift: {fmt_pct(decision['holdout_return_ranking_probe_success_lift'])}",
                f"- Holdout return-ranking probe return lift: {fmt_pct(decision['holdout_return_ranking_probe_return_lift'])}",
                f"- Score band ordering valid: {score_order_ok}",
                f"- Advantage head ordering valid: {advantage_order_ok}",
                f"- Return-ranking probe ordering valid: {return_ranking_probe_order_ok}",
                f"- Risk band ordering valid: {risk_order_ok}",
                f"- Active holdout months: {active_months}",
                f"- Development monthly positive months: {decision['development_monthly_positive_months']}/{decision['development_monthly_total_months']}",
                f"- Development min monthly success lift: {fmt_pct(decision['development_min_monthly_success_lift'])}",
                f"- Development min monthly return lift: {fmt_pct(decision['development_min_monthly_return_lift'])}",
                f"- Selected weight stability passed: {decision['selected_weight_stability_passed']}",
                f"- Selected balanced objective score: {decision['selected_weight_objective_score']:.6f}",
                "",
                "Formal output is not updated by this training pipeline.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print("OK: main model training pipeline completed")
    print(f"STATUS: {status}")
    print(f"PROMOTION_APPROVED: {passed}")
    print(f"DECISION: {DECISION_MD_PATH}")


if __name__ == "__main__":
    main()
