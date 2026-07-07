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
FEATURE_SCREEN_PATH = VALIDATION_DIR / "main_model_feature_screen.csv"
FEATURE_SIGNAL_PATH = VALIDATION_DIR / "data_learnability_feature_signal.csv"
VALIDATION_SUMMARY_PATH = VALIDATION_DIR / "main_model_validation_summary.csv"
CALIBRATION_PATH = VALIDATION_DIR / "main_model_calibration.csv"
DECISION_MD_PATH = DECISION_DIR / "main_model_decision.md"
DECISION_JSON_PATH = DECISION_DIR / "main_model_decision.json"

CONFIRMED_PLAN_ID = "drawdown_side_label_main_model_training_plan"
LOOKBACK_DAYS = 20
EPISODE_GAP_DAYS = 10
LOOKAHEAD_DAYS = 10
PROFIT_THRESHOLD = 0.03
ADVERSE_THRESHOLD = -0.03
RETURN_RANK_WINDOWS = [1, 3, 5, 10, 20]
MA_RANK_WINDOWS = [5, 10, 20]
SAME_DAY_ADVANTAGE_LOSS_WEIGHT = 3.0
FEATURE_SCREEN_MIN_ABS_CORR = 0.01
FEATURE_SCREEN_MAX_FEATURES = 48
FEATURE_SCREEN_MIN_FEATURES = 12
ATTENTION_DISPOSITION_FEATURES = [
    "attention_disposition_known_count_1d",
    "attention_disposition_known_count_3d",
    "attention_disposition_known_count_10d",
    "attention_active_on_signal_date",
    "disposition_active_on_signal_date",
    "days_since_last_attention_disposition",
    "has_attention_disposition_history_20d",
]

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


def validate_candidate_feature_inputs(config: dict) -> dict[str, Path]:
    candidates = config.get("candidate_model_feature_inputs", {})
    if not candidates:
        return {}
    attention_candidate = candidates.get("attention_disposition_events")
    if attention_candidate is None:
        return {}
    if not isinstance(attention_candidate, dict):
        fail("candidate_model_feature_inputs.attention_disposition_events must be an object")
    if attention_candidate.get("status") != "approved_for_next_training_candidate":
        fail("attention/disposition candidate input is not approved for next training")
    if attention_candidate.get("scope") != "attention_disposition_only":
        fail("attention/disposition candidate scope must remain attention_disposition_only")
    approval_path = attention_candidate.get("approval_decision")
    if approval_path != "decision_layer\\attention_disposition_model_input_approval_decision.json":
        fail("attention/disposition candidate approval decision path mismatch")
    path = Path(str(attention_candidate.get("path", "")))
    if path != PROJECT_ROOT / "inputs" / "event_risk_calendar_backfilled.csv":
        fail("attention/disposition candidate input must use the approved backfilled event file")
    if not path.exists():
        fail(f"missing attention/disposition candidate input: {path}")
    return {"attention_disposition_events": path}


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


def normalize_stock_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"\.0$", "", regex=True).str.replace(r"\D", "", regex=True)


def bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def normalize_attention_disposition_events(events: pd.DataFrame) -> pd.DataFrame:
    out = events.copy()
    out["stock_id"] = normalize_stock_id(out["stock_id"])
    out["signal_usable_date"] = pd.to_datetime(out["signal_usable_date"])
    out["event_effective_start_date"] = pd.to_datetime(out["event_effective_start_date"], errors="coerce")
    out["event_effective_end_date"] = pd.to_datetime(out["event_effective_end_date"], errors="coerce")
    out["known_before_signal_close_bool"] = bool_series(out["known_before_signal_close"])
    out["post_close_pre_next_open_bool"] = bool_series(out["post_close_pre_next_open"])
    out = out[
        out["event_type"].isin(["attention", "disposition"])
        & out["known_before_signal_close_bool"]
        & ~out["post_close_pre_next_open_bool"]
    ].copy()
    return out


def add_attention_disposition_features(
    stock_frame: pd.DataFrame,
    events: pd.DataFrame | None,
    market: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    out = stock_frame.copy()
    for feature in ATTENTION_DISPOSITION_FEATURES:
        out[feature] = 0.0
    if events is None or events.empty:
        out["days_since_last_attention_disposition"] = float(LOOKBACK_DAYS + 1)
        return out, ATTENTION_DISPOSITION_FEATURES.copy()

    trading_dates = sorted(pd.Timestamp(day) for day in market["日期"].dropna().unique())
    date_to_index = {day: idx for idx, day in enumerate(trading_dates)}
    if not date_to_index:
        fail("market trading date index is empty")
    latest_market_date = trading_dates[-1]
    eligible = events[events["signal_usable_date"].le(latest_market_date)].copy()
    eligible = eligible[eligible["signal_usable_date"].isin(date_to_index)].copy()

    records: dict[str, list[dict[str, int | str]]] = {}
    for _, row in eligible.iterrows():
        usable = pd.Timestamp(row["signal_usable_date"])
        usable_idx = date_to_index[usable]
        start = row["event_effective_start_date"]
        end = row["event_effective_end_date"]
        start_idx = date_to_index.get(pd.Timestamp(start), usable_idx) if pd.notna(start) else usable_idx
        end_idx = date_to_index.get(pd.Timestamp(end), start_idx) if pd.notna(end) else start_idx
        records.setdefault(row["stock_id"], []).append(
            {
                "event_type": row["event_type"],
                "usable_idx": int(usable_idx),
                "active_start_idx": int(start_idx),
                "active_end_idx": int(end_idx),
            }
        )
    for stock_records in records.values():
        stock_records.sort(key=lambda item: int(item["usable_idx"]))

    if "stock_id_for_event_features" not in out.columns:
        out["stock_id_for_event_features"] = normalize_stock_id(out["股票代號"])
    out["event_date_idx"] = out["日期"].map(date_to_index)
    missing_date = out["event_date_idx"].isna()
    if missing_date.any():
        out.loc[missing_date, "event_date_idx"] = -1
    out["event_date_idx"] = out["event_date_idx"].astype(int)

    for idx, row in out.iterrows():
        date_idx = int(row["event_date_idx"])
        if date_idx < 0:
            out.at[idx, "days_since_last_attention_disposition"] = float(LOOKBACK_DAYS + 1)
            continue
        stock_records = records.get(row["stock_id_for_event_features"], [])
        if not stock_records:
            out.at[idx, "days_since_last_attention_disposition"] = float(LOOKBACK_DAYS + 1)
            continue
        count_1d = sum(1 for event in stock_records if date_idx <= int(event["usable_idx"]) <= date_idx)
        count_3d = sum(1 for event in stock_records if date_idx - 2 <= int(event["usable_idx"]) <= date_idx)
        count_10d = sum(1 for event in stock_records if date_idx - 9 <= int(event["usable_idx"]) <= date_idx)
        count_20d = sum(1 for event in stock_records if date_idx - 19 <= int(event["usable_idx"]) <= date_idx)
        attention_active = any(
            event["event_type"] == "attention"
            and int(event["usable_idx"]) <= date_idx
            and int(event["active_start_idx"]) <= date_idx <= int(event["active_end_idx"])
            for event in stock_records
        )
        disposition_active = any(
            event["event_type"] == "disposition"
            and int(event["usable_idx"]) <= date_idx
            and int(event["active_start_idx"]) <= date_idx <= int(event["active_end_idx"])
            for event in stock_records
        )
        past_indices = [int(event["usable_idx"]) for event in stock_records if int(event["usable_idx"]) <= date_idx]
        days_since = date_idx - max(past_indices) if past_indices else LOOKBACK_DAYS + 1
        out.at[idx, "attention_disposition_known_count_1d"] = float(count_1d)
        out.at[idx, "attention_disposition_known_count_3d"] = float(count_3d)
        out.at[idx, "attention_disposition_known_count_10d"] = float(count_10d)
        out.at[idx, "attention_active_on_signal_date"] = float(attention_active)
        out.at[idx, "disposition_active_on_signal_date"] = float(disposition_active)
        out.at[idx, "days_since_last_attention_disposition"] = float(min(days_since, LOOKBACK_DAYS + 1))
        out.at[idx, "has_attention_disposition_history_20d"] = float(count_20d > 0)

    out = out.drop(columns=["stock_id_for_event_features", "event_date_idx"], errors="ignore")
    return out, ATTENTION_DISPOSITION_FEATURES.copy()


def add_theme_rotation_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = frame.copy()
    theme_parts: list[pd.DataFrame] = []
    feature_cols: list[str] = []

    for window in RETURN_RANK_WINDOWS:
        return_col = f"close_ret_{window}"
        theme = (
            out.groupby(["日期", "主分類"])
            .agg(
                theme_stock_count=("股票代號", "nunique"),
                **{
                    f"theme_avg_ret_{window}": (return_col, "mean"),
                    f"theme_median_ret_{window}": (return_col, "median"),
                    f"theme_positive_share_{window}": (return_col, lambda s: float((s > 0).mean())),
                },
            )
            .reset_index()
        )
        theme[f"theme_strength_rank_{window}"] = theme.groupby("日期")[f"theme_avg_ret_{window}"].rank(pct=True)
        theme[f"theme_breadth_rank_{window}"] = theme.groupby("日期")[f"theme_positive_share_{window}"].rank(pct=True)
        market_col = f"加權指數收盤_ret_{window}"
        if market_col in out.columns:
            market_daily = out[["日期", market_col]].drop_duplicates("日期")
            theme = theme.merge(market_daily, on="日期", how="left")
            theme[f"theme_vs_weighted_{window}"] = theme[f"theme_avg_ret_{window}"] - theme[market_col]
            theme = theme.drop(columns=[market_col])
        theme_parts.append(theme)
        feature_cols.extend(
            [
                f"theme_avg_ret_{window}",
                f"theme_median_ret_{window}",
                f"theme_positive_share_{window}",
                f"theme_strength_rank_{window}",
                f"theme_breadth_rank_{window}",
                f"theme_vs_weighted_{window}",
            ]
        )

    theme_daily = theme_parts[0]
    for part in theme_parts[1:]:
        keep = [c for c in part.columns if c != "theme_stock_count"]
        theme_daily = theme_daily.merge(part[keep], on=["日期", "主分類"], how="outer")

    for window in MA_RANK_WINDOWS:
        volume_col = f"volume_vs_ma_{window}"
        ma_col = f"close_vs_ma_{window}"
        part = (
            out.groupby(["日期", "主分類"])
            .agg(
                **{
                    f"theme_avg_volume_vs_ma_{window}": (volume_col, "mean"),
                    f"theme_avg_close_vs_ma_{window}": (ma_col, "mean"),
                }
            )
            .reset_index()
        )
        part[f"theme_volume_rank_{window}"] = part.groupby("日期")[f"theme_avg_volume_vs_ma_{window}"].rank(pct=True)
        part[f"theme_ma_position_rank_{window}"] = part.groupby("日期")[f"theme_avg_close_vs_ma_{window}"].rank(pct=True)
        theme_daily = theme_daily.merge(part, on=["日期", "主分類"], how="left")
        feature_cols.extend(
            [
                f"theme_avg_volume_vs_ma_{window}",
                f"theme_avg_close_vs_ma_{window}",
                f"theme_volume_rank_{window}",
                f"theme_ma_position_rank_{window}",
            ]
        )

    theme_daily["theme_acceleration_5_20"] = theme_daily["theme_avg_ret_5"] - theme_daily["theme_avg_ret_20"]
    theme_daily["theme_acceleration_rank_5_20"] = theme_daily.groupby("日期")["theme_acceleration_5_20"].rank(pct=True)
    theme_daily["theme_rotation_candidate_rank"] = (
        0.5 * theme_daily["theme_strength_rank_5"].fillna(0.5)
        + 0.3 * theme_daily["theme_acceleration_rank_5_20"].fillna(0.5)
        + 0.2 * theme_daily["theme_breadth_rank_5"].fillna(0.5)
    )
    feature_cols.extend(["theme_acceleration_5_20", "theme_acceleration_rank_5_20", "theme_rotation_candidate_rank"])

    out = out.merge(theme_daily, on=["日期", "主分類"], how="left")
    for window in RETURN_RANK_WINDOWS:
        col = f"stock_vs_theme_ret_{window}"
        out[col] = out[f"close_ret_{window}"] - out[f"theme_avg_ret_{window}"]
        feature_cols.append(col)

    feature_cols = [c for c in dict.fromkeys(feature_cols) if c in out.columns]
    out[feature_cols] = out[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out, feature_cols


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
    out["target_success"] = out["old_target_success"].astype(int)
    out["drawdown_minus3_before_or_same_success"] = (
        out["label_complete"]
        & has_profit
        & has_adverse
        & out["adverse_event_day"].le(out["profit_event_day"])
    ).astype(int)
    out["drawdown_minus3_before_success"] = (
        out["label_complete"]
        & has_profit
        & has_adverse
        & out["adverse_event_day"].lt(out["profit_event_day"])
    ).astype(int)
    out["same_day_profit_and_drawdown_minus3"] = out["same_day_both_event"].astype(int)
    out["hit_minus3_low_anytime_10d"] = (out["label_complete"] & has_adverse).astype(int)
    out["clean_success_label"] = (
        (out["target_success"] == 1)
        & (out["drawdown_minus3_before_or_same_success"] == 0)
    ).astype(int)
    out["painful_success_label"] = (
        (out["target_success"] == 1)
        & (out["drawdown_minus3_before_or_same_success"] == 1)
    ).astype(int)
    out["old_success_but_risk_failed"] = (
        (out["old_target_success"] == 1) & (out["risk_adjusted_10d_success"] == 0)
    ).astype(int)
    out["realized_10d_trade_return"] = out["future_10d_day10_close_return"]
    out.loc[~out["label_complete"], "realized_10d_trade_return"] = np.nan
    return out


def add_features(
    stock: pd.DataFrame,
    market: pd.DataFrame,
    theme: pd.DataFrame,
    attention_disposition_events: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, list[str]]:
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
    out, theme_rotation_features = add_theme_rotation_features(out)
    out, attention_disposition_features = add_attention_disposition_features(
        out,
        attention_disposition_events,
        market,
    )

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
    numeric_features.extend(theme_rotation_features)
    numeric_features.extend(attention_disposition_features)
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


def score_band_summary(scored: pd.DataFrame) -> dict:
    if scored.empty or scored["integrated_research_score"].nunique(dropna=True) < 4:
        return {
            "score_band_success_delta": math.nan,
            "score_band_advantage_delta": math.nan,
            "score_band_return_delta": math.nan,
            "score_band_order_passed": False,
        }
    work = scored.copy()
    work["score_band"] = pd.qcut(
        work["integrated_research_score"].rank(method="first"),
        q=4,
        labels=False,
        duplicates="drop",
    )
    grouped = work.groupby("score_band").agg(
        success_rate=("target_success", "mean"),
        advantage_rate=("same_day_advantage_label", "mean"),
        avg_return=("future_10d_high_close_return", "mean"),
    )
    if len(grouped) < 2:
        return {
            "score_band_success_delta": math.nan,
            "score_band_advantage_delta": math.nan,
            "score_band_return_delta": math.nan,
            "score_band_order_passed": False,
        }
    success_delta = float(grouped["success_rate"].iloc[-1] - grouped["success_rate"].iloc[0])
    advantage_delta = float(grouped["advantage_rate"].iloc[-1] - grouped["advantage_rate"].iloc[0])
    return_delta = float(grouped["avg_return"].iloc[-1] - grouped["avg_return"].iloc[0])
    return {
        "score_band_success_delta": success_delta,
        "score_band_advantage_delta": advantage_delta,
        "score_band_return_delta": return_delta,
        "score_band_order_passed": bool(success_delta > 0 and advantage_delta > 0 and return_delta > 0),
    }


def balanced_objective(row: pd.Series) -> float:
    success_lift = float(row["success_lift"]) if not pd.isna(row["success_lift"]) else -1.0
    return_lift = float(row["return_lift"]) if not pd.isna(row["return_lift"]) else -1.0
    score_band_success = float(row.get("score_band_success_delta", math.nan))
    score_band_advantage = float(row.get("score_band_advantage_delta", math.nan))
    score_band_return = float(row.get("score_band_return_delta", math.nan))
    monthly_positive = float(row.get("monthly_positive_months", 0) or 0)
    concentration_penalty = max(0.0, float(row["top_industry_share"]) - 0.50) if not pd.isna(row["top_industry_share"]) else 0.20
    weak_side = min(success_lift, return_lift)
    band_terms = [score_band_success, score_band_advantage, score_band_return]
    valid_band_terms = [value for value in band_terms if not pd.isna(value)]
    band_floor = min(valid_band_terms) if valid_band_terms else -1.0
    return (
        weak_side
        + 0.25 * success_lift
        + 0.25 * return_lift
        + 0.20 * band_floor
        + 0.05 * max(0.0, score_band_success if not pd.isna(score_band_success) else -1.0)
        + 0.02 * monthly_positive
        - 0.10 * concentration_penalty
    )


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
            row.update(score_band_summary(scored))
            row.update(monthly_lift_summary(scored, picked))
            row["balanced_objective_score"] = balanced_objective(pd.Series(row))
            rows.append(row)
    table = pd.DataFrame(rows)
    feasible = table[
        (table["days"] >= 10)
        & (table["success_lift"] > 0)
        & (table["return_lift"] > 0)
        & (table["monthly_stability_passed"])
        & (table["score_band_order_passed"])
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
            "score_band_success_delta",
            "score_band_advantage_delta",
            "score_band_return_delta",
            "success_lift",
            "return_lift",
            "success_rate",
            "days",
        ],
        ascending=[False, False, False, False, False, False, False, False, False],
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
    candidate_paths = validate_candidate_feature_inputs(config)
    stock = normalize_stock(read_csv(paths["stock_daily_all"]))
    market = normalize_market(read_csv(paths["market_daily"]))
    theme = normalize_theme(read_csv(paths["theme_group"]))
    attention_disposition_events = None
    if "attention_disposition_events" in candidate_paths:
        attention_disposition_events = normalize_attention_disposition_events(
            read_csv(candidate_paths["attention_disposition_events"])
        )
    stock_latest = stock["日期"].max()
    market_latest = market["日期"].max()
    if stock_latest != market_latest:
        fail(f"stock and market latest dates differ: {stock_latest.date()} vs {market_latest.date()}")
    labeled = add_labels(stock)
    featured, feature_cols = add_features(labeled, market, theme, attention_disposition_events)
    return add_relative_and_risk_labels(featured), feature_cols


def corr_sign(value: float) -> int:
    if pd.isna(value) or abs(float(value)) < 1e-12:
        return 0
    return 1 if float(value) > 0 else -1


def train_dev_stable(row: pd.Series, metric: str) -> bool:
    train_value = row.get(f"train_{metric}_corr")
    dev_value = row.get(f"development_{metric}_corr")
    train_sign = corr_sign(train_value)
    dev_sign = corr_sign(dev_value)
    if train_sign == 0 or train_sign != dev_sign:
        return False
    return min(abs(float(train_value)), abs(float(dev_value))) >= FEATURE_SCREEN_MIN_ABS_CORR


def screen_feature_columns(feature_cols: list[str]) -> tuple[list[str], pd.DataFrame]:
    if not FEATURE_SIGNAL_PATH.exists():
        fail("data learnability feature signal is required before feature-screened retraining")
    signal = pd.read_csv(FEATURE_SIGNAL_PATH, encoding="utf-8-sig")
    required = {
        "feature",
        "train_success_corr",
        "development_success_corr",
        "train_return_corr",
        "development_return_corr",
        "train_risk_filter_corr",
        "development_risk_filter_corr",
    }
    missing = required - set(signal.columns)
    if missing:
        fail("data_learnability_feature_signal.csv missing columns: " + ", ".join(sorted(missing)))
    available = signal[signal["feature"].isin(feature_cols)].copy()
    if available.empty:
        fail("feature screen found no overlapping features")

    rows: list[dict] = []
    for _, row in available.iterrows():
        stable_success = train_dev_stable(row, "success")
        stable_return = train_dev_stable(row, "return")
        stable_risk = train_dev_stable(row, "risk_filter")
        stable_metric_count = int(stable_success) + int(stable_return) + int(stable_risk)
        score = 0.0
        for metric, stable in [
            ("success", stable_success),
            ("return", stable_return),
            ("risk_filter", stable_risk),
        ]:
            if not stable:
                continue
            train_value = abs(float(row[f"train_{metric}_corr"]))
            dev_value = abs(float(row[f"development_{metric}_corr"]))
            score += dev_value + 0.5 * train_value
        rows.append(
            {
                "feature": row["feature"],
                "selected": False,
                "screening_score": score,
                "stable_metric_count": stable_metric_count,
                "stable_success": stable_success,
                "stable_return": stable_return,
                "stable_risk_filter": stable_risk,
                "used_holdout_for_selection": False,
                "train_success_corr": row.get("train_success_corr"),
                "development_success_corr": row.get("development_success_corr"),
                "holdout_success_corr": row.get("holdout_success_corr"),
                "train_return_corr": row.get("train_return_corr"),
                "development_return_corr": row.get("development_return_corr"),
                "holdout_return_corr": row.get("holdout_return_corr"),
                "train_risk_filter_corr": row.get("train_risk_filter_corr"),
                "development_risk_filter_corr": row.get("development_risk_filter_corr"),
                "holdout_risk_filter_corr": row.get("holdout_risk_filter_corr"),
            }
        )
    screen = pd.DataFrame(rows)
    candidates = screen[screen["stable_metric_count"] > 0].copy()
    if len(candidates) < FEATURE_SCREEN_MIN_FEATURES:
        fail(
            f"feature screen selected only {len(candidates)} candidates; "
            f"minimum is {FEATURE_SCREEN_MIN_FEATURES}"
        )
    candidates = candidates.sort_values(
        ["stable_metric_count", "screening_score"],
        ascending=[False, False],
    )
    selected_features = candidates.head(FEATURE_SCREEN_MAX_FEATURES)["feature"].astype(str).tolist()
    screen.loc[screen["feature"].isin(selected_features), "selected"] = True
    screen = screen.sort_values(["selected", "stable_metric_count", "screening_score"], ascending=[False, False, False])
    screen.to_csv(FEATURE_SCREEN_PATH, index=False, encoding="utf-8-sig")
    return selected_features, screen


def main() -> None:
    args = parse_args()
    config = load_json(CONFIG_PATH)
    plan = load_json(PLAN_PATH)
    validate_confirmation(args, plan)

    frame, all_feature_cols = build_training_frame(config)
    feature_cols, feature_screen = screen_feature_columns(all_feature_cols)
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
    candidate_region_validation_ok = bool(
        holdout_row["success_lift"] > 0
        and holdout_row["return_lift"] > 0
        and advantage_order_ok
        and return_ranking_probe_order_ok
        and holdout_probe_row["return_lift"] > 0
        and risk_order_ok
        and holdout_row["top_stock_share"] <= 0.20
        and holdout_row["top_industry_share"] <= 0.50
        and active_months >= 2
    )
    passed = candidate_region_validation_ok
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
        for c in all_feature_cols
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
        "drawdown_minus3_before_or_same_success",
        "drawdown_minus3_before_success",
        "same_day_profit_and_drawdown_minus3",
        "hit_minus3_low_anytime_10d",
        "clean_success_label",
        "painful_success_label",
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
                "- Core data sources: three approved CSV inputs.",
                "- Candidate feature input: approved attention/disposition events only.",
                "- Model: one hidden-layer numpy MLP with four outputs.",
                f"- Feature screen: selected {len(feature_cols)} of {len(all_feature_cols)} generated features.",
                "- Feature screen source: `validation_layer/data_learnability_feature_signal.csv`.",
                "- Feature screen uses train/development correlation stability only; holdout columns are audit-only.",
                f"- Feature screen min absolute train/development correlation: {FEATURE_SCREEN_MIN_ABS_CORR}.",
                f"- Feature screen max features: {FEATURE_SCREEN_MAX_FEATURES}.",
                "- Training heads: selection_success, same_day_advantage soft target, drawdown/failure_risk, episode_start.",
                "- Formal target_success is the 10-day +3% close touch rule.",
                "- Drawdown side labels: -3% low is modeled as risk context, not automatic target failure.",
                "- If +3% close happens after a -3% low, target_success remains success and painful_success_label records the path risk.",
                "- risk_adjusted_10d_success is retained only as a hard-risk comparison field.",
                "- Same-day advantage soft target: pure same-day return percentile.",
                "- Uses same-day relative return-ranking features against all stocks, same industry, and market indices.",
                "- Uses theme-rotation features inside the single main model feature contract.",
                "- Uses approved attention/disposition features as candidate risk/context inputs.",
                f"- same_day_advantage loss weight: {SAME_DAY_ADVANTAGE_LOSS_WEIGHT}.",
                "- Strategy tuning: selected on development with monthly stability and a balanced success/return objective.",
                "- Strategy tuning requires development score bands to improve success, same-day advantage, and high-close return from low to high score.",
                "- Holdout promotion uses candidate-region Top3 validation; all-row score-band ordering is retained as calibration diagnostics.",
                "- Development monthly stability requires most active months to have both success lift and return lift above zero.",
                f"- Feature lookback: {LOOKBACK_DAYS} trading days.",
                f"- Episode gap: {EPISODE_GAP_DAYS} trading days.",
                f"- Selected weights: {', '.join(str(v) for v in weights)}",
                f"- Selected gate: {gate if gate is not None else 'none'}",
                f"- Selected development positive months: {int(selected_strategy.get('monthly_positive_months', 0))}/{int(selected_strategy.get('monthly_total_months', 0))}",
                f"- Selected development score-band success delta: {float(selected_strategy.get('score_band_success_delta', math.nan)):.6f}",
                f"- Selected development score-band advantage delta: {float(selected_strategy.get('score_band_advantage_delta', math.nan)):.6f}",
                f"- Selected development score-band return delta: {float(selected_strategy.get('score_band_return_delta', math.nan)):.6f}",
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
        "target_contract": "drawdown_side_label_10d_touch_success",
        "status": status,
        "formal_approved": passed,
        "reason": reason,
        "feature_screen_enabled": True,
        "feature_screen_source": str(FEATURE_SIGNAL_PATH.relative_to(PROJECT_ROOT)),
        "feature_screen_output": str(FEATURE_SCREEN_PATH.relative_to(PROJECT_ROOT)),
        "feature_screen_uses_holdout_for_selection": False,
        "candidate_feature_inputs_enabled": True,
        "candidate_feature_input_keys": sorted(config.get("candidate_model_feature_inputs", {}).keys()),
        "attention_disposition_feature_count": len(ATTENTION_DISPOSITION_FEATURES),
        "attention_disposition_feature_names": ATTENTION_DISPOSITION_FEATURES,
        "original_feature_count": len(all_feature_cols),
        "selected_feature_count": len(feature_cols),
        "selected_feature_preview": feature_cols[:20],
        "selected_weights": list(weights),
        "selected_gate": gate,
        "training_loss_start": losses[0],
        "training_loss_end": losses[-1],
        "holdout_success_rate": float(holdout_row["success_rate"]) if not pd.isna(holdout_row["success_rate"]) else None,
        "holdout_success_lift": float(holdout_row["success_lift"]) if not pd.isna(holdout_row["success_lift"]) else None,
        "holdout_return_lift": float(holdout_row["return_lift"]) if not pd.isna(holdout_row["return_lift"]) else None,
        "holdout_old_target_success_rate": float(holdout["old_target_success"].mean()) if not holdout.empty else None,
        "holdout_primary_touch_success_rate": float(holdout["target_success"].mean()) if not holdout.empty else None,
        "holdout_risk_adjusted_success_rate": float(holdout["risk_adjusted_10d_success"].mean()) if not holdout.empty else None,
        "holdout_old_success_but_risk_failed_rate": float(holdout["old_success_but_risk_failed"].mean()) if not holdout.empty else None,
        "holdout_old_success_but_risk_failed_count": int(holdout["old_success_but_risk_failed"].sum()) if not holdout.empty else 0,
        "holdout_old_success_but_risk_failed_among_old_success": (
            float(holdout["old_success_but_risk_failed"].sum() / holdout["old_target_success"].sum())
            if not holdout.empty and int(holdout["old_target_success"].sum()) > 0
            else None
        ),
        "holdout_clean_success_rate": float(holdout["clean_success_label"].mean()) if not holdout.empty else None,
        "holdout_painful_success_rate": float(holdout["painful_success_label"].mean()) if not holdout.empty else None,
        "holdout_painful_success_among_success": (
            float(holdout["painful_success_label"].sum() / holdout["target_success"].sum())
            if not holdout.empty and int(holdout["target_success"].sum()) > 0
            else None
        ),
        "holdout_minus3_anytime_rate": float(holdout["hit_minus3_low_anytime_10d"].mean()) if not holdout.empty else None,
        "holdout_return_ranking_probe_success_lift": float(holdout_probe_row["success_lift"]) if not pd.isna(holdout_probe_row["success_lift"]) else None,
        "holdout_return_ranking_probe_return_lift": float(holdout_probe_row["return_lift"]) if not pd.isna(holdout_probe_row["return_lift"]) else None,
        "score_order_ok": score_order_ok,
        "score_band_ordering_required_for_promotion": False,
        "candidate_region_validation_ok": candidate_region_validation_ok,
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
        "selected_weight_score_band_passed": bool(selected_strategy.get("score_band_order_passed", False)),
        "development_score_band_success_delta": float(selected_strategy.get("score_band_success_delta", math.nan)),
        "development_score_band_advantage_delta": float(selected_strategy.get("score_band_advantage_delta", math.nan)),
        "development_score_band_return_delta": float(selected_strategy.get("score_band_return_delta", math.nan)),
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
                f"- Feature screen: selected {len(feature_cols)} of {len(all_feature_cols)} features",
                "- Feature screen holdout usage: audit-only, not selection",
                "- Target contract: drawdown_side_label_10d_touch_success",
                f"- Holdout primary +3% touch success rate: {fmt_pct(decision['holdout_primary_touch_success_rate'])}",
                f"- Holdout risk-adjusted success rate: {fmt_pct(decision['holdout_risk_adjusted_success_rate'])}",
                f"- Holdout success with -3% drawdown side risk among all rows: {fmt_pct(decision['holdout_old_success_but_risk_failed_rate'])}",
                f"- Holdout success with -3% drawdown side risk among successes: {fmt_pct(decision['holdout_old_success_but_risk_failed_among_old_success'])}",
                f"- Holdout clean success rate: {fmt_pct(decision['holdout_clean_success_rate'])}",
                f"- Holdout painful success rate: {fmt_pct(decision['holdout_painful_success_rate'])}",
                f"- Holdout painful success among successes: {fmt_pct(decision['holdout_painful_success_among_success'])}",
                f"- Holdout success rate: {fmt_pct(decision['holdout_success_rate'])}",
                f"- Holdout success lift: {fmt_pct(decision['holdout_success_lift'])}",
                f"- Holdout return lift: {fmt_pct(decision['holdout_return_lift'])}",
                f"- Holdout return-ranking probe success lift: {fmt_pct(decision['holdout_return_ranking_probe_success_lift'])}",
                f"- Holdout return-ranking probe return lift: {fmt_pct(decision['holdout_return_ranking_probe_return_lift'])}",
                f"- Score band ordering valid: {score_order_ok}",
                "- Score band ordering blocks promotion: False",
                f"- Candidate-region validation passed: {candidate_region_validation_ok}",
                f"- Advantage head ordering valid: {advantage_order_ok}",
                f"- Return-ranking probe ordering valid: {return_ranking_probe_order_ok}",
                f"- Risk band ordering valid: {risk_order_ok}",
                f"- Active holdout months: {active_months}",
                f"- Development monthly positive months: {decision['development_monthly_positive_months']}/{decision['development_monthly_total_months']}",
                f"- Development min monthly success lift: {fmt_pct(decision['development_min_monthly_success_lift'])}",
                f"- Development min monthly return lift: {fmt_pct(decision['development_min_monthly_return_lift'])}",
                f"- Selected weight stability passed: {decision['selected_weight_stability_passed']}",
                f"- Selected development score-band passed: {decision['selected_weight_score_band_passed']}",
                f"- Development score-band success delta: {fmt_pct(decision['development_score_band_success_delta'])}",
                f"- Development score-band advantage delta: {fmt_pct(decision['development_score_band_advantage_delta'])}",
                f"- Development score-band return delta: {fmt_pct(decision['development_score_band_return_delta'])}",
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
