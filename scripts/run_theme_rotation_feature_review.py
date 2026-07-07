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
    LOOKAHEAD_DAYS,
    LOOKBACK_DAYS,
    PROFIT_THRESHOLD,
    load_json,
    normalize_market,
    normalize_stock,
    normalize_stock_id,
    normalize_theme,
    read_csv,
    split_name,
    validate_inputs,
)


VALIDATION_DIR = PROJECT_ROOT / "validation_layer"
DECISION_DIR = PROJECT_ROOT / "decision_layer"

REVIEW_MD_PATH = VALIDATION_DIR / "theme_rotation_feature_review.md"
SUMMARY_PATH = VALIDATION_DIR / "theme_rotation_feature_summary.csv"
DAILY_PATH = VALIDATION_DIR / "theme_rotation_daily_strength.csv"
DECISION_JSON_PATH = DECISION_DIR / "theme_rotation_feature_decision.json"

WINDOWS = [1, 3, 5, 10, 20]
MIN_ABS_CORR = 0.01
MIN_STABLE_FEATURES = 6
TOP_PREVIEW = 15


def fail(message: str) -> None:
    raise SystemExit(f"FAIL: {message}")


def safe_corr(a: pd.Series, b: pd.Series) -> float:
    work = pd.DataFrame({"a": a, "b": b}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(work) < 30:
        return math.nan
    if work["a"].nunique(dropna=True) < 2 or work["b"].nunique(dropna=True) < 2:
        return math.nan
    return float(work["a"].corr(work["b"]))


def same_direction(a: float, b: float) -> bool:
    if pd.isna(a) or pd.isna(b):
        return False
    return (a >= 0 and b >= 0) or (a <= 0 and b <= 0)


def fmt_pct(value: object) -> str:
    try:
        if value is None or pd.isna(value):
            return "N/A"
        return f"{float(value):.2%}"
    except (TypeError, ValueError):
        return "N/A"


def build_base_frame(config: dict) -> tuple[pd.DataFrame, str]:
    paths = validate_inputs(config)
    stock = normalize_stock(read_csv(paths["stock_daily_all"]))
    stock["股票代號"] = normalize_stock_id(stock["股票代號"])
    market = normalize_market(read_csv(paths["market_daily"]))
    theme = normalize_theme(read_csv(paths["theme_group"]))
    theme["股票代號"] = normalize_stock_id(theme["股票代號"])

    if stock["日期"].max() != market["日期"].max():
        fail(
            "stock and market latest dates differ: "
            f"{stock['日期'].max().date()} vs {market['日期'].max().date()}"
        )

    out = stock.merge(market, on="日期", how="left", suffixes=("", "_market"))
    out = out.merge(theme, on="股票代號", how="left")
    out["主分類"] = out["主分類"].fillna("unknown")
    out["子分類"] = out["子分類"].fillna("unknown")
    out["股票名稱"] = out["股票名稱"].fillna("")
    out = out.sort_values(["股票代號", "日期"]).reset_index(drop=True)

    group = out.groupby("股票代號", sort=False)
    out["stock_trading_index"] = group.cumcount()
    out["buy_open_next"] = group["開盤價"].shift(-1)
    for horizon in range(1, LOOKAHEAD_DAYS + 1):
        out[f"future_close_{horizon}"] = group["收盤價"].shift(-horizon)
        out[f"future_date_{horizon}"] = group["日期"].shift(-horizon)

    close_cols = [f"future_close_{i}" for i in range(1, LOOKAHEAD_DAYS + 1)]
    date_cols = [f"future_date_{i}" for i in range(1, LOOKAHEAD_DAYS + 1)]
    out["label_complete"] = (
        out[close_cols].notna().all(axis=1)
        & out[date_cols].notna().all(axis=1)
        & out["buy_open_next"].notna()
        & (out["buy_open_next"] > 0)
    )
    close_returns = out[close_cols].div(out["buy_open_next"], axis=0) - 1.0
    close_returns = close_returns.replace([np.inf, -np.inf], np.nan)
    out["future_10d_high_close_return"] = close_returns.max(axis=1)
    out["target_success"] = (
        out["label_complete"] & (out["future_10d_high_close_return"] >= PROFIT_THRESHOLD)
    ).astype(int)

    for window in WINDOWS:
        out[f"stock_ret_{window}"] = group["收盤價"].pct_change(window)
    for window in [5, 10, 20]:
        ma = group["收盤價"].transform(lambda s: s.rolling(window, min_periods=window).mean())
        vol_ma = group["成交量(張)"].transform(lambda s: s.rolling(window, min_periods=window).mean())
        out[f"stock_close_vs_ma_{window}"] = out["收盤價"] / ma - 1.0
        out[f"stock_volume_vs_ma_{window}"] = out["成交量(張)"] / vol_ma - 1.0

    for market_col in ["加權指數收盤", "電子指數收盤", "櫃買指數收盤"]:
        for window in WINDOWS:
            out[f"{market_col}_ret_{window}"] = out[market_col].pct_change(window)

    completed = out["label_complete"]
    percentiles = out[completed].groupby("日期")["future_10d_high_close_return"].rank(pct=True)
    out["same_day_return_percentile"] = np.nan
    out.loc[percentiles.index, "same_day_return_percentile"] = percentiles.astype(float)
    out["split"] = split_name(out["日期"], out["label_complete"], config)
    out["has_full_20d_history"] = (out["stock_trading_index"] >= LOOKBACK_DAYS - 1).astype(int)
    return out, stock["日期"].max().strftime("%Y-%m-%d")


def add_theme_rotation_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    out = frame.copy()
    daily_theme_parts: list[pd.DataFrame] = []
    feature_cols: list[str] = []
    daily_feature_cols: list[str] = []

    for window in WINDOWS:
        ret_col = f"stock_ret_{window}"
        theme = (
            out.groupby(["日期", "主分類"])
            .agg(
                theme_stock_count=("股票代號", "nunique"),
                **{
                    f"theme_avg_ret_{window}": (ret_col, "mean"),
                    f"theme_median_ret_{window}": (ret_col, "median"),
                    f"theme_positive_share_{window}": (ret_col, lambda s: float((s > 0).mean())),
                },
            )
            .reset_index()
        )
        theme[f"theme_strength_rank_{window}"] = theme.groupby("日期")[f"theme_avg_ret_{window}"].rank(pct=True)
        theme[f"theme_breadth_rank_{window}"] = theme.groupby("日期")[f"theme_positive_share_{window}"].rank(pct=True)
        market_col = f"加權指數收盤_ret_{window}"
        market_daily = out[["日期", market_col]].drop_duplicates("日期")
        theme = theme.merge(market_daily, on="日期", how="left")
        theme[f"theme_vs_weighted_{window}"] = theme[f"theme_avg_ret_{window}"] - theme[market_col]
        theme = theme.drop(columns=[market_col])
        daily_theme_parts.append(theme)
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
        daily_feature_cols.extend(feature_cols[-6:])

    daily_theme = daily_theme_parts[0]
    for part in daily_theme_parts[1:]:
        keep = [c for c in part.columns if c not in {"theme_stock_count"}]
        daily_theme = daily_theme.merge(part[keep], on=["日期", "主分類"], how="outer")

    for window in [5, 10, 20]:
        vol_col = f"stock_volume_vs_ma_{window}"
        ma_col = f"stock_close_vs_ma_{window}"
        part = (
            out.groupby(["日期", "主分類"])
            .agg(
                **{
                    f"theme_avg_volume_vs_ma_{window}": (vol_col, "mean"),
                    f"theme_avg_close_vs_ma_{window}": (ma_col, "mean"),
                }
            )
            .reset_index()
        )
        part[f"theme_volume_rank_{window}"] = part.groupby("日期")[f"theme_avg_volume_vs_ma_{window}"].rank(pct=True)
        part[f"theme_ma_position_rank_{window}"] = part.groupby("日期")[f"theme_avg_close_vs_ma_{window}"].rank(pct=True)
        daily_theme = daily_theme.merge(part, on=["日期", "主分類"], how="left")
        feature_cols.extend(
            [
                f"theme_avg_volume_vs_ma_{window}",
                f"theme_avg_close_vs_ma_{window}",
                f"theme_volume_rank_{window}",
                f"theme_ma_position_rank_{window}",
            ]
        )
        daily_feature_cols.extend(feature_cols[-4:])

    daily_theme["theme_acceleration_5_20"] = daily_theme["theme_avg_ret_5"] - daily_theme["theme_avg_ret_20"]
    daily_theme["theme_acceleration_rank_5_20"] = daily_theme.groupby("日期")["theme_acceleration_5_20"].rank(pct=True)
    daily_theme["theme_rotation_candidate_rank"] = (
        0.5 * daily_theme["theme_strength_rank_5"].fillna(0.5)
        + 0.3 * daily_theme["theme_acceleration_rank_5_20"].fillna(0.5)
        + 0.2 * daily_theme["theme_breadth_rank_5"].fillna(0.5)
    )
    feature_cols.extend(["theme_acceleration_5_20", "theme_acceleration_rank_5_20", "theme_rotation_candidate_rank"])
    daily_feature_cols.extend(["theme_acceleration_5_20", "theme_acceleration_rank_5_20", "theme_rotation_candidate_rank"])

    out = out.merge(daily_theme, on=["日期", "主分類"], how="left")
    for window in WINDOWS:
        col = f"stock_vs_theme_ret_{window}"
        out[col] = out[f"stock_ret_{window}"] - out[f"theme_avg_ret_{window}"]
        feature_cols.append(col)

    feature_cols = [c for c in dict.fromkeys(feature_cols) if c in out.columns]
    out[feature_cols] = out[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    daily_feature_cols = [c for c in dict.fromkeys(daily_feature_cols) if c in daily_theme.columns]
    daily_theme[daily_feature_cols] = daily_theme[daily_feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out, feature_cols, daily_theme


def evaluate_features(frame: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    usable = frame[(frame["label_complete"]) & (frame["has_full_20d_history"] == 1)].copy()
    rows: list[dict] = []
    targets = {
        "success": "target_success",
        "return": "future_10d_high_close_return",
        "same_day_rank": "same_day_return_percentile",
    }
    for feature in feature_cols:
        row: dict[str, object] = {"feature": feature}
        stable_count = 0
        max_dev_abs = 0.0
        for metric, target in targets.items():
            train_corr = safe_corr(usable[usable["split"] == "train"][feature], usable[usable["split"] == "train"][target])
            dev_corr = safe_corr(
                usable[usable["split"] == "development"][feature],
                usable[usable["split"] == "development"][target],
            )
            holdout_corr = safe_corr(
                usable[usable["split"] == "holdout"][feature],
                usable[usable["split"] == "holdout"][target],
            )
            stable = (
                same_direction(train_corr, dev_corr)
                and abs(train_corr) >= MIN_ABS_CORR
                and abs(dev_corr) >= MIN_ABS_CORR
            )
            row[f"train_{metric}_corr"] = train_corr
            row[f"development_{metric}_corr"] = dev_corr
            row[f"holdout_{metric}_corr"] = holdout_corr
            row[f"stable_{metric}"] = stable
            stable_count += int(stable)
            if not pd.isna(dev_corr):
                max_dev_abs = max(max_dev_abs, abs(float(dev_corr)))
        row["stable_metric_count"] = stable_count
        row["screening_score"] = stable_count + max_dev_abs
        row["used_holdout_for_selection"] = False
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary = summary.sort_values(["stable_metric_count", "screening_score"], ascending=[False, False])
    return summary


def decide(summary: pd.DataFrame) -> dict:
    candidates = summary[summary["stable_metric_count"] >= 2].copy()
    stable_features = int(len(candidates))
    if stable_features >= MIN_STABLE_FEATURES:
        status = "candidate_for_main_model_feature_integration"
        next_step = "plan_theme_rotation_feature_contract"
        reason = (
            "Theme rotation features show stable train/development learnability and can be planned as main-model features."
        )
    elif stable_features > 0:
        status = "research_signal_but_not_ready"
        next_step = "keep_research_only"
        reason = "Theme rotation features have some signal, but not enough stable features for main-model integration."
    else:
        status = "discard_for_now"
        next_step = "do_not_integrate_theme_rotation_features"
        reason = "Theme rotation features did not show stable train/development learnability."
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": status,
        "recommended_next_step": next_step,
        "reason": reason,
        "minimum_stable_features": MIN_STABLE_FEATURES,
        "stable_feature_count": stable_features,
        "used_holdout_for_selection": False,
        "formal_output_changed": False,
    }


def write_review(summary: pd.DataFrame, decision: dict, latest_date: str, feature_count: int) -> None:
    top = summary.head(TOP_PREVIEW)
    lines = [
        "# Theme Rotation Feature Review",
        "",
        f"- Generated: {decision['generated_at']}",
        f"- Data latest date: {latest_date}",
        "- Review type: feature learnability review only.",
        "- Formal output: unchanged by this review.",
        "- This does not choose stocks.",
        "- This does not train or promote a model.",
        "- This does not create a second decision layer.",
        "- Holdout columns are audit-only, not used for feature selection.",
        "- research_score is not used and no probability is produced.",
        "",
        "## Decision",
        "",
        f"- Status: `{decision['status']}`",
        f"- Recommended next step: `{decision['recommended_next_step']}`",
        f"- Reason: {decision['reason']}",
        f"- Generated theme-rotation features: {feature_count}",
        f"- Stable candidate features: {decision['stable_feature_count']}",
        "",
        "## Top Feature Signals",
        "",
        "| feature | stable metrics | dev success corr | dev return corr | dev same-day rank corr | holdout return corr |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in top.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["feature"]),
                    str(int(row["stable_metric_count"])),
                    fmt_pct(row["development_success_corr"]),
                    fmt_pct(row["development_return_corr"]),
                    fmt_pct(row["development_same_day_rank_corr"]),
                    fmt_pct(row["holdout_return_corr"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Plain Meaning",
            "",
            "- This review checks whether group rotation can be useful as input material for the single main model.",
            "- It is not a new stock-picking module.",
            "- If accepted later, useful fields must be integrated into the existing main model feature contract.",
            "- The formal daily report must remain one report, not a comparison between branches.",
            "",
        ]
    )
    REVIEW_MD_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    config = load_json(CONFIG_PATH)
    base, latest_date = build_base_frame(config)
    featured, feature_cols, daily_theme = add_theme_rotation_features(base)
    summary = evaluate_features(featured, feature_cols)
    decision = decide(summary)

    summary.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")
    daily_theme.to_csv(DAILY_PATH, index=False, encoding="utf-8-sig")
    DECISION_JSON_PATH.write_text(json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_review(summary, decision, latest_date, len(feature_cols))

    print("OK: theme rotation feature review completed")
    print(f"STATUS: {decision['status']}")
    print(f"NEXT_STEP: {decision['recommended_next_step']}")
    print(f"REVIEW: {REVIEW_MD_PATH}")


if __name__ == "__main__":
    main()
