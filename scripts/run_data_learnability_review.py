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
    add_features,
    add_labels,
    add_relative_and_risk_labels,
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

REVIEW_MD_PATH = VALIDATION_DIR / "data_learnability_review.md"
FEATURE_SIGNAL_PATH = VALIDATION_DIR / "data_learnability_feature_signal.csv"
FAILURE_PROFILE_PATH = VALIDATION_DIR / "data_learnability_failure_profile.csv"
DECISION_JSON_PATH = DECISION_DIR / "data_learnability_decision.json"

SPLITS = ["train", "development", "holdout"]
CORR_THRESHOLD = 0.03


def fail(message: str) -> None:
    raise SystemExit(f"FAIL: {message}")


def safe_mean(series: pd.Series) -> float:
    value = series.mean()
    return float(value) if pd.notna(value) else math.nan


def safe_corr(x: pd.Series, y: pd.Series) -> float:
    work = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(work) < 30 or work["x"].nunique() <= 1 or work["y"].nunique() <= 1:
        return math.nan
    ranked_x = work["x"].rank(pct=True)
    return float(ranked_x.corr(work["y"]))


def safe_share(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return math.nan
    counts = frame[column].fillna("unknown").value_counts(normalize=True)
    return float(counts.iloc[0]) if not counts.empty else math.nan


def pct(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.2%}"


def build_review_frame(config: dict) -> tuple[pd.DataFrame, list[str], str]:
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
    reviewed = add_relative_and_risk_labels(featured)
    reviewed["split"] = split_name(reviewed["日期"], reviewed["label_complete"], config)
    return reviewed, feature_cols, stock_latest.strftime("%Y-%m-%d")


def split_feature_signal(frame: pd.DataFrame, feature_cols: list[str], split: str) -> pd.DataFrame:
    part = frame[
        frame["split"].eq(split)
        & frame["label_complete"]
        & frame["has_full_20d_history"].eq(1)
    ].copy()
    rows: list[dict] = []
    old_success = part[part["old_target_success"].eq(1)].copy()
    for feature in feature_cols:
        if feature not in part.columns:
            continue
        feature_series = pd.to_numeric(part[feature], errors="coerce")
        old_feature_series = pd.to_numeric(old_success[feature], errors="coerce") if not old_success.empty else pd.Series(dtype=float)
        rows.append(
            {
                "feature": feature,
                f"{split}_rows": int(feature_series.notna().sum()),
                f"{split}_success_corr": safe_corr(feature_series, part["target_success"]),
                f"{split}_return_corr": safe_corr(feature_series, part["future_10d_high_close_return"]),
                f"{split}_adverse_corr": safe_corr(feature_series, part["max_adverse_return"]),
                f"{split}_risk_filter_corr": safe_corr(old_feature_series, old_success["target_success"]) if not old_success.empty else math.nan,
                f"{split}_missing_rate": float(feature_series.isna().mean()) if len(feature_series) else math.nan,
            }
        )
    return pd.DataFrame(rows)


def sign(value: float) -> int:
    if pd.isna(value) or abs(value) < 1e-9:
        return 0
    return 1 if value > 0 else -1


def add_stability_flags(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    for metric in ["success_corr", "return_corr", "risk_filter_corr"]:
        split_cols = [f"{split}_{metric}" for split in SPLITS]
        out[f"{metric}_direction_stable"] = False
        out[f"{metric}_min_abs_corr"] = np.nan
        for idx, row in out.iterrows():
            values = [row[col] for col in split_cols]
            signs = [sign(value) for value in values]
            stable = signs[0] != 0 and signs.count(signs[0]) == len(signs)
            out.loc[idx, f"{metric}_direction_stable"] = bool(stable)
            if all(pd.notna(value) for value in values):
                out.loc[idx, f"{metric}_min_abs_corr"] = min(abs(float(value)) for value in values)
    return out


def build_feature_signal(frame: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    for split in SPLITS:
        table = split_feature_signal(frame, feature_cols, split)
        merged = table if merged is None else merged.merge(table, on="feature", how="outer")
    if merged is None:
        return pd.DataFrame()
    merged = add_stability_flags(merged)
    merged["holdout_abs_success_corr"] = merged["holdout_success_corr"].abs()
    merged["holdout_abs_return_corr"] = merged["holdout_return_corr"].abs()
    merged["holdout_abs_risk_filter_corr"] = merged["holdout_risk_filter_corr"].abs()
    return merged.sort_values(
        [
            "success_corr_direction_stable",
            "risk_filter_corr_direction_stable",
            "holdout_abs_success_corr",
            "holdout_abs_risk_filter_corr",
            "holdout_abs_return_corr",
        ],
        ascending=[False, False, False, False, False],
    )


def profile_record(section: str, split: str, group: str, part: pd.DataFrame) -> dict:
    complete = part[part["label_complete"]].copy()
    old_success_count = int(complete["old_target_success"].sum()) if not complete.empty else 0
    risk_failed_count = int(complete["old_success_but_risk_failed"].sum()) if not complete.empty else 0
    return {
        "section": section,
        "split": split,
        "group": group,
        "rows": int(len(complete)),
        "days": int(complete["日期"].nunique()) if not complete.empty else 0,
        "stocks": int(complete["股票代號"].nunique()) if not complete.empty else 0,
        "old_success_rate": safe_mean(complete["old_target_success"]) if not complete.empty else math.nan,
        "risk_adjusted_success_rate": safe_mean(complete["target_success"]) if not complete.empty else math.nan,
        "old_success_but_risk_failed_rate": safe_mean(complete["old_success_but_risk_failed"]) if not complete.empty else math.nan,
        "risk_failed_among_old_success": risk_failed_count / old_success_count if old_success_count else math.nan,
        "avg_10d_high_close_return": safe_mean(complete["future_10d_high_close_return"]) if not complete.empty else math.nan,
        "avg_max_adverse_return": safe_mean(complete["max_adverse_return"]) if not complete.empty else math.nan,
        "avg_realized_10d_trade_return": safe_mean(complete["realized_10d_trade_return"]) if not complete.empty else math.nan,
        "top_stock_share": safe_share(complete, "股票代號"),
        "top_industry_share": safe_share(complete, "主分類"),
    }


def add_quantile_group(frame: pd.DataFrame, column: str, target: str) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    if values.notna().sum() < 10 or values.nunique(dropna=True) < 3:
        return pd.Series("unknown", index=frame.index)
    labels = ["low", "middle", "high"]
    return pd.qcut(values.rank(method="first"), q=3, labels=labels, duplicates="drop").astype(str)


def build_failure_profile(frame: pd.DataFrame) -> pd.DataFrame:
    completed = frame[frame["label_complete"]].copy()
    records: list[dict] = []
    for split in SPLITS:
        split_frame = completed[completed["split"].eq(split)]
        records.append(profile_record("overall", split, split, split_frame))
        for event_type, part in split_frame.groupby("first_event_type", sort=True):
            records.append(profile_record("first_event_type", split, str(event_type), part))
        for industry, part in split_frame.groupby("主分類", sort=True):
            records.append(profile_record("industry", split, str(industry), part))
        month_frame = split_frame.copy()
        month_frame["month"] = month_frame["日期"].dt.strftime("%Y-%m")
        for month, part in month_frame.groupby("month", sort=True):
            records.append(profile_record("month", split, str(month), part))
        if "加權指數收盤_ret_20" in split_frame.columns:
            market_frame = split_frame.copy()
            market_frame["weighted_20d_return_group"] = add_quantile_group(
                market_frame, "加權指數收盤_ret_20", "weighted_20d_return_group"
            )
            for group, part in market_frame.groupby("weighted_20d_return_group", sort=True):
                records.append(profile_record("market_weighted_20d_return", split, str(group), part))
        if "market_breadth" in split_frame.columns:
            breadth_frame = split_frame.copy()
            breadth_frame["market_breadth_group"] = add_quantile_group(
                breadth_frame, "market_breadth", "market_breadth_group"
            )
            for group, part in breadth_frame.groupby("market_breadth_group", sort=True):
                records.append(profile_record("market_breadth", split, str(group), part))
    return pd.DataFrame(records)


def decide(feature_signal: pd.DataFrame, failure_profile: pd.DataFrame) -> dict:
    stable_success = feature_signal[
        feature_signal["success_corr_direction_stable"]
        & (feature_signal["holdout_abs_success_corr"] >= CORR_THRESHOLD)
    ].copy()
    stable_risk_filter = feature_signal[
        feature_signal["risk_filter_corr_direction_stable"]
        & (feature_signal["holdout_abs_risk_filter_corr"] >= CORR_THRESHOLD)
    ].copy()
    stable_return = feature_signal[
        feature_signal["return_corr_direction_stable"]
        & (feature_signal["holdout_abs_return_corr"] >= CORR_THRESHOLD)
    ].copy()
    holdout_overall = failure_profile[
        failure_profile["section"].eq("overall") & failure_profile["split"].eq("holdout")
    ]
    holdout = holdout_overall.iloc[0].to_dict() if not holdout_overall.empty else {}
    stable_success_count = int(len(stable_success))
    stable_risk_filter_count = int(len(stable_risk_filter))
    stable_return_count = int(len(stable_return))

    if stable_success_count >= 8 and stable_risk_filter_count >= 5 and stable_return_count >= 5:
        status = "learnable_signal_present"
        recommended = "feature_screen_then_retrain"
        reason = "三份資料內有跨 train/development/holdout 方向穩定的成功、風險過濾與報酬排序訊號。"
    elif stable_success_count >= 3 or stable_risk_filter_count >= 2 or stable_return_count >= 3:
        status = "weak_signal_but_not_enough_for_full_retrain"
        recommended = "feature_screen_before_any_retrain"
        reason = "有少量訊號，但不足以直接支持完整主模型重訓；應先縮小特徵集合與目標問題。"
    else:
        status = "insufficient_signal_in_current_inputs"
        recommended = "review_target_or_add_data"
        reason = "三份資料對先 +3% 且不能先 -3% 的順序風險，沒有足夠穩定訊號。"

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": status,
        "recommended_next_step": recommended,
        "reason": reason,
        "stable_success_feature_count": stable_success_count,
        "stable_risk_filter_feature_count": stable_risk_filter_count,
        "stable_return_feature_count": stable_return_count,
        "holdout_risk_adjusted_success_rate": holdout.get("risk_adjusted_success_rate"),
        "holdout_old_success_but_risk_failed_rate": holdout.get("old_success_but_risk_failed_rate"),
        "holdout_risk_failed_among_old_success": holdout.get("risk_failed_among_old_success"),
        "top_success_features": stable_success.head(10)["feature"].tolist(),
        "top_risk_filter_features": stable_risk_filter.head(10)["feature"].tolist(),
        "top_return_features": stable_return.head(10)["feature"].tolist(),
    }


def write_review_md(data_latest: str, decision: dict, feature_signal: pd.DataFrame, failure_profile: pd.DataFrame) -> None:
    holdout = failure_profile[
        failure_profile["section"].eq("overall") & failure_profile["split"].eq("holdout")
    ]
    holdout_row = holdout.iloc[0].to_dict() if not holdout.empty else {}
    lines = [
        "# Data Learnability Review",
        "",
        f"- Generated: {decision['generated_at']}",
        f"- Data latest date: {data_latest}",
        "- Scope: data/label learnability only; no model training; no stock candidates.",
        "- Formal output: unchanged by this review.",
        "",
        "## 白話結論",
        "",
        f"{decision['reason']}",
        "",
        f"- Review status: `{decision['status']}`",
        f"- Recommended next step: `{decision['recommended_next_step']}`",
        "",
        "## Holdout Snapshot",
        "",
        f"- Risk-adjusted success rate: {pct(holdout_row.get('risk_adjusted_success_rate'))}",
        f"- Old success but risk-failed rate: {pct(holdout_row.get('old_success_but_risk_failed_rate'))}",
        f"- Risk-failed among old successes: {pct(holdout_row.get('risk_failed_among_old_success'))}",
        f"- Average max adverse return: {pct(holdout_row.get('avg_max_adverse_return'))}",
        f"- Average realized rule return: {pct(holdout_row.get('avg_realized_10d_trade_return'))}",
        "",
        "## Learnable Signal Counts",
        "",
        f"- Stable success features: {decision['stable_success_feature_count']}",
        f"- Stable risk-filter features: {decision['stable_risk_filter_feature_count']}",
        f"- Stable return-ranking features: {decision['stable_return_feature_count']}",
        "",
        "## Top Feature Clues",
        "",
        "These are distinguishability clues, not buy reasons; a stable negative direction can still be useful for filtering risk.",
        "",
        "- Success target clues: " + (", ".join(decision["top_success_features"][:8]) or "none"),
        "- Risk-filter clues: " + (", ".join(decision["top_risk_filter_features"][:8]) or "none"),
        "- Return-ranking clues: " + (", ".join(decision["top_return_features"][:8]) or "none"),
        "",
        "## Boundary",
        "",
        "- This is not a probability model.",
        "- This does not update formal candidates.",
        "- This does not add a new model branch.",
        "- It only decides whether the current three CSV inputs contain stable enough signal for the risk-adjusted target.",
        "",
        "## Outputs",
        "",
        f"- `{FEATURE_SIGNAL_PATH.relative_to(PROJECT_ROOT)}`",
        f"- `{FAILURE_PROFILE_PATH.relative_to(PROJECT_ROOT)}`",
        f"- `{DECISION_JSON_PATH.relative_to(PROJECT_ROOT)}`",
        "",
    ]
    REVIEW_MD_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    config = load_json(CONFIG_PATH)
    frame, feature_cols, data_latest = build_review_frame(config)
    feature_signal = build_feature_signal(frame, feature_cols)
    failure_profile = build_failure_profile(frame)
    decision = decide(feature_signal, failure_profile)

    feature_signal.to_csv(FEATURE_SIGNAL_PATH, index=False, encoding="utf-8-sig")
    failure_profile.to_csv(FAILURE_PROFILE_PATH, index=False, encoding="utf-8-sig")
    DECISION_JSON_PATH.write_text(json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_review_md(data_latest, decision, feature_signal, failure_profile)

    print("OK: data learnability review completed")
    print(f"STATUS: {decision['status']}")
    print(f"NEXT_STEP: {decision['recommended_next_step']}")
    print(f"REVIEW: {REVIEW_MD_PATH}")


if __name__ == "__main__":
    main()
