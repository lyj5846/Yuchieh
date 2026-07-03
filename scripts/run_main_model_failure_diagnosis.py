from __future__ import annotations

import json
import math
import subprocess
import sys
import hashlib
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

from run_main_model_training_pipeline import build_training_frame, split_name


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "project_config.json"
MODEL_DIR = PROJECT_ROOT / "model_layer"
VALIDATION_DIR = PROJECT_ROOT / "validation_layer"
DECISION_DIR = PROJECT_ROOT / "decision_layer"
FORMAL_DIR = PROJECT_ROOT / ("formal" + "_layer")

SCORES_PATH = MODEL_DIR / "main_model_scores.csv"
VALIDATION_SUMMARY_PATH = VALIDATION_DIR / "main_model_validation_summary.csv"
CALIBRATION_PATH = VALIDATION_DIR / "main_model_calibration.csv"
DECISION_JSON_PATH = DECISION_DIR / "main_model_decision.json"
FORMAL_STATUS_PATH = FORMAL_DIR / "formal_status.md"
FORMAL_CANDIDATES_PATH = FORMAL_DIR / "formal_candidates.csv"

DIAGNOSIS_MD_PATH = VALIDATION_DIR / "main_model_failure_diagnosis.md"
DIAGNOSIS_CSV_PATH = VALIDATION_DIR / "main_model_failure_diagnosis.csv"
RECOMMENDATION_PATH = VALIDATION_DIR / "main_model_repair_recommendation.json"

EPISODE_GAP_DAYS = 10

def fail(message: str) -> None:
    raise SystemExit(f"FAIL: {message}")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def formal_candidate_row_count() -> int:
    if not FORMAL_CANDIDATES_PATH.exists():
        return 0
    with FORMAL_CANDIDATES_PATH.open("r", encoding="utf-8-sig") as f:
        return max(0, sum(1 for _ in f) - 1)


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_allowed_inputs(config: dict) -> dict[str, Path]:
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


def max_date_from_csv(path: Path) -> str:
    df = pd.read_csv(path, encoding="utf-8-sig", usecols=["日期"])
    return pd.to_datetime(df["日期"]).max().strftime("%Y-%m-%d")


def require_columns(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [col for col in cols if col not in df.columns]
    if missing:
        fail(f"{name} missing columns: {', '.join(missing)}")


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
            stock_index = int(row["diagnostic_trading_index"])
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


def summary_for_picks(full: pd.DataFrame, picked: pd.DataFrame) -> dict[str, float]:
    if picked.empty:
        return {
            "rows": 0,
            "days": 0,
            "success_rate": math.nan,
            "same_day_baseline_success_rate": math.nan,
            "success_lift": math.nan,
            "avg_10d_high_close_return": math.nan,
            "same_day_baseline_avg_return": math.nan,
            "return_lift": math.nan,
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
        "rows": float(len(picked)),
        "days": float(picked["日期"].nunique()),
        "success_rate": float(picked["target_success"].mean()),
        "same_day_baseline_success_rate": float(joined["day_success"].mean()),
        "success_lift": float(joined["pick_success"].mean() - joined["day_success"].mean()),
        "avg_10d_high_close_return": float(picked["future_10d_high_close_return"].mean()),
        "same_day_baseline_avg_return": float(joined["day_return"].mean()),
        "return_lift": float(joined["pick_return"].mean() - joined["day_return"].mean()),
    }


def band_table(df: pd.DataFrame, score_col: str, target_col: str, bands: int = 4) -> pd.DataFrame:
    work = df[[score_col, target_col, "future_10d_high_close_return"]].replace([np.inf, -np.inf], np.nan).dropna()
    if work.empty:
        return pd.DataFrame(columns=["band", "rows", "avg_score", "actual_rate", "avg_10d_high_close_return"])
    try:
        work["band"] = pd.qcut(work[score_col], q=bands, labels=False, duplicates="drop")
    except ValueError:
        work["band"] = 0
    return (
        work.groupby("band", dropna=False)
        .agg(
            rows=(target_col, "size"),
            avg_score=(score_col, "mean"),
            actual_rate=(target_col, "mean"),
            avg_10d_high_close_return=("future_10d_high_close_return", "mean"),
        )
        .reset_index()
    )


def high_low_delta(table: pd.DataFrame, metric: str = "actual_rate") -> float:
    if table.empty or len(table) < 2:
        return math.nan
    ordered = table.sort_values("band")
    return float(ordered.iloc[-1][metric] - ordered.iloc[0][metric])


def status_from_delta(delta: float, expected_positive: bool = True) -> str:
    if math.isnan(delta):
        return "unknown"
    if expected_positive:
        return "pass" if delta > 0 else "fail"
    return "pass" if delta < 0 else "fail"


def add_row(
    rows: list[dict],
    diagnostic_area: str,
    check_name: str,
    split: str,
    metric: str,
    value: float | int | str,
    reference: float | int | str,
    status: str,
    finding: str,
    repair_signal: str,
    extra: dict | None = None,
) -> None:
    row = {
        "diagnostic_area": diagnostic_area,
        "check_name": check_name,
        "split": split,
        "metric": metric,
        "value": value,
        "threshold_or_reference": reference,
        "status": status,
        "finding": finding,
        "repair_signal": repair_signal,
    }
    if extra:
        row.update(extra)
    rows.append(row)


def format_pct(value: float) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:.2%}"


def safe_corr(df: pd.DataFrame, left: str, right: str) -> float:
    if left not in df.columns or right not in df.columns:
        return math.nan
    work = df[[left, right]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(work) < 20:
        return math.nan
    if work[left].nunique(dropna=True) < 2 or work[right].nunique(dropna=True) < 2:
        return math.nan
    return float(work[left].corr(work[right]))


def sign_of(value: float) -> int:
    if math.isnan(value) or abs(value) < 1e-9:
        return 0
    return 1 if value > 0 else -1


def correlation_status(train_value: float, dev_value: float, holdout_value: float) -> str:
    signs = [sign_of(train_value), sign_of(dev_value), sign_of(holdout_value)]
    if 0 in signs:
        return "unknown"
    return "pass" if signs[0] == signs[1] == signs[2] else "fail"


def add_feature_learnability_rows(
    rows: list[dict],
    feature_frame: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    target_name: str,
    top_n: int = 10,
) -> tuple[int, int]:
    splits = {
        "train": feature_frame[feature_frame["split"].eq("train")],
        "development": feature_frame[feature_frame["split"].eq("development")],
        "holdout": feature_frame[feature_frame["split"].eq("holdout")],
    }
    train_corrs = []
    for feature in feature_cols:
        corr = safe_corr(splits["train"], feature, target_col)
        if not math.isnan(corr):
            train_corrs.append((feature, corr))
    train_corrs.sort(key=lambda item: abs(item[1]), reverse=True)
    selected = train_corrs[:top_n]
    stable_count = 0
    checked_count = 0
    for feature, train_corr in selected:
        dev_corr = safe_corr(splits["development"], feature, target_col)
        holdout_corr = safe_corr(splits["holdout"], feature, target_col)
        status = correlation_status(train_corr, dev_corr, holdout_corr)
        if status != "unknown":
            checked_count += 1
        if status == "pass":
            stable_count += 1
        add_row(
            rows,
            "feature_learnability",
            f"{target_name}_feature_direction_{feature}",
            "train_development_holdout",
            "holdout_corr",
            holdout_corr,
            f"train_corr={train_corr:.6f}; development_corr={dev_corr:.6f}",
            status,
            f"{feature} direction stability against {target_name}",
            "repair_return_ranking_features",
            extra={
                "feature_name": feature,
                "target_name": target_name,
                "train_value": train_corr,
                "development_value": dev_corr,
                "holdout_value": holdout_corr,
                "direction_stability": status,
            },
        )
    add_row(
        rows,
        "feature_learnability",
        f"{target_name}_stable_feature_count",
        "train_development_holdout",
        "stable_features",
        stable_count,
        f"checked_top_features={checked_count}",
        "pass" if stable_count >= 3 else "fail",
        f"stable top feature count for {target_name}",
        "repair_return_ranking_features",
        extra={"target_name": target_name, "stable_feature_count": stable_count, "checked_feature_count": checked_count},
    )
    return stable_count, checked_count


def daily_pick_vs_baseline(full: pd.DataFrame, picked: pd.DataFrame) -> pd.DataFrame:
    if picked.empty:
        return pd.DataFrame(columns=["日期", "pick_success", "pick_return", "day_success", "day_return", "success_lift", "return_lift"])
    selected_days = picked["日期"].drop_duplicates()
    baseline = full[full["日期"].isin(selected_days)].groupby("日期").agg(
        day_success=("target_success", "mean"),
        day_return=("future_10d_high_close_return", "mean"),
    )
    chosen = picked.groupby("日期").agg(
        pick_success=("target_success", "mean"),
        pick_return=("future_10d_high_close_return", "mean"),
    )
    joined = chosen.join(baseline, how="inner").reset_index()
    joined["success_lift"] = joined["pick_success"] - joined["day_success"]
    joined["return_lift"] = joined["pick_return"] - joined["day_return"]
    return joined


def raw_score_top3_summary(full: pd.DataFrame, score_col: str) -> dict[str, float]:
    work = full.copy()
    original_score = work["integrated_research_score"].copy()
    work["integrated_research_score"] = work[score_col]
    picked = select_top3(work, gate=None)
    result = summary_for_picks(work, picked)
    work["integrated_research_score"] = original_score
    return result


def main() -> None:
    config = read_json(CONFIG_PATH)
    paths = validate_allowed_inputs(config)
    data_latest_date = max(max_date_from_csv(paths["stock_daily_all"]), max_date_from_csv(paths["market_daily"]))

    for path in [SCORES_PATH, VALIDATION_SUMMARY_PATH, CALIBRATION_PATH, DECISION_JSON_PATH]:
        if not path.exists():
            fail(f"missing required main model artifact: {path.relative_to(PROJECT_ROOT)}")

    decision = read_json(DECISION_JSON_PATH)
    selected_gate = decision.get("selected_gate")
    gate = None if selected_gate in ("", None) else float(selected_gate)

    scores = pd.read_csv(SCORES_PATH, encoding="utf-8-sig")
    validation = pd.read_csv(VALIDATION_SUMMARY_PATH, encoding="utf-8-sig")
    calibration = pd.read_csv(CALIBRATION_PATH, encoding="utf-8-sig")
    feature_frame, feature_cols = build_training_frame(config)
    feature_frame["split"] = split_name(feature_frame["日期"], feature_frame["label_complete"], config)
    feature_frame = feature_frame[
        (feature_frame["label_complete"]) & (feature_frame["has_full_20d_history"] == 1)
    ].copy()

    require_columns(
        scores,
        [
            "日期",
            "股票代號",
            "股票名稱",
            "主分類",
            "split",
            "integrated_research_score",
            "success_advantage_head",
            "same_day_advantage_head",
            "risk_head",
            "episode_head",
            "target_success",
            "selection_success_label",
            "same_day_advantage_label",
            "same_day_advantage_target",
            "same_day_return_percentile",
            "risk_label",
            "relative_top20_label",
            "episode_start_label",
            "future_10d_high_close_return",
        ],
        "main_model_scores.csv",
    )
    scores["日期"] = pd.to_datetime(scores["日期"])
    scores["股票代號"] = scores["股票代號"].astype(str)
    scores = scores.sort_values(["股票代號", "日期"]).reset_index(drop=True)
    scores["diagnostic_trading_index"] = scores.groupby("股票代號", sort=False).cumcount()

    completed = scores[scores["split"].isin(["development", "holdout"])].copy()
    development = completed[completed["split"].eq("development")].copy()
    holdout = completed[completed["split"].eq("holdout")].copy()
    if development.empty or holdout.empty:
        fail("development and holdout scores are required for diagnosis")

    dev_picks = select_top3(development, gate)
    holdout_picks = select_top3(holdout, gate)
    dev_summary = summary_for_picks(development, dev_picks)
    holdout_summary = summary_for_picks(holdout, holdout_picks)

    diag_rows: list[dict] = []
    same_day_stable_features, same_day_checked_features = add_feature_learnability_rows(
        diag_rows,
        feature_frame,
        feature_cols,
        "same_day_return_percentile",
        "same_day_return_percentile",
    )
    future_return_stable_features, future_return_checked_features = add_feature_learnability_rows(
        diag_rows,
        feature_frame,
        feature_cols,
        "future_10d_high_close_return",
        "future_10d_high_close_return",
    )

    holdout_row = validation[
        validation["split"].eq("holdout") & validation["strategy"].eq("integrated_main_top3")
    ].iloc[0]
    dev_row = validation[
        validation["split"].eq("development") & validation["strategy"].eq("integrated_main_top3")
    ].iloc[0]
    holdout_probe_row = validation[
        validation["split"].eq("holdout") & validation["strategy"].eq("return_ranking_probe_top3")
    ].iloc[0]
    dev_probe_row = validation[
        validation["split"].eq("development") & validation["strategy"].eq("return_ranking_probe_top3")
    ].iloc[0]

    holdout_lift = float(holdout_row["success_lift"])
    return_lift = float(holdout_row["return_lift"])
    add_row(
        diag_rows,
        "holdout_vs_market",
        "selected_top3_success_lift",
        "holdout",
        "success_lift",
        holdout_lift,
        "> 0",
        "fail" if holdout_lift <= 0 else "pass",
        "holdout selected candidates did not beat same-day market baseline",
        "repair_return_ranking_features",
    )
    add_row(
        diag_rows,
        "return_ranking_probe",
        "probe_top3_return_lift",
        "holdout",
        "return_lift",
        float(holdout_probe_row["return_lift"]),
        "> 0",
        "pass" if float(holdout_probe_row["return_lift"]) > 0 else "fail",
        "same_day_advantage_head standalone Top3 return lift",
        "redefine_return_target",
        extra={"development_value": float(dev_probe_row["return_lift"])},
    )
    add_row(
        diag_rows,
        "return_ranking_probe",
        "probe_top3_success_lift",
        "holdout",
        "success_lift",
        float(holdout_probe_row["success_lift"]),
        "> 0",
        "pass" if float(holdout_probe_row["success_lift"]) > 0 else "fail",
        "same_day_advantage_head standalone Top3 success lift",
        "redefine_return_target",
        extra={"development_value": float(dev_probe_row["success_lift"])},
    )
    add_row(
        diag_rows,
        "holdout_vs_market",
        "selected_top3_return_lift",
        "holdout",
        "return_lift",
        return_lift,
        "> 0",
        "pass" if return_lift > 0 else "fail",
        "return lift is near flat even though success lift is negative",
        "repair_return_ranking_features",
    )

    dev_lift = float(dev_row["success_lift"])
    drift = holdout_lift - dev_lift
    add_row(
        diag_rows,
        "dev_holdout_drift",
        "development_to_holdout_success_lift_change",
        "development_to_holdout",
        "success_lift_delta",
        drift,
        "near 0 or positive",
        "fail" if drift < -0.05 else "pass",
        "development looked strong but holdout lost same-day advantage",
        "repair_return_ranking_features",
    )
    add_row(
        diag_rows,
        "dev_holdout_drift",
        "selected_day_market_baseline_jump",
        "development_to_holdout",
        "same_day_baseline_delta",
        float(holdout_row["same_day_baseline_success_rate"]) - float(dev_row["same_day_baseline_success_rate"]),
        "stable baseline",
        "warn",
        "selected holdout days were much stronger market days, but model still did not add selection value",
        "repair_score_weighting",
    )

    integrated_bands = calibration[
        calibration["band_type"].eq("integrated_score_success")
    ].sort_values("band")
    integrated_delta = high_low_delta(integrated_bands)
    add_row(
        diag_rows,
        "score_band_ordering",
        "integrated_score_high_minus_low_success",
        "holdout",
        "actual_rate_delta",
        integrated_delta,
        "> 0",
        status_from_delta(integrated_delta),
        "higher integrated score band is not more successful than the low band",
        "repair_return_ranking_features",
    )

    success_bands = band_table(holdout, "success_advantage_head", "target_success")
    selection_bands = band_table(holdout, "success_advantage_head", "selection_success_label")
    advantage_bands = band_table(holdout, "same_day_advantage_head", "same_day_advantage_label")
    advantage_target_bands = band_table(holdout, "same_day_advantage_head", "same_day_advantage_target")
    legacy_top20_bands = band_table(holdout, "same_day_advantage_head", "relative_top20_label")
    episode_bands = band_table(holdout, "episode_head", "episode_start_label")
    risk_bands = band_table(holdout, "risk_head", "risk_label")

    head_specs = [
        ("success_advantage_head", "target_success", success_bands, "success_advantage_head_high_minus_low_success", "repair_return_ranking_features"),
        ("success_advantage_head", "selection_success_label", selection_bands, "success_advantage_head_high_minus_low_selection", "repair_return_ranking_features"),
        ("same_day_advantage_head", "same_day_advantage_label", advantage_bands, "same_day_advantage_head_high_minus_low_advantage", "repair_return_ranking_features"),
        ("same_day_advantage_head", "same_day_advantage_target", advantage_target_bands, "same_day_advantage_head_high_minus_low_target", "repair_return_ranking_features"),
        ("same_day_advantage_head", "relative_top20_label", legacy_top20_bands, "same_day_advantage_head_high_minus_low_legacy_top20", "repair_return_ranking_features"),
        ("episode_head", "episode_start_label", episode_bands, "episode_head_high_minus_low_episode_start", "repair_return_ranking_features"),
        ("risk_head", "risk_label", risk_bands, "risk_head_high_minus_low_failure", "repair_score_weighting"),
    ]
    for head_name, target_name, table, check_name, repair_signal in head_specs:
        delta = high_low_delta(table)
        status = status_from_delta(delta)
        finding = f"{head_name} vs {target_name} high-low separation"
        add_row(
            diag_rows,
            "head_diagnostics",
            check_name,
            "holdout",
            "actual_rate_delta",
            delta,
            "> 0",
            status,
            finding,
            repair_signal,
        )

    dev_advantage_target_delta = high_low_delta(
        band_table(development, "same_day_advantage_head", "same_day_advantage_target")
    )
    holdout_advantage_target_delta = high_low_delta(
        band_table(holdout, "same_day_advantage_head", "same_day_advantage_target")
    )
    dev_advantage_label_delta = high_low_delta(
        band_table(development, "same_day_advantage_head", "same_day_advantage_label")
    )
    holdout_advantage_label_delta = high_low_delta(
        band_table(holdout, "same_day_advantage_head", "same_day_advantage_label")
    )
    generalization_status = "pass"
    if dev_advantage_target_delta > 0 and holdout_advantage_target_delta <= 0:
        generalization_status = "fail"
    elif dev_advantage_target_delta <= 0 and holdout_advantage_target_delta <= 0:
        generalization_status = "fail"
    add_row(
        diag_rows,
        "head_generalization",
        "same_day_advantage_head_dev_to_holdout_target_delta",
        "development_to_holdout",
        "holdout_minus_development_delta",
        holdout_advantage_target_delta - dev_advantage_target_delta,
        f"development_delta={dev_advantage_target_delta:.6f}; holdout_delta={holdout_advantage_target_delta:.6f}",
        generalization_status,
        "same_day_advantage_head does not generalize to holdout return percentile ordering",
        "repair_return_ranking_features",
        extra={
            "target_name": "same_day_advantage_target",
            "development_value": dev_advantage_target_delta,
            "holdout_value": holdout_advantage_target_delta,
        },
    )
    add_row(
        diag_rows,
        "head_generalization",
        "same_day_advantage_head_dev_to_holdout_label_delta",
        "development_to_holdout",
        "holdout_minus_development_delta",
        holdout_advantage_label_delta - dev_advantage_label_delta,
        f"development_delta={dev_advantage_label_delta:.6f}; holdout_delta={holdout_advantage_label_delta:.6f}",
        "pass" if holdout_advantage_label_delta > 0 else "fail",
        "same_day_advantage_head does not separate top 30% same-day return names in holdout",
        "repair_return_ranking_features",
        extra={
            "target_name": "same_day_advantage_label",
            "development_value": dev_advantage_label_delta,
            "holdout_value": holdout_advantage_label_delta,
        },
    )

    selected_weights = decision.get("selected_weights") or [math.nan, math.nan, math.nan, math.nan]
    success_w, advantage_w, episode_w, risk_w = [float(v) for v in selected_weights]
    contribution_values = {
        "success_advantage_head": abs(success_w) * float(holdout["success_advantage_head"].std()),
        "same_day_advantage_head": abs(advantage_w) * float(holdout["same_day_advantage_head"].std()),
        "episode_head": abs(episode_w) * float(holdout["episode_head"].std()),
        "risk_head": abs(risk_w) * float(holdout["risk_head"].std()),
    }
    contribution_total = sum(v for v in contribution_values.values() if not math.isnan(v))
    advantage_contribution_share = (
        contribution_values["same_day_advantage_head"] / contribution_total if contribution_total else math.nan
    )
    integrated_advantage_corr = safe_corr(holdout, "integrated_research_score", "same_day_advantage_head")
    raw_advantage_summary = raw_score_top3_summary(holdout, "same_day_advantage_head")
    selected_weight_stability_passed = bool(decision.get("selected_weight_stability_passed", False))
    dev_monthly_positive = int(decision.get("development_monthly_positive_months") or 0)
    dev_monthly_total = int(decision.get("development_monthly_total_months") or 0)
    dev_min_monthly_success_lift = (
        math.nan if decision.get("development_min_monthly_success_lift") is None else float(decision.get("development_min_monthly_success_lift"))
    )
    dev_min_monthly_return_lift = (
        math.nan if decision.get("development_min_monthly_return_lift") is None else float(decision.get("development_min_monthly_return_lift"))
    )
    selected_objective_score = (
        math.nan if decision.get("selected_weight_objective_score") is None else float(decision.get("selected_weight_objective_score"))
    )
    add_row(
        diag_rows,
        "score_weighting",
        "same_day_advantage_contribution_share",
        "holdout",
        "weighted_std_share",
        advantage_contribution_share,
        ">= 0.20 preferred",
        "warn" if advantage_contribution_share < 0.20 else "pass",
        "same-day advantage head contribution inside integrated score",
        "repair_score_weighting",
        extra={
            "selected_weights": ",".join(str(v) for v in selected_weights),
            "holdout_value": advantage_contribution_share,
        },
    )
    add_row(
        diag_rows,
        "score_weighting",
        "integrated_score_to_advantage_head_corr",
        "holdout",
        "correlation",
        integrated_advantage_corr,
        "> 0",
        "pass" if integrated_advantage_corr > 0 else "fail",
        "integrated score relationship to same-day advantage head",
        "repair_score_weighting",
    )
    add_row(
        diag_rows,
        "score_weighting",
        "raw_same_day_advantage_top3_return_lift",
        "holdout",
        "return_lift",
        raw_advantage_summary["return_lift"],
        f"integrated_return_lift={return_lift:.6f}",
        "pass" if raw_advantage_summary["return_lift"] > return_lift and raw_advantage_summary["return_lift"] > 0 else "fail",
        "raw same-day advantage head Top3 return lift compared with integrated score Top3",
        "repair_score_weighting",
    )
    monthly_stability_ratio = dev_monthly_positive / dev_monthly_total if dev_monthly_total else math.nan
    add_row(
        diag_rows,
        "score_weighting",
        "selected_weight_development_monthly_stability",
        "development",
        "positive_month_share",
        monthly_stability_ratio,
        ">= 0.60 and at least two months",
        "pass" if selected_weight_stability_passed else "fail",
        "selected score weights must work across most development months, not only one month",
        "repair_score_weighting",
        extra={
            "positive_months": dev_monthly_positive,
            "total_months": dev_monthly_total,
            "selected_objective_score": selected_objective_score,
        },
    )
    add_row(
        diag_rows,
        "score_weighting",
        "selected_weight_min_monthly_success_lift",
        "development",
        "success_lift",
        dev_min_monthly_success_lift,
        "reported for stability audit",
        "warn" if dev_min_monthly_success_lift <= 0 else "pass",
        "worst selected-weight development month for success lift",
        "repair_score_weighting",
    )
    add_row(
        diag_rows,
        "score_weighting",
        "selected_weight_min_monthly_return_lift",
        "development",
        "return_lift",
        dev_min_monthly_return_lift,
        "reported for stability audit",
        "warn" if dev_min_monthly_return_lift <= 0 else "pass",
        "worst selected-weight development month for return lift",
        "repair_score_weighting",
    )

    active_months = int(holdout_picks["日期"].dt.strftime("%Y-%m").nunique()) if not holdout_picks.empty else 0
    top_stock_share = float(holdout_picks["股票代號"].value_counts(normalize=True).iloc[0]) if not holdout_picks.empty else math.nan
    top_industry_share = float(holdout_picks["主分類"].fillna("unknown").value_counts(normalize=True).iloc[0]) if not holdout_picks.empty else math.nan
    add_row(
        diag_rows,
        "concentration",
        "active_holdout_months",
        "holdout",
        "months",
        active_months,
        ">= 3 preferred",
        "warn" if active_months < 3 else "pass",
        "holdout picks are spread across too few active months to be considered stable",
        "repair_score_weighting",
    )
    add_row(
        diag_rows,
        "concentration",
        "top_stock_share",
        "holdout",
        "share",
        top_stock_share,
        "<= 0.20",
        "pass" if top_stock_share <= 0.20 else "fail",
        "single stock concentration is acceptable",
        "repair_return_ranking_features",
    )
    add_row(
        diag_rows,
        "concentration",
        "top_industry_share",
        "holdout",
        "share",
        top_industry_share,
        "<= 0.50",
        "pass" if top_industry_share <= 0.50 else "fail",
        "single industry concentration is acceptable",
        "repair_return_ranking_features",
    )

    daily_lift = daily_pick_vs_baseline(holdout, holdout_picks)
    if not daily_lift.empty:
        daily_lift["month"] = daily_lift["日期"].dt.strftime("%Y-%m")
        monthly_return_lift = daily_lift.groupby("month")["return_lift"].mean()
        negative_months = int((monthly_return_lift < 0).sum())
        worst_month = str(monthly_return_lift.idxmin())
        worst_month_lift = float(monthly_return_lift.min())
    else:
        negative_months = 0
        worst_month = ""
        worst_month_lift = math.nan
    add_row(
        diag_rows,
        "return_failure_concentration",
        "negative_return_lift_month_count",
        "holdout",
        "months",
        negative_months,
        "0 preferred",
        "pass" if negative_months == 0 else "fail",
        "return lift failure concentration across holdout months",
        "repair_return_ranking_features",
        extra={"worst_month": worst_month, "worst_month_lift": worst_month_lift},
    )
    if not holdout_picks.empty:
        industry_returns = (
            holdout_picks.groupby("主分類")["future_10d_high_close_return"]
            .agg(["size", "mean"])
            .sort_values(["size", "mean"], ascending=[False, True])
        )
        top_industry_name = str(industry_returns.index[0])
        top_industry_avg_return = float(industry_returns.iloc[0]["mean"])
        top_industry_count = int(industry_returns.iloc[0]["size"])
    else:
        top_industry_name = ""
        top_industry_avg_return = math.nan
        top_industry_count = 0
    add_row(
        diag_rows,
        "return_failure_concentration",
        "largest_pick_industry_avg_return",
        "holdout",
        "avg_10d_high_close_return",
        top_industry_avg_return,
        f"rows={top_industry_count}",
        "warn",
        "largest selected industry return profile",
        "repair_return_ranking_features",
        extra={"industry_name": top_industry_name, "industry_row_count": top_industry_count},
    )

    score_distribution = []
    for split_label, split_df in [("development", development), ("holdout", holdout)]:
        score_distribution.append(
            {
                "split": split_label,
                "score_mean": float(split_df["integrated_research_score"].mean()),
                "score_std": float(split_df["integrated_research_score"].std()),
                "day_max_mean": float(split_df.groupby("日期")["integrated_research_score"].max().mean()),
                "market_success_mean": float(split_df.groupby("日期")["target_success"].mean().mean()),
                "market_return_mean": float(split_df.groupby("日期")["future_10d_high_close_return"].mean().mean()),
            }
        )
    dev_dist, holdout_dist = score_distribution
    score_mean_delta = holdout_dist["score_mean"] - dev_dist["score_mean"]
    add_row(
        diag_rows,
        "dev_holdout_drift",
        "model_output_distribution_shift",
        "development_to_holdout",
        "score_mean_delta",
        score_mean_delta,
        "small absolute shift",
        "warn" if abs(score_mean_delta) > 0.02 else "pass",
        "model output distribution shifted between development and holdout",
        "repair_return_ranking_features",
    )

    diagnosis = pd.DataFrame(diag_rows)
    diagnosis.to_csv(DIAGNOSIS_CSV_PATH, index=False, encoding="utf-8-sig")

    success_advantage_delta = float(
        diagnosis.loc[
            diagnosis["check_name"].eq("success_advantage_head_high_minus_low_success"),
            "value",
        ].iloc[0]
    )
    advantage_head_delta = float(
        diagnosis.loc[
            diagnosis["check_name"].eq("same_day_advantage_head_high_minus_low_advantage"),
            "value",
        ].iloc[0]
    )
    advantage_target_delta = float(
        diagnosis.loc[
            diagnosis["check_name"].eq("same_day_advantage_head_high_minus_low_target"),
            "value",
        ].iloc[0]
    )
    risk_head_delta = float(
        diagnosis.loc[
            diagnosis["check_name"].eq("risk_head_high_minus_low_failure"),
            "value",
        ].iloc[0]
    )

    raw_advantage_return_lift = float(raw_advantage_summary["return_lift"])
    target_contract = str(decision.get("target_contract", ""))
    formal_approved = bool(decision.get("formal_approved", False))
    if formal_approved:
        recommended_repair_id = "ready_for_formal_review"
        recommendation_summary = "回撤旁支標籤主模型已通過訓練驗證；下一步只能由正式入口決定是否更新候選。"
    elif advantage_target_delta <= 0 or raw_advantage_return_lift <= 0:
        recommended_repair_id = "redefine_return_target"
        recommendation_summary = (
            "return-ranking probe 仍未通過；同日報酬排序目標目前無法由現有特徵穩定學出來。"
        )
    elif not selected_weight_stability_passed:
        recommended_repair_id = "repair_score_weighting"
        recommendation_summary = "先修整合分數權重；目前選出的權重沒有通過 development 月度穩定檢查。"
    elif holdout_lift <= 0 and return_lift > 0:
        recommended_repair_id = "redefine_return_target"
        recommendation_summary = "權重已通過 development 月度穩定，且報酬 lift 為正，但成功率仍輸同日市場；下一步不該再調權重，應檢討正式交易目標。"
    elif raw_advantage_return_lift > 0 and return_lift <= 0:
        recommended_repair_id = "redefine_return_target"
        recommendation_summary = "權重修正後仍無法同時保留成功率與報酬優勢；下一步應檢討正式交易目標，而不是繼續補丁式調權重。"
    else:
        recommended_repair_id = "repair_score_weighting"
        recommendation_summary = "return-ranking probe 已有線索，但仍未完整通過正式驗證；若要再前進，應檢討交易目標或正式通過條件，而不是新增平行模型。"

    market_relation = "above" if float(holdout_row["success_rate"]) >= float(holdout_row["same_day_baseline_success_rate"]) else "below"
    if holdout_lift > 0 and return_lift > 0:
        plain_result_line = "- holdout 同時有成功率優勢與報酬優勢，但仍需確認 head 方向、風險與集中度是否過關。"
    elif holdout_lift <= 0 and return_lift > 0:
        plain_result_line = "- holdout 報酬優勢已轉正，但成功率輸同日市場，代表報酬排序有線索，整合分數仍不夠平衡。"
    elif holdout_lift > 0 and return_lift <= 0:
        plain_result_line = "- holdout 成功率有優勢，但平均 10 日高收報酬仍輸同日市場，代表排序品質還不乾淨。"
    else:
        plain_result_line = "- holdout 成功率與報酬優勢都沒有通過，主模型不能升正式。"
    if advantage_target_delta > 0 and raw_advantage_return_lift > 0:
        head_result_line = "- 同日報酬排序 head 已轉正，問題已從「head 學反」收斂成「整合分數如何同時保留成功率與報酬優勢」。"
    else:
        head_result_line = "- 同日報酬排序 head 在 holdout 仍未站穩，代表目前報酬排序目標仍無法穩定泛化。"
    evidence = [
        f"Target contract: {target_contract or 'legacy_target_success'}.",
        f"Holdout success rate {format_pct(float(holdout_row['success_rate']))}, {market_relation} same-day market baseline {format_pct(float(holdout_row['same_day_baseline_success_rate']))}.",
        f"Holdout success lift {format_pct(holdout_lift)}; development success lift {format_pct(dev_lift)}.",
        f"Integrated score high-low success delta {format_pct(integrated_delta)}.",
        f"Success advantage head high-low success delta {format_pct(success_advantage_delta)}.",
        f"Same-day advantage head high-low advantage delta {format_pct(advantage_head_delta)}.",
        f"Same-day advantage head high-low soft target delta {format_pct(advantage_target_delta)}.",
        f"Stable same-day return ranking feature count {same_day_stable_features}/{same_day_checked_features}.",
        f"Raw same-day advantage Top3 return lift {format_pct(raw_advantage_return_lift)}.",
        f"Return-ranking probe holdout success lift {format_pct(float(holdout_probe_row['success_lift']))}.",
        f"Same-day advantage contribution share inside integrated score {format_pct(advantage_contribution_share)}.",
        f"Selected weight development monthly stability {dev_monthly_positive}/{dev_monthly_total}; objective score {selected_objective_score:.6f}.",
        f"Risk head high-low failure delta {format_pct(risk_head_delta)}; risk separation is not the primary blocker.",
    ]

    formal_hashes = {
        "formal_status_sha256": sha256(FORMAL_STATUS_PATH),
        "formal_candidates_sha256": sha256(FORMAL_CANDIDATES_PATH),
    }
    formal_candidate_rows = formal_candidate_row_count()
    if formal_approved and formal_candidate_rows > 0:
        formal_reason = "formal entrypoint has promoted current validated candidates"
    elif formal_approved:
        formal_reason = "formal entrypoint may promote candidates when the latest date passes the selected score gate"
    else:
        formal_reason = "main model failed same-day success lift or formal ordering checks"

    recommendation = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_latest_date": data_latest_date,
        "diagnosis_status": "completed",
        "main_model_status": decision.get("status"),
        "recommended_repair_id": recommended_repair_id,
        "alternative_repair_ids": [],
        "recommendation_summary": recommendation_summary,
        "formal_outputs_unchanged": True,
        "formal_hashes": formal_hashes,
        "evidence": evidence,
        "blocked_from_formal_reason": formal_reason,
        "root_cause_summary": {
            "stable_same_day_return_features": same_day_stable_features,
            "checked_same_day_return_features": same_day_checked_features,
            "stable_future_return_features": future_return_stable_features,
            "checked_future_return_features": future_return_checked_features,
            "dev_advantage_target_delta": dev_advantage_target_delta,
            "holdout_advantage_target_delta": holdout_advantage_target_delta,
            "raw_advantage_return_lift": raw_advantage_return_lift,
            "integrated_return_lift": return_lift,
            "advantage_contribution_share": advantage_contribution_share,
            "selected_weight_stability_passed": selected_weight_stability_passed,
            "development_monthly_positive_months": dev_monthly_positive,
            "development_monthly_total_months": dev_monthly_total,
            "development_min_monthly_success_lift": dev_min_monthly_success_lift,
            "development_min_monthly_return_lift": dev_min_monthly_return_lift,
            "selected_weight_objective_score": selected_objective_score,
            "negative_return_lift_months": negative_months,
            "formal_candidate_rows": formal_candidate_rows,
        },
    }
    RECOMMENDATION_PATH.write_text(
        json.dumps(recommendation, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    md_lines = [
        "# Main Model Failure Diagnosis",
        "",
        f"- Generated: {recommendation['generated_at']}",
        f"- Data latest date: {data_latest_date}",
        f"- Main model status: `{decision.get('status')}`",
        "- Formal output: unchanged",
        "",
        "## 結論",
        "",
        (
            "主模型已通過訓練驗證；但正式輸出仍只能由 `scripts/run_main_pipeline.py` 決定。"
            if formal_approved
            else "主模型沒有升正式，原因不是訓練沒有跑，而是 holdout 沒有同時通過成功率優勢、風險排序與正式驗證檢查。"
        ),
        "",
        f"唯一建議: `{recommended_repair_id}`",
        "",
        recommendation_summary,
        "",
        "## 核心證據",
        "",
    ]
    for item in evidence:
        md_lines.append(f"- {item}")
    md_lines.extend(
        [
            "",
            "## 白話解讀",
            "",
            "- 目前模型會收斂，但收斂到的不是可正式使用的選股排序。",
            plain_result_line,
            head_result_line,
            "",
            "## 同日報酬排序根因",
            "",
            f"- 穩定的同日報酬排序特徵數: {same_day_stable_features}/{same_day_checked_features}。",
            f"- same_day_advantage_head 在 development 的 soft target 高低差: {format_pct(dev_advantage_target_delta)}。",
            f"- same_day_advantage_head 在 holdout 的 soft target 高低差: {format_pct(holdout_advantage_target_delta)}。",
            f"- 單看 same_day_advantage_head 的 Top3 return lift: {format_pct(raw_advantage_return_lift)}。",
            f"- 整合分數中的 same_day_advantage 權重貢獻占比: {format_pct(advantage_contribution_share)}。",
            f"- holdout 負報酬優勢月份數: {negative_months}。",
            "",
            f"根因判定: `{recommended_repair_id}`。{recommendation_summary}",
            "",
            "## 整合分數權重根因",
            "",
            f"- development 月度穩定: {dev_monthly_positive}/{dev_monthly_total} 個出手月份同時通過 success lift 與 return lift。",
            f"- development 最差月 success lift: {format_pct(dev_min_monthly_success_lift)}。",
            f"- development 最差月 return lift: {format_pct(dev_min_monthly_return_lift)}。",
            f"- 選定權重的 balanced objective score: {selected_objective_score:.6f}。",
            f"- 權重穩定檢查通過: {selected_weight_stability_passed}。",
            "",
            "## 輸出檔案",
            "",
            f"- `{DIAGNOSIS_CSV_PATH.relative_to(PROJECT_ROOT)}`",
            f"- `{RECOMMENDATION_PATH.relative_to(PROJECT_ROOT)}`",
            "",
        ]
    )
    DIAGNOSIS_MD_PATH.write_text("\n".join(md_lines), encoding="utf-8")

    print("OK: main model failure diagnosis completed")
    print(f"DIAGNOSIS: {DIAGNOSIS_MD_PATH}")
    print(f"RECOMMENDATION: {RECOMMENDATION_PATH}")


if __name__ == "__main__":
    main()
