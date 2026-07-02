from __future__ import annotations

import subprocess
import sys
from pathlib import Path

try:
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
SCORES_PATH = PROJECT_ROOT / "model_layer" / "main_model_scores.csv"
TRAINING_SPEC_PATH = PROJECT_ROOT / "model_layer" / "main_model_training_spec.md"
DECISION_JSON_PATH = PROJECT_ROOT / "decision_layer" / "main_model_decision.json"
STRATEGY_TUNING_PATH = PROJECT_ROOT / "validation_layer" / "main_model_strategy_tuning.csv"

REQUIRED_COLUMNS = {
    "target_success",
    "risk_adjusted_10d_success",
    "old_target_success",
    "old_success_but_risk_failed",
    "selection_success_label",
    "same_day_advantage_label",
    "same_day_advantage_target",
    "same_day_return_percentile",
    "relative_top20_label",
    "success_advantage_head",
    "same_day_advantage_head",
    "risk_head",
    "episode_head",
    "future_10d_high_close_return",
    "future_10d_low_close_return",
    "max_adverse_return",
    "profit_event_day",
    "adverse_event_day",
    "same_day_both_event",
    "realized_10d_trade_return",
    "daily_market_avg_return",
}

REQUIRED_RETURN_RANKING_FEATURES = {
    "same_day_return_rank_1",
    "same_day_return_rank_3",
    "same_day_return_rank_5",
    "same_day_return_rank_10",
    "same_day_return_rank_20",
    "industry_return_rank_1",
    "industry_return_rank_3",
    "industry_return_rank_5",
    "industry_return_rank_10",
    "industry_return_rank_20",
    "industry_volume_rank_5",
    "industry_volume_rank_10",
    "industry_volume_rank_20",
    "industry_ma_position_rank_5",
    "industry_ma_position_rank_10",
    "industry_ma_position_rank_20",
    "return_vs_weighted_1",
    "return_vs_weighted_3",
    "return_vs_weighted_5",
    "return_vs_weighted_10",
    "return_vs_weighted_20",
    "return_vs_electronics_1",
    "return_vs_electronics_3",
    "return_vs_electronics_5",
    "return_vs_electronics_10",
    "return_vs_electronics_20",
    "return_vs_otc_1",
    "return_vs_otc_3",
    "return_vs_otc_5",
    "return_vs_otc_10",
    "return_vs_otc_20",
}


def fail(message: str) -> None:
    raise SystemExit(f"FAIL: {message}")


def main() -> None:
    if not SCORES_PATH.exists():
        fail(f"missing model scores: {SCORES_PATH}")
    if not TRAINING_SPEC_PATH.exists():
        fail(f"missing training spec: {TRAINING_SPEC_PATH}")
    if not DECISION_JSON_PATH.exists():
        fail(f"missing decision json: {DECISION_JSON_PATH}")
    if not STRATEGY_TUNING_PATH.exists():
        fail(f"missing strategy tuning table: {STRATEGY_TUNING_PATH}")

    scores = pd.read_csv(SCORES_PATH, encoding="utf-8-sig")
    missing = sorted((REQUIRED_COLUMNS | REQUIRED_RETURN_RANKING_FEATURES) - set(scores.columns))
    if missing:
        fail("main_model_scores.csv missing columns: " + ", ".join(missing))
    if "rank_head" in scores.columns:
        fail("rank_head must be replaced by same_day_advantage_head")

    completed = scores[scores["split"].isin(["train", "development", "holdout"])].copy()
    if completed.empty:
        fail("completed train/development/holdout scores are required")
    target_mismatch = completed[
        completed["target_success"].astype(int) != completed["risk_adjusted_10d_success"].astype(int)
    ]
    if not target_mismatch.empty:
        fail("target_success must equal risk_adjusted_10d_success")
    impossible_new_success = completed[
        (completed["target_success"].astype(int) == 1)
        & (completed["old_target_success"].astype(int) != 1)
    ]
    if not impossible_new_success.empty:
        fail("risk-adjusted target_success cannot be 1 when old_target_success is 0")
    expected_old_success_failed = (
        (completed["old_target_success"].astype(int) == 1)
        & (completed["target_success"].astype(int) == 0)
    ).astype(int)
    old_success_failed_mismatch = completed[
        completed["old_success_but_risk_failed"].astype(int) != expected_old_success_failed
    ]
    if not old_success_failed_mismatch.empty:
        fail("old_success_but_risk_failed must equal old_target_success=1 and target_success=0")
    same_day_tie_success = completed[
        (completed["same_day_both_event"].astype(int) == 1)
        & (completed["target_success"].astype(int) == 1)
    ]
    if not same_day_tie_success.empty:
        fail("same-day profit/adverse tie must be conservative failure")

    invalid_selection = completed[
        (completed["selection_success_label"] == 1)
        & (
            (completed["target_success"] != 1)
            | (completed["same_day_return_percentile"] < 0.50)
        )
    ]
    if not invalid_selection.empty:
        fail("selection_success_label must imply target_success and same-day return percentile >= 0.50")

    expected_advantage = (completed["same_day_return_percentile"] >= 0.70).astype(int)
    mismatch = completed[completed["same_day_advantage_label"].astype(int) != expected_advantage]
    if not mismatch.empty:
        fail("same_day_advantage_label must match same-day return percentile >= 0.70")

    if completed["same_day_return_percentile"].lt(0).any() or completed["same_day_return_percentile"].gt(1).any():
        fail("same_day_return_percentile must be between 0 and 1")
    if completed["same_day_advantage_target"].lt(0).any() or completed["same_day_advantage_target"].gt(1).any():
        fail("same_day_advantage_target must be between 0 and 1")
    target_mismatch = completed[
        (completed["same_day_advantage_target"] - completed["same_day_return_percentile"]).abs()
        > 1e-9
    ]
    if not target_mismatch.empty:
        fail("same_day_advantage_target must equal same_day_return_percentile")

    if int(completed["selection_success_label"].sum()) == 0:
        fail("selection_success_label has no positive samples")
    if int(completed["same_day_advantage_label"].sum()) == 0:
        fail("same_day_advantage_label has no positive samples")
    rank_cols = [
        c
        for c in REQUIRED_RETURN_RANKING_FEATURES
        if c.startswith("same_day_return_rank_")
        or c.startswith("industry_return_rank_")
        or c.startswith("industry_volume_rank_")
        or c.startswith("industry_ma_position_rank_")
    ]
    for col in sorted(rank_cols):
        if completed[col].lt(0).any() or completed[col].gt(1).any():
            fail(f"{col} must be between 0 and 1")

    spec_text = TRAINING_SPEC_PATH.read_text(encoding="utf-8")
    forbidden_phrases = [
        "target_success multiplied by same-day return percentile",
        "target_success * same_day_return_percentile",
        "Formal target_success is unchanged",
    ]
    for phrase in forbidden_phrases:
        if phrase in spec_text:
            fail(f"training spec must not contain old mixed target wording: {phrase}")
    if "pure same-day return percentile" not in spec_text:
        fail("training spec must describe same_day_advantage_target as pure same-day return percentile")
    for phrase in [
        "Formal target_success is risk-adjusted_10d_success",
        "+3% close must occur before any -3% low",
        "old_target_success for comparison only",
    ]:
        if phrase not in spec_text:
            fail(f"training spec missing risk-adjusted target phrase: {phrase}")
    for phrase in [
        "same-day relative return-ranking features",
        "same_day_advantage loss weight",
    ]:
        if phrase not in spec_text:
            fail(f"training spec missing phrase: {phrase}")

    import json

    decision = json.loads(DECISION_JSON_PATH.read_text(encoding="utf-8"))
    if decision.get("confirmed_plan") != "risk_adjusted_main_model_training_plan":
        fail("decision json must come from risk_adjusted_main_model_training_plan")
    if decision.get("target_contract") != "risk_adjusted_10d_success":
        fail("decision json must record risk_adjusted_10d_success target contract")
    for key in [
        "holdout_old_target_success_rate",
        "holdout_risk_adjusted_success_rate",
        "holdout_old_success_but_risk_failed_rate",
        "holdout_old_success_but_risk_failed_count",
        "holdout_old_success_but_risk_failed_among_old_success",
        "holdout_return_ranking_probe_return_lift",
        "holdout_return_ranking_probe_success_lift",
        "return_ranking_probe_order_ok",
        "same_day_advantage_loss_weight",
        "development_monthly_positive_months",
        "development_monthly_total_months",
        "development_min_monthly_success_lift",
        "development_min_monthly_return_lift",
        "selected_weight_stability_passed",
        "selected_weight_objective_score",
    ]:
        if key not in decision:
            fail(f"decision json missing required field: {key}")

    if decision["development_monthly_total_months"] <= 0:
        fail("decision json must record at least one development month")
    if decision["development_monthly_positive_months"] > decision["development_monthly_total_months"]:
        fail("development positive months cannot exceed total months")
    if decision["selected_weight_stability_passed"] is not True:
        fail("selected strategy must pass development monthly stability before holdout validation")

    tuning = pd.read_csv(STRATEGY_TUNING_PATH, encoding="utf-8-sig")
    required_tuning_columns = {
        "monthly_positive_months",
        "monthly_total_months",
        "min_monthly_success_lift",
        "min_monthly_return_lift",
        "monthly_stability_passed",
        "balanced_objective_score",
    }
    missing_tuning = sorted(required_tuning_columns - set(tuning.columns))
    if missing_tuning:
        fail("main_model_strategy_tuning.csv missing columns: " + ", ".join(missing_tuning))
    if tuning["balanced_objective_score"].isna().all():
        fail("strategy tuning table must include at least one objective score")

    print("OK: main model label contract passed")
    print(f"SCORES: {SCORES_PATH}")


if __name__ == "__main__":
    main()
