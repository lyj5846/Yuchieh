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
FEATURE_SCREEN_PATH = PROJECT_ROOT / "validation_layer" / "main_model_feature_screen.csv"

REQUIRED_COLUMNS = {
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
    if not FEATURE_SCREEN_PATH.exists():
        fail(f"missing feature screen table: {FEATURE_SCREEN_PATH}")

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
        completed["target_success"].astype(int) != completed["old_target_success"].astype(int)
    ]
    if not target_mismatch.empty:
        fail("target_success must equal old +3% touch target")
    impossible_hard_success = completed[
        (completed["risk_adjusted_10d_success"].astype(int) == 1)
        & (completed["target_success"].astype(int) != 1)
    ]
    if not impossible_hard_success.empty:
        fail("hard risk-adjusted success cannot be 1 when primary target_success is 0")
    expected_old_success_failed = (
        (completed["old_target_success"].astype(int) == 1)
        & (completed["risk_adjusted_10d_success"].astype(int) == 0)
    ).astype(int)
    old_success_failed_mismatch = completed[
        completed["old_success_but_risk_failed"].astype(int) != expected_old_success_failed
    ]
    if not old_success_failed_mismatch.empty:
        fail("old_success_but_risk_failed must equal old_target_success=1 and hard risk-adjusted success=0")
    same_day_tie_hard_success = completed[
        (completed["same_day_both_event"].astype(int) == 1)
        & (completed["risk_adjusted_10d_success"].astype(int) == 1)
    ]
    if not same_day_tie_hard_success.empty:
        fail("same-day profit/adverse tie must fail only the hard risk-adjusted comparison")
    same_day_tie_primary_failure = completed[
        (completed["same_day_both_event"].astype(int) == 1)
        & (completed["target_success"].astype(int) != 1)
    ]
    if not same_day_tie_primary_failure.empty:
        fail("same-day +3% and -3% must remain primary target success under side-label contract")
    expected_clean = (
        (completed["target_success"].astype(int) == 1)
        & (completed["drawdown_minus3_before_or_same_success"].astype(int) == 0)
    ).astype(int)
    expected_painful = (
        (completed["target_success"].astype(int) == 1)
        & (completed["drawdown_minus3_before_or_same_success"].astype(int) == 1)
    ).astype(int)
    if not completed["clean_success_label"].astype(int).eq(expected_clean).all():
        fail("clean_success_label must mark target successes without prior/same-day -3% drawdown")
    if not completed["painful_success_label"].astype(int).eq(expected_painful).all():
        fail("painful_success_label must mark target successes with prior/same-day -3% drawdown")

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
        "Formal target_success is the 10-day +3% close touch rule",
        "Drawdown side labels",
        "risk_adjusted_10d_success is retained only as a hard-risk comparison field",
    ]:
        if phrase not in spec_text:
            fail(f"training spec missing drawdown side-label target phrase: {phrase}")
    for phrase in [
        "Formal target_success is risk-adjusted_10d_success",
        "+3% close must occur before any -3% low within 10 trading days",
        "Conservative tie rule",
        "Old +3% touch target is retained as old_target_success for comparison only",
    ]:
        if phrase in spec_text:
            fail(f"training spec must not keep old hard-risk target wording: {phrase}")
    for phrase in [
        "Feature screen: selected",
        "Feature screen uses train/development correlation stability only",
        "same-day relative return-ranking features",
        "same_day_advantage loss weight",
    ]:
        if phrase not in spec_text:
            fail(f"training spec missing phrase: {phrase}")

    import json

    decision = json.loads(DECISION_JSON_PATH.read_text(encoding="utf-8"))
    if decision.get("confirmed_plan") != "drawdown_side_label_main_model_training_plan":
        fail("decision json must come from drawdown_side_label_main_model_training_plan")
    if decision.get("target_contract") != "drawdown_side_label_10d_touch_success":
        fail("decision json must record drawdown_side_label_10d_touch_success target contract")
    for key in [
        "holdout_old_target_success_rate",
        "holdout_primary_touch_success_rate",
        "holdout_risk_adjusted_success_rate",
        "holdout_old_success_but_risk_failed_rate",
        "holdout_old_success_but_risk_failed_count",
        "holdout_old_success_but_risk_failed_among_old_success",
        "holdout_clean_success_rate",
        "holdout_painful_success_rate",
        "holdout_painful_success_among_success",
        "holdout_minus3_anytime_rate",
        "holdout_return_ranking_probe_return_lift",
        "holdout_return_ranking_probe_success_lift",
        "return_ranking_probe_order_ok",
        "score_band_ordering_required_for_promotion",
        "candidate_region_validation_ok",
        "same_day_advantage_loss_weight",
        "development_monthly_positive_months",
        "development_monthly_total_months",
        "development_min_monthly_success_lift",
        "development_min_monthly_return_lift",
        "selected_weight_stability_passed",
        "selected_weight_score_band_passed",
        "development_score_band_success_delta",
        "development_score_band_advantage_delta",
        "development_score_band_return_delta",
        "selected_weight_objective_score",
        "feature_screen_enabled",
        "feature_screen_source",
        "feature_screen_output",
        "feature_screen_uses_holdout_for_selection",
        "original_feature_count",
        "selected_feature_count",
        "selected_feature_preview",
    ]:
        if key not in decision:
            fail(f"decision json missing required field: {key}")
    if decision["feature_screen_enabled"] is not True:
        fail("decision json must record enabled feature screening")
    if decision["feature_screen_uses_holdout_for_selection"] is not False:
        fail("feature screen must not use holdout for selection")
    if decision["selected_feature_count"] <= 0 or decision["selected_feature_count"] > decision["original_feature_count"]:
        fail("selected feature count must be positive and not exceed original feature count")

    if decision["development_monthly_total_months"] <= 0:
        fail("decision json must record at least one development month")
    if decision["development_monthly_positive_months"] > decision["development_monthly_total_months"]:
        fail("development positive months cannot exceed total months")
    if decision["selected_weight_stability_passed"] is not True:
        fail("selected strategy must pass development monthly stability before holdout validation")
    if decision["selected_weight_score_band_passed"] is not True:
        fail("selected strategy must pass development score-band ordering before holdout validation")
    if decision["score_band_ordering_required_for_promotion"] is not False:
        fail("holdout all-row score-band ordering must be diagnostic, not a formal promotion blocker")
    if decision["status"] == "passed_holdout_validation" and decision["candidate_region_validation_ok"] is not True:
        fail("passed model must have candidate-region validation passed")

    tuning = pd.read_csv(STRATEGY_TUNING_PATH, encoding="utf-8-sig")
    required_tuning_columns = {
        "monthly_positive_months",
        "monthly_total_months",
        "min_monthly_success_lift",
        "min_monthly_return_lift",
        "monthly_stability_passed",
        "score_band_success_delta",
        "score_band_advantage_delta",
        "score_band_return_delta",
        "score_band_order_passed",
        "balanced_objective_score",
    }
    missing_tuning = sorted(required_tuning_columns - set(tuning.columns))
    if missing_tuning:
        fail("main_model_strategy_tuning.csv missing columns: " + ", ".join(missing_tuning))
    if tuning["balanced_objective_score"].isna().all():
        fail("strategy tuning table must include at least one objective score")

    feature_screen = pd.read_csv(FEATURE_SCREEN_PATH, encoding="utf-8-sig")
    required_feature_screen_cols = {
        "feature",
        "selected",
        "screening_score",
        "stable_metric_count",
        "stable_success",
        "stable_return",
        "stable_risk_filter",
        "used_holdout_for_selection",
    }
    missing_feature_screen = sorted(required_feature_screen_cols - set(feature_screen.columns))
    if missing_feature_screen:
        fail("main_model_feature_screen.csv missing columns: " + ", ".join(missing_feature_screen))
    selected = feature_screen[feature_screen["selected"].astype(str).str.lower().isin(["true", "1"])]
    if selected.empty:
        fail("feature screen must select at least one feature")
    if len(selected) != int(decision["selected_feature_count"]):
        fail("decision selected_feature_count must match feature screen selected rows")
    holdout_used = selected["used_holdout_for_selection"].astype(str).str.lower().isin(["true", "1"]).any()
    if holdout_used:
        fail("selected features must not use holdout for selection")
    if selected["stable_metric_count"].astype(float).le(0).any():
        fail("selected features must have at least one stable train/development metric")

    print("OK: main model label contract passed")
    print(f"SCORES: {SCORES_PATH}")


if __name__ == "__main__":
    main()
