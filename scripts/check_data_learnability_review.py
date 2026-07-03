from __future__ import annotations

import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REVIEW_MD_PATH = PROJECT_ROOT / "validation_layer" / "data_learnability_review.md"
FEATURE_SIGNAL_PATH = PROJECT_ROOT / "validation_layer" / "data_learnability_feature_signal.csv"
FAILURE_PROFILE_PATH = PROJECT_ROOT / "validation_layer" / "data_learnability_failure_profile.csv"
DECISION_JSON_PATH = PROJECT_ROOT / "decision_layer" / "data_learnability_decision.json"

ALLOWED_STATUS = {
    "learnable_signal_present",
    "weak_signal_but_not_enough_for_full_retrain",
    "insufficient_signal_in_current_inputs",
}
ALLOWED_NEXT_STEP = {
    "feature_screen_then_retrain",
    "feature_screen_before_any_retrain",
    "review_target_or_add_data",
}


def fail(message: str) -> None:
    raise SystemExit(f"FAIL: {message}")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        fail(f"missing output: {path.relative_to(PROJECT_ROOT)}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    if not REVIEW_MD_PATH.exists():
        fail("missing data_learnability_review.md")
    review_text = REVIEW_MD_PATH.read_text(encoding="utf-8")
    for phrase in [
        "data/label learnability only",
        "Formal output: unchanged by this review.",
        "This is not a probability model.",
        "This does not add a new model branch.",
    ]:
        if phrase not in review_text:
            fail(f"review missing phrase: {phrase}")

    if not DECISION_JSON_PATH.exists():
        fail("missing data_learnability_decision.json")
    decision = json.loads(DECISION_JSON_PATH.read_text(encoding="utf-8"))
    if decision.get("status") not in ALLOWED_STATUS:
        fail("unexpected data learnability status")
    if decision.get("recommended_next_step") not in ALLOWED_NEXT_STEP:
        fail("unexpected data learnability next step")
    if not isinstance(decision.get("stable_success_feature_count"), int):
        fail("decision must include stable_success_feature_count")
    if not isinstance(decision.get("stable_risk_filter_feature_count"), int):
        fail("decision must include stable_risk_filter_feature_count")
    if not isinstance(decision.get("stable_return_feature_count"), int):
        fail("decision must include stable_return_feature_count")
    for key in [
        "holdout_primary_success_rate",
        "holdout_hard_risk_adjusted_success_rate",
        "holdout_drawdown_side_risk_rate",
        "holdout_drawdown_side_risk_among_success",
    ]:
        if key not in decision:
            fail(f"decision missing drawdown side-label field: {key}")

    feature_rows = read_csv_rows(FEATURE_SIGNAL_PATH)
    if not feature_rows:
        fail("data_learnability_feature_signal.csv is empty")
    required_feature_cols = {
        "feature",
        "train_success_corr",
        "development_success_corr",
        "holdout_success_corr",
        "success_corr_direction_stable",
        "risk_filter_corr_direction_stable",
        "return_corr_direction_stable",
    }
    missing_feature_cols = required_feature_cols - set(feature_rows[0])
    if missing_feature_cols:
        fail("feature signal output missing columns: " + ", ".join(sorted(missing_feature_cols)))

    profile_rows = read_csv_rows(FAILURE_PROFILE_PATH)
    if not profile_rows:
        fail("data_learnability_failure_profile.csv is empty")
    required_profile_cols = {
        "section",
        "split",
        "group",
        "rows",
        "primary_success_rate",
        "hard_risk_adjusted_success_rate",
        "clean_success_rate",
        "painful_success_rate",
        "drawdown_side_risk_rate",
        "drawdown_side_risk_among_success",
    }
    missing_profile_cols = required_profile_cols - set(profile_rows[0])
    if missing_profile_cols:
        fail("failure profile output missing columns: " + ", ".join(sorted(missing_profile_cols)))

    overall_splits = {
        row["split"]
        for row in profile_rows
        if row["section"] == "overall"
    }
    if overall_splits != {"train", "development", "holdout"}:
        fail("failure profile must contain overall train/development/holdout rows")

    print("OK: data learnability review contract passed")
    print(f"REPORT: {REVIEW_MD_PATH}")


if __name__ == "__main__":
    main()
