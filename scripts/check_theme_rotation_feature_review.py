from __future__ import annotations

import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REVIEW_MD_PATH = PROJECT_ROOT / "validation_layer" / "theme_rotation_feature_review.md"
SUMMARY_PATH = PROJECT_ROOT / "validation_layer" / "theme_rotation_feature_summary.csv"
DAILY_PATH = PROJECT_ROOT / "validation_layer" / "theme_rotation_daily_strength.csv"
DECISION_JSON_PATH = PROJECT_ROOT / "decision_layer" / "theme_rotation_feature_decision.json"
RUNNER_PATH = PROJECT_ROOT / "scripts" / "run_theme_rotation_feature_review.py"

ALLOWED_STATUS = {
    "candidate_for_main_model_feature_integration",
    "research_signal_but_not_ready",
    "discard_for_now",
}
ALLOWED_NEXT_STEP = {
    "plan_theme_rotation_feature_contract",
    "keep_research_only",
    "do_not_integrate_theme_rotation_features",
}


def fail(message: str) -> None:
    raise SystemExit(f"FAIL: {message}")


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        fail(f"missing output: {path.relative_to(PROJECT_ROOT)}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        fail(f"{path.relative_to(PROJECT_ROOT)} is empty")
    return rows


def main() -> None:
    if not REVIEW_MD_PATH.exists():
        fail("missing theme_rotation_feature_review.md")
    review = REVIEW_MD_PATH.read_text(encoding="utf-8")
    for phrase in [
        "feature learnability review only",
        "Formal output: unchanged by this review.",
        "This does not choose stocks.",
        "This does not train or promote a model.",
        "This does not create a second decision layer.",
        "Holdout columns are audit-only, not used for feature selection.",
        "research_score is not used and no probability is produced.",
    ]:
        if phrase not in review:
            fail(f"theme rotation review missing phrase: {phrase}")

    if not DECISION_JSON_PATH.exists():
        fail("missing theme_rotation_feature_decision.json")
    decision = json.loads(DECISION_JSON_PATH.read_text(encoding="utf-8"))
    if decision.get("status") not in ALLOWED_STATUS:
        fail("unexpected theme rotation feature review status")
    if decision.get("recommended_next_step") not in ALLOWED_NEXT_STEP:
        fail("unexpected theme rotation feature next step")
    if decision.get("used_holdout_for_selection") is not False:
        fail("theme rotation review must not use holdout for feature selection")
    if decision.get("formal_output_changed") is not False:
        fail("theme rotation review must not change formal output")

    summary_rows = read_rows(SUMMARY_PATH)
    required_summary = {
        "feature",
        "stable_metric_count",
        "screening_score",
        "train_success_corr",
        "development_success_corr",
        "holdout_success_corr",
        "train_return_corr",
        "development_return_corr",
        "holdout_return_corr",
        "train_same_day_rank_corr",
        "development_same_day_rank_corr",
        "holdout_same_day_rank_corr",
        "used_holdout_for_selection",
    }
    missing_summary = required_summary - set(summary_rows[0])
    if missing_summary:
        fail("theme_rotation_feature_summary.csv missing columns: " + ", ".join(sorted(missing_summary)))
    if any(row.get("used_holdout_for_selection") not in {"False", "false", "0"} for row in summary_rows):
        fail("feature summary must mark holdout as audit-only")

    daily_rows = read_rows(DAILY_PATH)
    required_daily = {
        "日期",
        "主分類",
        "theme_stock_count",
        "theme_strength_rank_5",
        "theme_acceleration_rank_5_20",
        "theme_rotation_candidate_rank",
    }
    missing_daily = required_daily - set(daily_rows[0])
    if missing_daily:
        fail("theme_rotation_daily_strength.csv missing columns: " + ", ".join(sorted(missing_daily)))

    runner = RUNNER_PATH.read_text(encoding="utf-8")
    formal_layer_marker = "formal" + "_layer"
    if formal_layer_marker in runner:
        fail("theme rotation review runner must not reference the formal output layer")
    forbidden_terms = [
        "calibrated" + "_success" + "_rate",
        "70" + " / 80 / 85",
        "\u96f7" + "\u9054",
        "\u54c1\u8cea" + "\u5206\u6578",
        "\u5931\u6557" + "\u5206\u6578",
    ]
    for forbidden in forbidden_terms:
        if forbidden in review or forbidden in runner:
            fail(f"theme rotation review must not introduce forbidden formal wording: {forbidden}")

    print("OK: theme rotation feature review contract passed")
    print(f"REPORT: {REVIEW_MD_PATH}")


if __name__ == "__main__":
    main()
