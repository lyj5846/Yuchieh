from __future__ import annotations

import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REVIEW_MD_PATH = PROJECT_ROOT / "validation_layer" / "repeat_signal_episode_review.md"
SUMMARY_PATH = PROJECT_ROOT / "validation_layer" / "repeat_signal_episode_summary.csv"
EVENTS_PATH = PROJECT_ROOT / "validation_layer" / "repeat_signal_episode_events.csv"
DECISION_JSON_PATH = PROJECT_ROOT / "decision_layer" / "repeat_signal_episode_decision.json"

ALLOWED_STATUS = {
    "keep_tracking_only",
    "allow_reentry_after_reset",
    "insufficient_signal",
}
ALLOWED_NEXT_STEP = {
    "keep_formal_tracking_only",
    "plan_reentry_label_contract",
    "collect_more_completed_repeat_events",
}
REQUIRED_SCENARIOS = {
    "within_10_not_success",
    "within_10_after_success",
    "after_10_reappeared",
    "returned_after_leaving_top10",
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
        fail("missing repeat_signal_episode_review.md")
    review = REVIEW_MD_PATH.read_text(encoding="utf-8")
    for phrase in [
        "label-only repeat signal episode review",
        "Formal output: unchanged by this review.",
        "This does not choose stocks.",
        "This does not train or promote a model.",
        "research_score is a ranking score, not a probability.",
        "10 trading days is a validation window, not a wave boundary.",
    ]:
        if phrase not in review:
            fail(f"repeat signal review missing phrase: {phrase}")

    if not DECISION_JSON_PATH.exists():
        fail("missing repeat_signal_episode_decision.json")
    decision = json.loads(DECISION_JSON_PATH.read_text(encoding="utf-8"))
    if decision.get("status") not in ALLOWED_STATUS:
        fail("unexpected repeat signal episode status")
    if decision.get("recommended_next_step") not in ALLOWED_NEXT_STEP:
        fail("unexpected repeat signal episode next step")
    if not decision.get("reason"):
        fail("decision must include reason")

    summary_rows = read_rows(SUMMARY_PATH)
    required_summary = {
        "scenario",
        "split",
        "events",
        "completed_events",
        "success_rate",
        "avg_10d_high_close_return",
        "daily_market_success_rate",
        "success_lift",
        "return_lift",
        "max_stock_share",
        "max_industry_share",
    }
    missing_summary = required_summary - set(summary_rows[0])
    if missing_summary:
        fail("repeat_signal_episode_summary.csv missing columns: " + ", ".join(sorted(missing_summary)))
    scenarios = {row["scenario"] for row in summary_rows}
    if not REQUIRED_SCENARIOS.issubset(scenarios):
        fail("summary missing required scenarios: " + ", ".join(sorted(REQUIRED_SCENARIOS - scenarios)))
    for scenario in REQUIRED_SCENARIOS:
        splits = {row["split"] for row in summary_rows if row["scenario"] == scenario}
        if not {"overall", "train", "development", "holdout"}.issubset(splits):
            fail(f"scenario {scenario} must include overall/train/development/holdout")

    event_rows = read_rows(EVENTS_PATH)
    required_events = {
        "event_date",
        "stock_id",
        "raw_top_rank",
        "research_score",
        "prior_signal_date",
        "days_since_prior_signal",
        "prior_status_as_of_event",
        "event_label_status",
        "event_future_10d_high_close_return",
        "daily_market_avg_return",
        "event_beat_market",
    }
    missing_events = required_events - set(event_rows[0])
    if missing_events:
        fail("repeat_signal_episode_events.csv missing columns: " + ", ".join(sorted(missing_events)))

    print("OK: repeat signal episode review contract passed")
    print(f"REPORT: {REVIEW_MD_PATH}")


if __name__ == "__main__":
    main()
