from __future__ import annotations

import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REVIEW_MD_PATH = PROJECT_ROOT / "validation_layer" / "target_sensitivity_review.md"
SUMMARY_PATH = PROJECT_ROOT / "validation_layer" / "target_sensitivity_summary.csv"
MONTHLY_PATH = PROJECT_ROOT / "validation_layer" / "target_sensitivity_monthly.csv"
DECISION_JSON_PATH = PROJECT_ROOT / "decision_layer" / "target_sensitivity_decision.json"

ALLOWED_STATUS = {
    "current_target_not_best",
    "current_target_label_viable_but_model_failed",
    "current_target_too_unstable",
}
ALLOWED_NEXT_STEP = {
    "review_target_contract_change",
    "review_data_enrichment",
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
        fail("missing target_sensitivity_review.md")
    review = REVIEW_MD_PATH.read_text(encoding="utf-8")
    for phrase in [
        "label-only target sensitivity",
        "Formal output: unchanged by this review.",
        "This does not choose stocks.",
        "This does not train or promote a model.",
        "not a probability report",
    ]:
        if phrase not in review:
            fail(f"target sensitivity review missing phrase: {phrase}")

    if not DECISION_JSON_PATH.exists():
        fail("missing target_sensitivity_decision.json")
    decision = json.loads(DECISION_JSON_PATH.read_text(encoding="utf-8"))
    if decision.get("status") not in ALLOWED_STATUS:
        fail("unexpected target sensitivity status")
    if decision.get("recommended_next_step") not in ALLOWED_NEXT_STEP:
        fail("unexpected target sensitivity next step")
    if not decision.get("current_target_id"):
        fail("decision must include current_target_id")
    if not decision.get("best_non_old_target_id"):
        fail("decision must include best_non_old_target_id")

    summary_rows = read_rows(SUMMARY_PATH)
    required_summary = {
        "target_id",
        "section",
        "split",
        "success_rate",
        "adverse_first_rate",
        "avg_realized_rule_return",
        "decision_score",
        "stable_across_splits",
    }
    missing_summary = required_summary - set(summary_rows[0])
    if missing_summary:
        fail("target_sensitivity_summary.csv missing columns: " + ", ".join(sorted(missing_summary)))
    overall_splits = {
        row["split"]
        for row in summary_rows
        if row["section"] == "overall"
        and row["target_id"] == decision["current_target_id"]
    }
    if overall_splits != {"train", "development", "holdout"}:
        fail("current target must include train/development/holdout overall rows")

    target_ids = {row["target_id"] for row in summary_rows}
    if "old_touch_3pct_10d" not in target_ids:
        fail("summary must include old target baseline")
    if decision["current_target_id"] not in target_ids:
        fail("summary must include current target")
    if decision["best_non_old_target_id"] not in target_ids:
        fail("summary must include best non-old target")

    monthly_rows = read_rows(MONTHLY_PATH)
    required_monthly = {"target_id", "section", "group", "split", "success_rate"}
    missing_monthly = required_monthly - set(monthly_rows[0])
    if missing_monthly:
        fail("target_sensitivity_monthly.csv missing columns: " + ", ".join(sorted(missing_monthly)))

    print("OK: target sensitivity review contract passed")
    print(f"REPORT: {REVIEW_MD_PATH}")


if __name__ == "__main__":
    main()
