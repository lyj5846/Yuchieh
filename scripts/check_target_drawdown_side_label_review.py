from __future__ import annotations

import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = PROJECT_ROOT / "label_layer" / "target_drawdown_side_label_contract.md"
BASELINE_PATH = PROJECT_ROOT / "validation_layer" / "target_drawdown_side_label_baseline.csv"
DECISION_MD_PATH = PROJECT_ROOT / "decision_layer" / "target_drawdown_side_label_decision.md"
DECISION_JSON_PATH = PROJECT_ROOT / "decision_layer" / "target_drawdown_side_label_decision.json"


def fail(message: str) -> None:
    raise SystemExit(f"FAIL: {message}")


def require_text(path: Path, phrases: list[str]) -> str:
    if not path.exists():
        fail(f"missing output: {path.relative_to(PROJECT_ROOT)}")
    text = path.read_text(encoding="utf-8")
    for phrase in phrases:
        if phrase not in text:
            fail(f"{path.relative_to(PROJECT_ROOT)} missing phrase: {phrase}")
    return text


def read_baseline() -> list[dict[str, str]]:
    if not BASELINE_PATH.exists():
        fail("missing target_drawdown_side_label_baseline.csv")
    with BASELINE_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        fail("target_drawdown_side_label_baseline.csv is empty")
    required = {
        "section",
        "split",
        "group",
        "rows",
        "old_touch_success_rate",
        "risk_adjusted_hard_success_rate",
        "proposed_primary_success_rate",
        "clean_success_rate",
        "painful_success_rate",
        "painful_success_among_success_rate",
        "avg_max_adverse_return_10d",
    }
    missing = required - set(rows[0])
    if missing:
        fail("target_drawdown_side_label_baseline.csv missing columns: " + ", ".join(sorted(missing)))
    return rows


def read_decision() -> dict:
    if not DECISION_JSON_PATH.exists():
        fail("missing target_drawdown_side_label_decision.json")
    with DECISION_JSON_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    contract = require_text(
        CONTRACT_PATH,
        [
            "label-only review",
            "`target_success`: any close in the next 1 to 10 trading days is at least +3% above buy open.",
            "If price first drops to -3% but later reaches +3%, `target_success` still remains success.",
            "This contract does not use -3% as an automatic failure.",
            "does not write formal output",
        ],
    )
    forbidden_contract_phrases = [
        "Success: the profit event happens before the adverse event.",
        "Conservative tie rule",
        "adverse event wins",
    ]
    for phrase in forbidden_contract_phrases:
        if phrase in contract:
            fail(f"drawdown side-label contract must not keep hard-failure phrase: {phrase}")

    decision_text = require_text(
        DECISION_MD_PATH,
        [
            "Decision status:",
            "Formal output: unchanged by this review.",
            "主目標仍算成功",
            "red-light change",
        ],
    )
    if "calibrated probability" in decision_text.lower():
        fail("decision must not describe this review as a calibrated probability")

    rows = read_baseline()
    overall = [row for row in rows if row["section"] == "overall"]
    overall_splits = {row["split"] for row in overall}
    if overall_splits != {"train", "development", "holdout"}:
        fail("baseline must contain overall train/development/holdout rows")

    for row in overall:
        split = row["split"]
        old_rate = float(row["old_touch_success_rate"])
        hard_rate = float(row["risk_adjusted_hard_success_rate"])
        proposed_rate = float(row["proposed_primary_success_rate"])
        painful_success_count = float(row.get("painful_success_count", "0") or 0)
        if abs(proposed_rate - old_rate) > 1e-12:
            fail(f"proposed primary success must equal old +3% touch success for {split}")
        if hard_rate > proposed_rate:
            fail(f"hard risk-adjusted rate cannot exceed proposed primary success for {split}")
        if painful_success_count <= 0:
            fail(f"{split} must have recovered painful success examples")

    decision = read_decision()
    if decision.get("formal_output_changed") is not False:
        fail("decision json must record formal_output_changed=false")
    if decision.get("recommended_next_step") != "update_main_label_contract_to_drawdown_side_labels":
        fail("decision json recommended_next_step is not the expected target-contract repair")
    if decision.get("red_light_required_for_next_step") is not True:
        fail("decision json must mark next step as red light")

    print("OK: target drawdown side-label review contract passed")


if __name__ == "__main__":
    main()
