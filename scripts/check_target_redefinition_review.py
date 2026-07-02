from __future__ import annotations

import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = PROJECT_ROOT / "label_layer" / "target_redefinition_contract.md"
BASELINE_PATH = PROJECT_ROOT / "validation_layer" / "target_redefinition_baseline.csv"
DECISION_PATH = PROJECT_ROOT / "decision_layer" / "target_redefinition_decision.md"


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
        fail("missing target_redefinition_baseline.csv")
    with BASELINE_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        fail("target_redefinition_baseline.csv is empty")
    required = {
        "section",
        "split",
        "group",
        "rows",
        "old_success_rate",
        "risk_adjusted_success_rate",
        "old_success_but_risk_failed_count",
        "same_day_both_event_count",
        "avg_realized_10d_trade_return",
    }
    missing = required - set(rows[0])
    if missing:
        fail("target_redefinition_baseline.csv missing columns: " + ", ".join(sorted(missing)))
    return rows


def main() -> None:
    require_text(
        CONTRACT_PATH,
        [
            "label-only review",
            "Success: the profit event happens before the adverse event.",
            "Conservative tie rule",
            "does not write formal output",
        ],
    )
    decision_text = require_text(
        DECISION_PATH,
        [
            "Decision status:",
            "Formal output: unchanged by this review.",
            "Do not update formal candidates from this review.",
        ],
    )
    if "calibrated probability" in decision_text.lower():
        fail("decision must not describe this review as a calibrated probability")

    rows = read_baseline()
    overall_splits = {
        row["split"]
        for row in rows
        if row["section"] == "overall"
    }
    if overall_splits != {"train", "development", "holdout"}:
        fail("baseline must contain overall train/development/holdout rows")

    for row in rows:
        if row["section"] != "overall":
            continue
        old_rate = float(row["old_success_rate"])
        new_rate = float(row["risk_adjusted_success_rate"])
        if new_rate > old_rate:
            fail(f"risk-adjusted success rate cannot exceed old success rate for {row['split']}")

    print("OK: target redefinition review contract passed")


if __name__ == "__main__":
    main()
