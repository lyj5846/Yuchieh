from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLANNING_DIR = PROJECT_ROOT / "planning_layer"
PLAN_JSON_PATH = PLANNING_DIR / "current_model_plan.json"
PLAN_MD_PATH = PLANNING_DIR / "current_model_plan.md"
AUDIT_PATH = PLANNING_DIR / "planning_audit.md"
REPORT_PATH = PLANNING_DIR / "planning_contract_report.md"

REQUIRED_TOP_LEVEL_KEYS = {
    "plan_id",
    "generated_at",
    "data_latest_date",
    "planning_status",
    "problem_statement",
    "recommended_experiment_id",
    "confirmation_required",
    "experiment_candidates",
}

REQUIRED_CANDIDATE_KEYS = {
    "id",
    "hypothesis",
    "why_now",
    "target_label",
    "allowed_inputs",
    "feature_changes",
    "model_changes",
    "validation_checks",
    "pass_criteria",
    "rejection_criteria",
    "expected_outputs",
    "risk_notes",
}


def fail(message: str) -> None:
    raise SystemExit(f"FAIL: {message}")


def load_plan() -> dict:
    if not PLAN_JSON_PATH.exists():
        fail(f"missing plan json: {PLAN_JSON_PATH}")
    with PLAN_JSON_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def collect_issues(plan: dict) -> list[str]:
    issues: list[str] = []
    missing = REQUIRED_TOP_LEVEL_KEYS - set(plan)
    if missing:
        issues.append(f"missing top-level keys: {sorted(missing)}")
        return issues

    candidates = plan["experiment_candidates"]
    if not isinstance(candidates, list):
        issues.append("experiment_candidates must be a list")
        return issues
    if not 1 <= len(candidates) <= 3:
        issues.append("experiment_candidates must contain 1 to 3 items")
    if plan["confirmation_required"] is not True:
        issues.append("confirmation_required must be true")

    ids = [candidate.get("id") for candidate in candidates]
    if plan["recommended_experiment_id"] not in set(ids):
        issues.append("recommended_experiment_id must match one candidate id")
    if len(ids) != len(set(ids)):
        issues.append("candidate ids must be unique")

    for candidate in candidates:
        missing_candidate = REQUIRED_CANDIDATE_KEYS - set(candidate)
        if missing_candidate:
            issues.append(f"candidate {candidate.get('id')} missing keys: {sorted(missing_candidate)}")

    serialized = json.dumps(plan, ensure_ascii=False)
    if "第幾輪" in serialized:
        issues.append("formal planning output must not use round-number framing")
    if "research_score_probability" in serialized:
        issues.append("research scores must not be presented as calibrated rates")

    return issues


def check_files_exist() -> list[str]:
    issues: list[str] = []
    for path in [PLAN_MD_PATH, PLAN_JSON_PATH, AUDIT_PATH]:
        if not path.exists():
            issues.append(f"missing planning output: {path.name}")
    return issues


def check_planning_script_boundaries() -> list[str]:
    issues: list[str] = []
    script = PROJECT_ROOT / "scripts" / "run_planning_pipeline.py"
    if not script.exists():
        issues.append("missing run_planning_pipeline.py")
        return issues
    text = script.read_text(encoding="utf-8", errors="ignore")
    formal_dir_literal = '"formal' + '_layer"'
    if formal_dir_literal in text:
        issues.append("planning pipeline must not reference formal output paths")
    if "FORMAL_" in text:
        issues.append("planning pipeline must not define formal output handles")
    return issues


def write_report(issues: list[str]) -> None:
    status = "PASS" if not issues else "FAIL"
    lines = [
        "# Planning Contract Report",
        "",
        f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Status: {status}",
        "- Required confirmation: true",
        "- Maximum experiment candidates: 3",
        "",
    ]
    if issues:
        lines.append("## Issues")
        lines.append("")
        for issue in issues:
            lines.append(f"- {issue}")
    else:
        lines.append("No planning contract violations found.")
    lines.append("")
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    issues = check_files_exist()
    plan = load_plan() if PLAN_JSON_PATH.exists() else {}
    if plan:
        issues.extend(collect_issues(plan))
    issues.extend(check_planning_script_boundaries())
    write_report(issues)
    if issues:
        fail("planning contract violations:\n" + "\n".join(issues))
    print("OK: planning contract passed")
    print(f"REPORT: {REPORT_PATH}")


if __name__ == "__main__":
    main()
