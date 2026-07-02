from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_PATHS = [
    PROJECT_ROOT / ".github" / "workflows" / "model-contracts.yml",
    PROJECT_ROOT / ".github" / "pull_request_template.md",
    PROJECT_ROOT / ".github" / "ISSUE_TEMPLATE" / "model-repair.yml",
    PROJECT_ROOT / "docs" / "github_workflow_contract.md",
    PROJECT_ROOT / "scripts" / "check_github_connection.py",
    PROJECT_ROOT / "scripts" / "check_repository_static_contract.py",
    PROJECT_ROOT / "scripts" / "run_local_quality_gate.py",
]


def fail(message: str) -> None:
    raise SystemExit(f"FAIL: {message}")


def main() -> None:
    missing = [path.relative_to(PROJECT_ROOT) for path in REQUIRED_PATHS if not path.exists()]
    if missing:
        fail("missing GitHub governance files: " + ", ".join(str(path) for path in missing))

    workflow = (PROJECT_ROOT / ".github" / "workflows" / "model-contracts.yml").read_text(encoding="utf-8")
    for command in [
        "python scripts/check_github_governance.py",
        "python scripts/check_repository_static_contract.py",
        "python scripts/check_planning_contract.py",
    ]:
        if command not in workflow:
            fail(f"workflow missing command: {command}")

    pr_template = (PROJECT_ROOT / ".github" / "pull_request_template.md").read_text(encoding="utf-8")
    for phrase in [
        "Research scores are not described as calibrated success rates",
        "Formal output was not changed by a research or training script",
        "scripts/run_main_pipeline.py",
    ]:
        if phrase not in pr_template:
            fail(f"pull request template missing phrase: {phrase}")

    print("OK: GitHub governance contract passed")


if __name__ == "__main__":
    main()
