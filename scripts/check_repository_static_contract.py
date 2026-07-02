from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def fail(message: str) -> None:
    raise SystemExit(f"FAIL: {message}")


def main() -> None:
    config_path = PROJECT_ROOT / "project_config.json"
    if not config_path.exists():
        fail("missing project_config.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    allowed = config.get("allowed_inputs", {})
    if set(allowed) != {"stock_daily_all", "market_daily", "theme_group"}:
        fail("allowed_inputs must contain exactly stock_daily_all, market_daily, theme_group")
    old_project = "stock" + "_raw_only" + "_project"
    if old_project in json.dumps(config, ensure_ascii=False):
        fail("project_config.json must not reference the old project")

    required_dirs = [
        "data_layer",
        "label_layer",
        "feature_layer",
        "model_layer",
        "planning_layer",
        "validation_layer",
        "decision_layer",
        "formal" + "_layer",
        "research_layer",
        "scripts",
        "docs",
        "inputs",
    ]
    missing_dirs = [name for name in required_dirs if not (PROJECT_ROOT / name).is_dir()]
    if missing_dirs:
        fail("missing required dirs: " + ", ".join(missing_dirs))

    large_repo_output = PROJECT_ROOT / "model_layer" / "main_model_scores.csv"
    if large_repo_output.exists():
        gitignore = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
        if "model_layer/main_model_scores.csv" not in gitignore:
            fail("large local model score output must be ignored by Git")

    workflow = PROJECT_ROOT / ".github" / "workflows" / "model-contracts.yml"
    if not workflow.exists():
        fail("missing GitHub Actions workflow")
    workflow_text = workflow.read_text(encoding="utf-8")
    forbidden_remote_checks = [
        "check_main_model_label_contract.py",
        "check_main_model_failure_diagnosis.py",
        "validate_clean_project.py",
    ]
    for forbidden in forbidden_remote_checks:
        if forbidden in workflow_text:
            fail(f"remote workflow must not run data-bound check: {forbidden}")

    print("OK: repository static contract passed")


if __name__ == "__main__":
    main()
