from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str]) -> None:
    printable = " ".join(command)
    print(f"\n==> {printable}")
    result = subprocess.run(command, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main() -> None:
    python = sys.executable
    scripts = sorted(str(path.relative_to(PROJECT_ROOT)) for path in (PROJECT_ROOT / "scripts").glob("*.py"))
    run([python, "-m", "py_compile", *scripts])
    for script in [
        "scripts/run_target_redefinition_review.py",
        "scripts/check_github_governance.py",
        "scripts/check_repository_static_contract.py",
        "scripts/check_planning_contract.py",
        "scripts/check_main_model_label_contract.py",
        "scripts/check_main_model_failure_diagnosis.py",
        "scripts/run_data_learnability_review.py",
        "scripts/check_data_learnability_review.py",
        "scripts/run_target_sensitivity_review.py",
        "scripts/check_target_sensitivity_review.py",
        "scripts/run_data_enrichment_review.py",
        "scripts/check_data_enrichment_review.py",
        "scripts/run_event_risk_collector_spec.py",
        "scripts/check_event_risk_collector_spec.py",
        "scripts/check_target_redefinition_review.py",
        "scripts/check_architecture_contract.py",
        "scripts/validate_clean_project.py",
        "scripts/check_github_connection.py",
    ]:
        run([python, script])

    print("\nOK: local quality gate passed")


if __name__ == "__main__":
    main()
