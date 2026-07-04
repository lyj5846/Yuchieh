from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def select_python() -> str:
    bundled = (
        Path.home()
        / ".cache"
        / "codex-runtimes"
        / "codex-primary-runtime"
        / "dependencies"
        / "python"
        / "python.exe"
    )
    if not bundled.exists():
        return sys.executable
    try:
        import numpy  # noqa: F401
        import pandas  # noqa: F401
    except ModuleNotFoundError:
        return str(bundled)
    return sys.executable


def run(command: list[str]) -> None:
    printable = " ".join(command)
    print(f"\n==> {printable}")
    result = subprocess.run(command, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main() -> None:
    python = select_python()
    scripts = sorted(str(path.relative_to(PROJECT_ROOT)) for path in (PROJECT_ROOT / "scripts").glob("*.py"))
    run([python, "-m", "py_compile", *scripts])
    for script in [
        "scripts/run_target_redefinition_review.py",
        "scripts/check_github_governance.py",
        "scripts/check_repository_static_contract.py",
        "scripts/check_planning_contract.py",
        "scripts/check_main_model_label_contract.py",
        "scripts/check_main_model_failure_diagnosis.py",
        "scripts/check_formal_tracking_contract.py",
        "scripts/run_data_learnability_review.py",
        "scripts/check_data_learnability_review.py",
        "scripts/run_target_sensitivity_review.py",
        "scripts/check_target_sensitivity_review.py",
        "scripts/run_target_drawdown_side_label_review.py",
        "scripts/check_target_drawdown_side_label_review.py",
        "scripts/run_repeat_signal_episode_review.py",
        "scripts/check_repeat_signal_episode_review.py",
        "scripts/run_data_enrichment_review.py",
        "scripts/check_data_enrichment_review.py",
        "scripts/run_event_risk_collector_spec.py",
        "scripts/check_event_risk_collector_spec.py",
        "scripts/check_event_risk_calendar_collector.py",
        "scripts/run_event_risk_calendar_coverage_review.py",
        "scripts/check_event_risk_calendar_coverage_review.py",
        "scripts/check_historical_event_risk_backfill_collector.py",
        "scripts/run_historical_event_risk_backfill_coverage_review.py",
        "scripts/check_historical_event_risk_backfill_coverage_review.py",
        "scripts/run_attention_disposition_feature_contract.py",
        "scripts/check_attention_disposition_feature_contract.py",
        "scripts/run_attention_disposition_feature_generation_check.py",
        "scripts/check_attention_disposition_feature_generation_check.py",
        "scripts/run_attention_disposition_model_input_approval_review.py",
        "scripts/check_attention_disposition_model_input_approval_review.py",
        "scripts/check_attention_disposition_main_training_wiring.py",
        "scripts/check_target_redefinition_review.py",
        "scripts/check_architecture_contract.py",
        "scripts/validate_clean_project.py",
        "scripts/check_github_connection.py",
    ]:
        run([python, script])

    print("\nOK: local quality gate passed")


if __name__ == "__main__":
    main()
