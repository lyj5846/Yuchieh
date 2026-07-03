from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "project_config.json"
REPORT_PATH = PROJECT_ROOT / "validation_layer" / "architecture_contract_report.md"

REQUIRED_DIRS = [
    "data_layer",
    "label_layer",
    "feature_layer",
    "model_layer",
    "planning_layer",
    "validation_layer",
    "decision_layer",
    "formal_layer",
    "research_layer",
    "scripts",
    "docs",
    "inputs",
]

FORMAL_WRITER_ALLOWLIST = {
    PROJECT_ROOT / "scripts" / "run_main_pipeline.py",
    PROJECT_ROOT / "scripts" / "run_update_to_date.py",
    PROJECT_ROOT / "scripts" / "check_architecture_contract.py",
}


def read_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def fail(message: str) -> None:
    raise SystemExit(f"FAIL: {message}")


def text_files() -> list[Path]:
    suffixes = {".py", ".md", ".json", ".txt"}
    blocked_roots = {PROJECT_ROOT / "research_layer"}
    files: list[Path] = []
    for path in PROJECT_ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        if any(root in path.parents for root in blocked_roots):
            continue
        files.append(path)
    return files


def check_directories() -> list[str]:
    issues: list[str] = []
    for name in REQUIRED_DIRS:
        if not (PROJECT_ROOT / name).is_dir():
            issues.append(f"missing directory: {name}")
    retired_dir = "experiment" + "_layer"
    if (PROJECT_ROOT / retired_dir).exists():
        issues.append(f"retired {retired_dir} still exists; use research_layer")
    return issues


def check_inputs(config: dict) -> list[str]:
    issues: list[str] = []
    allowed = config.get("allowed_inputs", {})
    if set(allowed) != {"stock_daily_all", "market_daily", "theme_group"}:
        issues.append("allowed_inputs must contain exactly the three approved sources")
    old_marker = "stock" + "_raw_only" + "_project"
    for name, value in allowed.items():
        path = Path(value)
        if old_marker in str(path):
            issues.append(f"input points to old project: {name}")
        if not path.exists():
            issues.append(f"missing input: {name} -> {path}")
        if name == "theme_group" and PROJECT_ROOT not in path.parents:
            issues.append("theme_group must live inside this clean project")
    return issues


def check_formal_writers() -> list[str]:
    issues: list[str] = []
    for path in (PROJECT_ROOT / "scripts").glob("*.py"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if '"formal_layer"' not in text:
            continue
        if path.resolve() not in {p.resolve() for p in FORMAL_WRITER_ALLOWLIST}:
            issues.append(f"script references formal output directly: {path.name}")
    return issues


def check_forbidden_surface_terms() -> list[str]:
    issues: list[str] = []
    old_project = "stock" + "_raw_only" + "_project"
    old_report = "model" + "_layer_10d_high_close"
    forbidden = [
        old_project,
        old_report,
        "70" + " / 80 / 85",
        "70" + "/80/85",
        "\u96f7\u9054",
        "\u54c1\u8cea\u5206\u6578",
        "\u5931\u6557\u5206\u6578",
    ]
    allow = {
        (PROJECT_ROOT / "project_config.json").resolve(),
        (PROJECT_ROOT / "scripts" / "check_architecture_contract.py").resolve(),
        REPORT_PATH.resolve(),
    }
    for path in text_files():
        if path.resolve() in allow:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for term in forbidden:
            if term and term in text:
                issues.append(f"forbidden surface term in {path.relative_to(PROJECT_ROOT)}: {term}")
    return issues


def write_report(issues: list[str]) -> None:
    status = "PASS" if not issues else "FAIL"
    lines = [
        "# Architecture Contract Report",
        "",
        f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Status: {status}",
        "- Formal writer: `scripts/run_main_pipeline.py`",
        "- Update summary writer: `scripts/run_update_to_date.py`",
        "- Research output layer: `research_layer`",
        "",
    ]
    if issues:
        lines.append("## Issues")
        lines.append("")
        for issue in issues:
            lines.append(f"- {issue}")
    else:
        lines.append("No architecture violations found.")
    lines.append("")
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    config = read_config()
    issues: list[str] = []
    issues.extend(check_directories())
    issues.extend(check_inputs(config))
    issues.extend(check_formal_writers())
    issues.extend(check_forbidden_surface_terms())
    write_report(issues)
    if issues:
        fail("architecture contract violations:\n" + "\n".join(issues))
    print("OK: architecture contract passed")
    print(f"REPORT: {REPORT_PATH}")


if __name__ == "__main__":
    main()
