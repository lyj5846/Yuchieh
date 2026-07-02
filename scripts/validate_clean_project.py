from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "project_config.json"
REPORT_PATH = PROJECT_ROOT / "data_layer" / "initial_data_quality_report.md"
OLD_PROJECT_MARKER = "stock" + "_raw_only" + "_project"


def read_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def fail(message: str) -> None:
    raise SystemExit(f"FAIL: {message}")


def scan_forbidden_references(config: dict) -> list[str]:
    forbidden = [s for s in config["forbidden_path_fragments"] if s]
    hits: list[str] = []
    forbidden.append(OLD_PROJECT_MARKER)
    for path in PROJECT_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if PROJECT_ROOT / "research_layer" in path.parents:
            continue
        if path.name == "validate_clean_project.py":
            continue
        if path.suffix.lower() not in {".py", ".json", ".md", ".txt"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for fragment in forbidden:
            if fragment in text:
                hits.append(f"{path}: {fragment}")
    return hits


def csv_summary(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = 0
        last_row = None
        for row in reader:
            rows += 1
            last_row = row
    return {
        "path": str(path),
        "columns": len(header),
        "rows": rows,
        "first_columns": header[:12],
        "last_row_first_values": (last_row or [])[:8],
    }


def main() -> None:
    config = read_config()
    allowed = config["allowed_inputs"]
    paths = {name: Path(value) for name, value in allowed.items()}

    for name, path in paths.items():
        if not path.exists():
            fail(f"missing allowed input {name}: {path}")

    for name, path in paths.items():
        if name == "theme_group" and PROJECT_ROOT not in path.parents:
            fail("theme_group must live inside the clean project inputs folder")
        if OLD_PROJECT_MARKER in str(path):
            fail(f"allowed input still points to old project: {name}")

    hits = scan_forbidden_references(config)
    unexpected_hits = [
        hit
        for hit in hits
        if "project_config.json" not in hit and "README.md" not in hit
    ]
    if unexpected_hits:
        fail("forbidden references found:\n" + "\n".join(unexpected_hits))

    summaries = {name: csv_summary(path) for name, path in paths.items()}
    lines = [
        "# Initial Data Quality Report",
        "",
        f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Formal candidate default: {config['formal_candidate_default']}",
        "",
        "## Allowed Inputs",
        "",
    ]
    for name, summary in summaries.items():
        lines.extend(
            [
                f"### {name}",
                "",
                f"- Path: `{summary['path']}`",
                f"- Rows: {summary['rows']}",
                f"- Columns: {summary['columns']}",
                f"- First columns: `{', '.join(summary['first_columns'])}`",
                f"- Last row preview: `{', '.join(summary['last_row_first_values'])}`",
                "",
            ]
        )

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"OK: clean project validation passed")
    print(f"REPORT: {REPORT_PATH}")


if __name__ == "__main__":
    main()

