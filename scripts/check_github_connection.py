from __future__ import annotations

import shutil
import subprocess
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = PROJECT_ROOT / "validation_layer" / "github_connection_report.md"


def resolve_command(command: str) -> str | None:
    found = shutil.which(command)
    if found:
        return found
    if command == "git":
        for candidate in [
            Path("C:/Program Files/Git/cmd/git.exe"),
            Path("C:/Program Files/Git/bin/git.exe"),
            Path("C:/Program Files (x86)/Git/cmd/git.exe"),
            Path("C:/Program Files (x86)/Git/bin/git.exe"),
            Path.home() / "AppData/Local/Programs/Git/cmd/git.exe",
            Path.home() / "AppData/Local/Programs/Git/bin/git.exe",
        ]:
            if candidate.exists():
                return str(candidate)
    if command == "gh":
        for candidate in [
            Path("C:/Program Files/GitHub CLI/gh.exe"),
            Path.home() / "AppData/Local/Programs/GitHub CLI/gh.exe",
        ]:
            if candidate.exists():
                return str(candidate)
    return None


def command_version(command: str, args: list[str]) -> tuple[bool, str]:
    executable = resolve_command(command)
    if not executable:
        return False, "not found"
    try:
        result = subprocess.run(
            [executable, *args],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except Exception as exc:  # pragma: no cover - environment probe
        return False, f"error: {exc}"
    output = (result.stdout or result.stderr).strip().splitlines()
    return result.returncode == 0, output[0] if output else "found"


def main() -> None:
    git_ok, git_info = command_version("git", ["--version"])
    gh_ok, gh_info = command_version("gh", ["--version"])
    is_git_repo = (PROJECT_ROOT / ".git").is_dir()
    github_files_ready = all(
        path.exists()
        for path in [
            PROJECT_ROOT / ".github" / "workflows" / "model-contracts.yml",
            PROJECT_ROOT / ".github" / "pull_request_template.md",
            PROJECT_ROOT / ".github" / "ISSUE_TEMPLATE" / "model-repair.yml",
        ]
    )
    ready_for_remote = bool(git_ok and is_git_repo)
    lines = [
        "# GitHub Connection Report",
        "",
        f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- GitHub governance files ready: {github_files_ready}",
        f"- Local git command: {git_info}",
        f"- Local gh command: {gh_info}",
        f"- Local git repository: {is_git_repo}",
        f"- Ready for remote push: {ready_for_remote}",
        "",
        "## Meaning",
        "",
    ]
    if ready_for_remote:
        lines.append("Local Git is ready. The next step is adding a GitHub remote and opening the first PR.")
    else:
        lines.extend(
            [
                "GitHub governance files are ready, but this project is not yet connected to a remote repository.",
                "To finish GitHub connection, install/enable Git locally or provide a GitHub repository that can receive these files.",
            ]
        )
    lines.append("")
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print("OK: GitHub connection check completed")
    print(f"REPORT: {REPORT_PATH}")


if __name__ == "__main__":
    main()
