from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "project_config.json"
SCORES_PATH = PROJECT_ROOT / "model_layer" / "main_model_scores.csv"
FORMAL_DIR = PROJECT_ROOT / "formal_layer"
FORMAL_CANDIDATES_PATH = FORMAL_DIR / "formal_candidates.csv"
FORMAL_TRACKING_PATH = FORMAL_DIR / "formal_candidate_tracking.csv"
FORMAL_DAILY_REPORT_PATH = FORMAL_DIR / "formal_daily_report.md"
FORMAL_STATUS_PATH = FORMAL_DIR / "formal_status.md"
CONFIRMED_PLAN_ID = "drawdown_side_label_main_model_training_plan"


def fail(message: str) -> None:
    raise SystemExit(f"FAIL: {message}")


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
    return str(bundled) if bundled.exists() else sys.executable


def parse_as_of_date(value: str) -> str:
    raw = str(value).strip()
    formats = ["%Y-%m-%d", "%Y/%m/%d", "%Y%m%d", "%m/%d", "%m-%d"]
    for fmt in formats:
        try:
            parsed = datetime.strptime(raw, fmt)
            if fmt in {"%m/%d", "%m-%d"}:
                parsed = parsed.replace(year=datetime.now().year)
            return parsed.strftime("%Y-%m-%d")
        except ValueError:
            continue
    fail(f"cannot parse --as-of-date: {value}")


def read_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def csv_dates(path: Path) -> Counter[str]:
    if not path.exists():
        fail(f"missing CSV: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            fail(f"missing header: {path}")
        date_column = reader.fieldnames[0]
        dates: Counter[str] = Counter()
        for row in reader:
            value = str(row.get(date_column, "")).strip()
            if not value:
                continue
            dates[parse_as_of_date(value)] += 1
    if not dates:
        fail(f"missing date rows: {path}")
    return dates


def latest_date(dates: Counter[str]) -> str:
    return sorted(dates)[-1]


def validate_raw_sources(config: dict, as_of_date: str) -> dict:
    allowed = config.get("allowed_inputs", {})
    if set(allowed) != {"stock_daily_all", "market_daily", "theme_group"}:
        fail("allowed_inputs must contain exactly stock_daily_all, market_daily, theme_group")

    stock_dates = csv_dates(Path(allowed["stock_daily_all"]))
    market_dates = csv_dates(Path(allowed["market_daily"]))
    if as_of_date not in stock_dates:
        fail(f"stock_daily_all.csv does not contain {as_of_date}")
    if as_of_date not in market_dates:
        fail(f"market_daily.csv does not contain {as_of_date}")

    return {
        "stock_latest": latest_date(stock_dates),
        "market_latest": latest_date(market_dates),
        "stock_as_of_rows": stock_dates[as_of_date],
        "market_as_of_rows": market_dates[as_of_date],
    }


def score_dates() -> Counter[str]:
    if not SCORES_PATH.exists():
        return Counter()
    with SCORES_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return Counter()
        date_column = reader.fieldnames[0]
        dates: Counter[str] = Counter()
        for row in reader:
            value = str(row.get(date_column, "")).strip()
            if value:
                dates[parse_as_of_date(value)] += 1
    return dates


def run_command(command: list[str]) -> None:
    print("\n==> " + " ".join(command))
    result = subprocess.run(command, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def run_capture(command: list[str]) -> str:
    result = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(result.returncode)
    return result.stdout.strip()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def table_from_section(text: str, heading: str) -> list[str]:
    marker = f"## {heading}"
    if marker not in text:
        return []
    rest = text.split(marker, 1)[1]
    next_heading = rest.find("\n## ")
    section = rest if next_heading == -1 else rest[:next_heading]
    rows = []
    for line in section.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        if stripped.replace("|", "").strip() == "":
            continue
        rows.append(stripped)
    return rows


def markdown_count_table(counts: Counter[str]) -> list[str]:
    lines = ["| 狀態 | 筆數 |", "| --- | ---: |"]
    if not counts:
        lines.append("| 無 | 0 |")
        return lines
    for status, count in sorted(counts.items()):
        lines.append(f"| {status or 'blank'} | {count} |")
    return lines


def markdown_check_table(checks: list[str]) -> list[str]:
    lines = ["| 檢查項目 | 結果 |", "| --- | --- |"]
    for check in checks:
        lines.append(f"| {check} | passed |")
    return lines


def write_summary(
    as_of_date: str,
    raw_info: dict,
    score_before: Counter[str],
    score_after: Counter[str],
    retrained: bool,
    checks: list[str],
) -> Path:
    candidates = read_csv_rows(FORMAL_CANDIDATES_PATH)
    tracking = read_csv_rows(FORMAL_TRACKING_PATH)
    report_text = FORMAL_DAILY_REPORT_PATH.read_text(encoding="utf-8") if FORMAL_DAILY_REPORT_PATH.exists() else ""
    new_candidate_table = table_from_section(report_text, "今日新進正式候選")
    continuation_table = table_from_section(report_text, "高分續強但已追蹤")
    status_counts = Counter(row.get("tracking_status", "") for row in tracking)
    score_after_latest = latest_date(score_after) if score_after else "none"
    score_before_latest = latest_date(score_before) if score_before else "none"

    if not new_candidate_table:
        new_candidate_table = [
            "| 股票 | 原始排名 | research_score | 連續被推薦次數 | 類型 |",
            "| --- | --- | --- | --- | --- |",
            "|  |  |  |  |  |",
        ]
    if not continuation_table:
        continuation_table = [
            "| 股票 | 原始排名 | research_score | 連續被推薦次數 | 前次訊號日 | 追蹤狀態 | 目前最高收盤報酬 | 最高收盤報酬日期 |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
            "|  |  |  |  |  |  |  |  |",
        ]
    candidate_note = (
        "正式候選已產生。"
        if candidates
        else "無；已評分但未通過正式候選條件。"
    )

    path = FORMAL_DIR / f"update_run_summary_{as_of_date.replace('-', '')}.md"
    path.write_text(
        "\n".join(
            [
                "# Update Run Summary",
                "",
                f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"- Requested as-of date: {as_of_date}",
                f"- Raw stock latest date: {raw_info['stock_latest']}",
                f"- Raw market latest date: {raw_info['market_latest']}",
                f"- Stock rows on as-of date: {raw_info['stock_as_of_rows']}",
                f"- Market rows on as-of date: {raw_info['market_as_of_rows']}",
                f"- Score latest date before sync: {score_before_latest}",
                f"- Score latest date after sync: {score_after_latest}",
                f"- Main model retrained: {str(retrained).lower()}",
                "- Score note: research_score is a ranking score, not a calibrated probability.",
                "",
                "## 今日新進正式候選",
                "",
                candidate_note,
                "",
                *new_candidate_table,
                "",
                "## 高分續強但已追蹤",
                "",
                *continuation_table,
                "",
                "## 正式候選追蹤狀態",
                "",
                *markdown_count_table(status_counts),
                "",
                "## Checks",
                "",
                *markdown_check_table(checks),
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def git_executable() -> str:
    windows_git = Path(r"C:\Program Files\Git\cmd\git.exe")
    return str(windows_git) if windows_git.exists() else "git"


def commit_and_push(as_of_date: str) -> None:
    git = git_executable()
    status = run_capture([git, "status", "--porcelain"])
    if not status:
        print("OK: no Git changes to commit")
        return
    run_command([git, "add", "-A"])
    run_command([git, "commit", "-m", f"Update formal report to {as_of_date}"])
    run_command([git, "push", "origin", "main"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the complete update-to-date workflow.")
    parser.add_argument("--as-of-date", required=True, help="Completed trading date, e.g. 2026-07-02 or 7/2.")
    parser.add_argument("--force-train", action="store_true", help="Retrain even if model scores already contain the date.")
    parser.add_argument("--no-git", action="store_true", help="Run the full workflow but skip commit/push.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    as_of_date = parse_as_of_date(args.as_of_date)
    python = select_python()
    config = read_config()
    raw_info = validate_raw_sources(config, as_of_date)

    before_scores = score_dates()
    retrained = False
    if args.force_train or as_of_date not in before_scores:
        retrained = True
        run_command(
            [
                python,
                "scripts/run_main_model_training_pipeline.py",
                "--confirmed-plan",
                CONFIRMED_PLAN_ID,
            ]
        )

    after_scores = score_dates()
    if as_of_date not in after_scores:
        fail(
            f"model scores still do not contain {as_of_date} after sync; "
            "formal output was not generated."
        )

    run_command([python, "scripts/run_main_pipeline.py", "--as-of-date", as_of_date])
    run_command([python, "scripts/run_main_model_failure_diagnosis.py"])
    run_command([python, "scripts/run_local_quality_gate.py"])

    summary = write_summary(
        as_of_date=as_of_date,
        raw_info=raw_info,
        score_before=before_scores,
        score_after=after_scores,
        retrained=retrained,
        checks=[
            "run_main_pipeline",
            "run_main_model_failure_diagnosis",
            "run_local_quality_gate",
        ],
    )
    print(f"SUMMARY: {summary}")

    if not args.no_git:
        commit_and_push(as_of_date)

    print("OK: update-to-date workflow completed")
    print(f"AS_OF_DATE: {as_of_date}")


if __name__ == "__main__":
    main()
