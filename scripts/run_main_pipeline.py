from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "project_config.json"

DATA_STATUS_PATH = PROJECT_ROOT / "data_layer" / "main_pipeline_data_status.md"
LABEL_CONTRACT_PATH = PROJECT_ROOT / "label_layer" / "label_contract.md"
FEATURE_CONTRACT_PATH = PROJECT_ROOT / "feature_layer" / "feature_contract.md"
MODEL_CONTRACT_PATH = PROJECT_ROOT / "model_layer" / "main_model_contract.md"
VALIDATION_CONTRACT_PATH = PROJECT_ROOT / "validation_layer" / "validation_contract.md"
DECISION_PATH = PROJECT_ROOT / "decision_layer" / "main_pipeline_decision.md"
FORMAL_STATUS_PATH = PROJECT_ROOT / "formal_layer" / "formal_status.md"
FORMAL_CANDIDATES_PATH = PROJECT_ROOT / "formal_layer" / "formal_candidates.csv"


def read_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def fail(message: str) -> None:
    raise SystemExit(f"FAIL: {message}")


def parse_date(value: str) -> datetime:
    value = str(value).strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%y/%m/%d"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    fail(f"cannot parse date: {value}")


def csv_stats(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if not reader.fieldnames:
            fail(f"missing header: {path}")
    date_values = [parse_date(row["日期"]) for row in rows if row.get("日期")]
    if not date_values:
        fail(f"missing date values: {path}")
    return {
        "path": path,
        "rows": len(rows),
        "columns": len(reader.fieldnames or []),
        "latest_date": max(date_values).strftime("%Y-%m-%d"),
        "first_columns": (reader.fieldnames or [])[:12],
    }


def validate_inputs(config: dict) -> dict[str, Path]:
    allowed = config.get("allowed_inputs", {})
    if set(allowed) != {"stock_daily_all", "market_daily", "theme_group"}:
        fail("allowed_inputs must contain exactly stock_daily_all, market_daily, theme_group")
    old_marker = "stock" + "_raw_only" + "_project"
    paths = {name: Path(value) for name, value in allowed.items()}
    for name, path in paths.items():
        if old_marker in str(path):
            fail(f"input points to old project: {name}")
        if not path.exists():
            fail(f"missing input {name}: {path}")
        if name == "theme_group" and PROJECT_ROOT not in path.parents:
            fail("theme_group must be inside this clean project")
    return paths


def write_layer_contracts(stock_stats: dict, market_stats: dict, theme_path: Path) -> None:
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    DATA_STATUS_PATH.write_text(
        "\n".join(
            [
                "# Main Pipeline Data Status",
                "",
                f"- Generated: {generated}",
                f"- Stock latest date: {stock_stats['latest_date']}",
                f"- Market latest date: {market_stats['latest_date']}",
                f"- Stock rows: {stock_stats['rows']}",
                f"- Market rows: {market_stats['rows']}",
                f"- Theme file: `{theme_path}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    LABEL_CONTRACT_PATH.write_text(
        "\n".join(
            [
                "# Label Contract",
                "",
                "- Signal time: after the signal-day close.",
                "- Buy assumption: next trading day open.",
                "- Success: a close in the next 1 to 10 trading days reaches +3 percent before any low in that same window reaches -3 percent from the buy open.",
                "- Conservative tie rule: if the +3 percent close and -3 percent low happen on the same trading day, success is false.",
                "- `old_target_success` keeps the old +3 percent touch rule for comparison only.",
                "- Unfinished: if the future 10 trading day window is incomplete, the sample is tracking-only.",
                "- Same-day market comparison is validation support, not a stock label by itself.",
                "- Episode grouping prevents the same stock from being repeatedly counted as new within one short wave.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    FEATURE_CONTRACT_PATH.write_text(
        "\n".join(
            [
                "# Feature Contract",
                "",
                "Allowed features must be knowable on or before the signal day.",
                "",
                "- Stock price and volume history.",
                "- Institution and margin data history.",
                "- Day-trade data history.",
                "- Market index, volume, breadth, institution, and margin data history.",
                "- Theme group as a categorical feature or validation grouping only.",
                "",
                "Forbidden inputs include future prices, manual conclusions, old report outputs, and post-result explanations.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    MODEL_CONTRACT_PATH.write_text(
        "\n".join(
            [
                "# Main Model Contract",
                "",
                "The project has one formal model route.",
                "",
                "The model may learn four tasks inside one integrated training flow:",
                "",
                "- 10 trading day success.",
                "- Failure risk.",
                "- Same-day relative advantage.",
                "- Episode starting point.",
                "",
                "A raw model score is only a research ranking score. It can be called a success rate only after calibration passes.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    VALIDATION_CONTRACT_PATH.write_text(
        "\n".join(
            [
                "# Validation Contract",
                "",
                "A model can update formal output only after holdout validation passes.",
                "",
                "Required checks:",
                "",
                "- Same-day market baseline comparison.",
                "- Current benchmark comparison.",
                "- Score band direction.",
                "- Failure-risk band direction.",
                "- Monthly stability.",
                "- Stock and industry concentration.",
                "- Maximum Top 3 formal candidates per day.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_formal_files(config: dict, latest_date: str, reason: str) -> None:
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    FORMAL_STATUS_PATH.write_text(
        "\n".join(
            [
                "# Formal Status",
                "",
                f"- Generated: {generated}",
                "- Status: active",
                "- Formal source: `scripts/run_main_pipeline.py`",
                "- Current strategy: retained benchmark",
                f"- Raw data latest date: {latest_date}",
                f"- Result: {config['formal_candidate_default']}",
                f"- Reason: {reason}",
                "- Rule: research outputs cannot update formal candidates directly.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    with FORMAL_CANDIDATES_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "date",
                "stock_id",
                "stock_name",
                "output_type",
                "research_score",
                "calibrated_success_rate",
                "calibration_sample_count",
                "actual_hit_rate",
                "avg_10d_high_close_return",
                "main_basis",
                "main_risk",
                "tracking_status",
            ]
        )


def main() -> None:
    config = read_config()
    paths = validate_inputs(config)
    stock_stats = csv_stats(paths["stock_daily_all"])
    market_stats = csv_stats(paths["market_daily"])
    if stock_stats["latest_date"] != market_stats["latest_date"]:
        fail(
            "stock_daily_all and market_daily latest dates do not match: "
            f"{stock_stats['latest_date']} vs {market_stats['latest_date']}"
        )

    latest_date = stock_stats["latest_date"]
    write_layer_contracts(stock_stats, market_stats, paths["theme_group"])
    reason = (
        "architecture reset completed; no new integrated model is promoted until "
        "the single main model passes validation"
    )
    write_formal_files(config, latest_date, reason)
    DECISION_PATH.write_text(
        "\n".join(
            [
                "# Main Pipeline Decision",
                "",
                f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"- Raw data latest date: {latest_date}",
                "- Decision: keep current formal benchmark.",
                f"- Formal result: {config['formal_candidate_default']}",
                "- Next allowed work: train the single integrated main model through this pipeline only.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print("OK: main pipeline completed")
    print(f"LATEST_DATE: {latest_date}")
    print(f"FORMAL_STATUS: {FORMAL_STATUS_PATH}")
    print(f"FORMAL_CANDIDATES: {FORMAL_CANDIDATES_PATH}")


if __name__ == "__main__":
    main()
