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
MAIN_MODEL_DECISION_PATH = PROJECT_ROOT / "decision_layer" / "main_model_decision.json"
MAIN_MODEL_SCORES_PATH = PROJECT_ROOT / "model_layer" / "main_model_scores.csv"
MAIN_MODEL_VALIDATION_SUMMARY_PATH = PROJECT_ROOT / "validation_layer" / "main_model_validation_summary.csv"


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


def parse_float(value: str | int | float | None) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt_float(value: str | int | float | None, digits: int = 6) -> str:
    parsed = parse_float(value)
    if parsed is None:
        return ""
    return f"{parsed:.{digits}f}"


def fmt_pct(value: str | int | float | None) -> str:
    parsed = parse_float(value)
    if parsed is None:
        return "N/A"
    return f"{parsed:.2%}"


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
                "- Success: a close in the next 1 to 10 trading days reaches +3 percent from the buy open.",
                "- Drawdown side risk: any -3 percent low is a risk label, not an automatic failure.",
                "- If price first reaches -3 percent low and later reaches +3 percent close, primary success remains true.",
                "- `risk_adjusted_10d_success` is kept only as a hard-risk comparison field.",
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
                "- Candidate-region Top 3 success lift and return lift.",
                "- Score band direction as calibration diagnostics, not a hard promotion blocker.",
                "- Failure-risk band direction.",
                "- Monthly stability.",
                "- Stock and industry concentration.",
                "- Maximum Top 3 formal candidates per day.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def read_main_model_decision() -> dict:
    if not MAIN_MODEL_DECISION_PATH.exists():
        return {}
    with MAIN_MODEL_DECISION_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def approved_main_model(decision: dict) -> bool:
    return bool(
        decision
        and decision.get("formal_approved") is True
        and decision.get("status") == "passed_holdout_validation"
        and decision.get("candidate_region_validation_ok") is True
    )


def latest_model_candidates(decision: dict, latest_date: str) -> list[dict]:
    if not MAIN_MODEL_SCORES_PATH.exists():
        fail("approved main model is missing model_layer/main_model_scores.csv")
    with MAIN_MODEL_SCORES_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader if row.get("日期") == latest_date]
    if not rows:
        fail(f"main model scores do not contain latest raw data date: {latest_date}")
    for row in rows:
        row["_score"] = parse_float(row.get("integrated_research_score")) or float("-inf")
    gate = parse_float(decision.get("selected_gate"))
    if gate is not None and max(row["_score"] for row in rows) < gate:
        return []
    rows.sort(key=lambda row: row["_score"], reverse=True)
    return rows[:3]


def main_model_holdout_summary() -> dict:
    if not MAIN_MODEL_VALIDATION_SUMMARY_PATH.exists():
        return {}
    with MAIN_MODEL_VALIDATION_SUMMARY_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("split") == "holdout" and row.get("strategy") == "integrated_main_top3":
                return row
    return {}


def write_formal_files(config: dict, latest_date: str, reason: str, decision: dict | None = None, candidates: list[dict] | None = None) -> None:
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    active = bool(candidates)
    current_strategy = "single integrated main model Top3" if active else "retained benchmark"
    result = "正式候選已產生" if active else config["formal_candidate_default"]
    FORMAL_STATUS_PATH.write_text(
        "\n".join(
            [
                "# Formal Status",
                "",
                f"- Generated: {generated}",
                "- Status: active",
                "- Formal source: `scripts/run_main_pipeline.py`",
                f"- Current strategy: {current_strategy}",
                f"- Raw data latest date: {latest_date}",
                f"- Result: {result}",
                f"- Reason: {reason}",
                "- Rule: training outputs cannot update formal candidates directly.",
                "- Score note: research_score is a ranking score, not a calibrated probability.",
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
        if active:
            assert decision is not None
            holdout_summary = main_model_holdout_summary()
            basis = (
                "candidate-region Top3 passed holdout: "
                f"success_lift={fmt_pct(decision.get('holdout_success_lift'))}, "
                f"return_lift={fmt_pct(decision.get('holdout_return_lift'))}"
            )
            risk = (
                "research score is not calibrated probability; latest signal is tracking-only"
            )
            if decision.get("score_order_ok") is False:
                risk += "; all-row score-band ordering remains diagnostic warning"
            for row in candidates or []:
                writer.writerow(
                    [
                        latest_date,
                        row.get("股票代號", ""),
                        row.get("股票名稱", ""),
                        "formal_candidate",
                        fmt_float(row.get("integrated_research_score")),
                        "",
                        "",
                        fmt_float(decision.get("holdout_success_rate")),
                        fmt_float(holdout_summary.get("avg_10d_high_close_return")),
                        basis,
                        risk,
                        "tracking",
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
    main_model_decision = read_main_model_decision()
    if approved_main_model(main_model_decision):
        candidates = latest_model_candidates(main_model_decision, latest_date)
        if candidates:
            reason = "single main model passed candidate-region holdout validation"
        else:
            reason = "main model passed validation, but latest date did not pass the selected score gate"
        write_formal_files(config, latest_date, reason, main_model_decision, candidates)
        decision_line = "promote validated single main model."
        formal_result = "formal candidates written" if candidates else config["formal_candidate_default"]
    else:
        reason = (
            "architecture reset completed; no new integrated model is promoted until "
            "the single main model passes validation"
        )
        write_formal_files(config, latest_date, reason)
        decision_line = "keep current formal benchmark."
        formal_result = config["formal_candidate_default"]
    DECISION_PATH.write_text(
        "\n".join(
            [
                "# Main Pipeline Decision",
                "",
                f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"- Raw data latest date: {latest_date}",
                f"- Decision: {decision_line}",
                f"- Formal result: {formal_result}",
                "- Next allowed work: track formal candidates; retrain only through the single main model pipeline.",
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
