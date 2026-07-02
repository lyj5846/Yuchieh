from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "project_config.json"

PLANNING_DIR = PROJECT_ROOT / "planning_layer"
PLAN_MD_PATH = PLANNING_DIR / "current_model_plan.md"
PLAN_JSON_PATH = PLANNING_DIR / "current_model_plan.json"
AUDIT_PATH = PLANNING_DIR / "planning_audit.md"

CONTRACT_SOURCES = [
    PROJECT_ROOT / "docs" / "architecture_contract.md",
    PROJECT_ROOT / "docs" / "ai_self_learning_contract.md",
    PROJECT_ROOT / "label_layer" / "label_contract.md",
    PROJECT_ROOT / "feature_layer" / "feature_contract.md",
    PROJECT_ROOT / "model_layer" / "main_model_contract.md",
    PROJECT_ROOT / "validation_layer" / "validation_contract.md",
    PROJECT_ROOT / "decision_layer" / "main_pipeline_decision.md",
]

REQUIRED_CANDIDATE_KEYS = [
    "id",
    "hypothesis",
    "why_now",
    "target_label",
    "allowed_inputs",
    "feature_changes",
    "model_changes",
    "validation_checks",
    "pass_criteria",
    "rejection_criteria",
    "expected_outputs",
    "risk_notes",
]


def fail(message: str) -> None:
    raise SystemExit(f"FAIL: {message}")


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_date(value: str) -> datetime:
    value = str(value).strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%y/%m/%d"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    fail(f"cannot parse date: {value}")


def csv_latest_date(path: Path) -> str:
    latest: datetime | None = None
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "日期" not in reader.fieldnames:
            fail(f"missing date column: {path}")
        for row in reader:
            if not row.get("日期"):
                continue
            date = parse_date(row["日期"])
            latest = date if latest is None or date > latest else latest
    if latest is None:
        fail(f"no dated rows: {path}")
    return latest.strftime("%Y-%m-%d")


def validate_inputs(config: dict) -> dict[str, Path]:
    allowed = config.get("allowed_inputs", {})
    if set(allowed) != {"stock_daily_all", "market_daily", "theme_group"}:
        fail("allowed_inputs must contain exactly three approved sources")
    old_marker = "stock" + "_raw_only" + "_project"
    paths = {name: Path(value) for name, value in allowed.items()}
    for name, path in paths.items():
        if not path.exists():
            fail(f"missing input {name}: {path}")
        if old_marker in str(path):
            fail(f"old project path is not allowed: {name}")
        if name == "theme_group" and PROJECT_ROOT not in path.parents:
            fail("theme_group must live inside this clean project")
    return paths


def read_contract_sources() -> dict[str, str]:
    sources: dict[str, str] = {}
    for path in CONTRACT_SOURCES:
        if not path.exists():
            fail(f"missing planning source: {path.relative_to(PROJECT_ROOT)}")
        sources[str(path.relative_to(PROJECT_ROOT))] = path.read_text(encoding="utf-8", errors="ignore")
    return sources


def build_plan(config: dict, data_latest_date: str) -> dict:
    allowed_inputs = list(config["allowed_inputs"].keys())
    candidates = [
        {
            "id": "risk_adjusted_main_model_training_plan",
            "hypothesis": "The single integrated deep-learning route should first screen for train/development-stable features, then retrain on risk_adjusted_10d_success so weak or unstable features do not dilute the useful signal.",
            "why_now": "The data learnability review found stable success, risk-filter, and return-ranking clues, while the full-feature retrain still failed holdout. The next step is feature-screened retraining, not a new model branch.",
            "target_label": [
                "risk_adjusted_10d_success",
                "old_target_success_comparison",
                "failure_risk",
                "same_day_relative_advantage",
                "episode_start",
            ],
            "allowed_inputs": allowed_inputs,
            "feature_changes": [
                "Create signal-day features from stock price, volume, institution, margin, day-trade, market, and theme data.",
                "Screen features using train/development correlation stability from the data learnability review.",
                "Do not use holdout correlations to select features; holdout is validation-only.",
                "Use only data known on or before the signal day.",
                "Keep theme data as a categorical feature or validation group only.",
                "Use future prices only in the label layer to identify +3% close before -3% low.",
            ],
            "model_changes": [
                "Retrain the existing one hidden-layer numpy MLP route in the main model layer.",
                "Keep raw outputs as research ranking scores until calibration passes.",
                "Keep old_target_success only as a comparison field.",
                "Produce at most Top 3 candidates only after validation accepts the route.",
            ],
            "validation_checks": [
                "same_day_market_baseline",
                "current_benchmark_comparison",
                "score_band_direction",
                "risk_band_direction",
                "monthly_stability",
                "stock_and_industry_concentration",
            ],
            "pass_criteria": [
                "Holdout result beats same-day market baseline.",
                "Holdout result is not materially weaker than the current benchmark.",
                "Higher score bands perform better than lower score bands.",
                "Higher risk bands fail more often than lower risk bands.",
                "Candidate count remains at most Top 3 per day.",
            ],
            "rejection_criteria": [
                "Holdout lift is not positive after same-day baseline comparison.",
                "Score bands reverse direction.",
                "Risk bands do not separate failures.",
                "Result depends on a narrow stock, date, or industry group.",
            ],
            "expected_outputs": [
                "main_model_training_spec.md",
                "main_model_feature_screen.csv",
                "main_model_validation_summary.csv",
                "main_model_decision.md",
            ],
            "risk_notes": [
                "The stricter target will lower raw success rate; compare against the new same-day baseline, not the old touch target.",
                "Do not expose raw scores as success rates.",
            ],
        },
        {
            "id": "validation_harness_first_plan",
            "hypothesis": "A stronger validation harness may prevent another cycle of attractive research results that cannot become formal output.",
            "why_now": "The project now has contracts, but planning can still fail if acceptance checks are not executable before training.",
            "target_label": [
                "10_day_success",
                "same_day_relative_advantage",
                "failure_risk",
            ],
            "allowed_inputs": allowed_inputs,
            "feature_changes": [
                "No new model features in this plan.",
                "Build reusable validation datasets from the approved label contract.",
            ],
            "model_changes": [
                "No model training in this plan.",
                "Prepare validation functions that future training must call.",
            ],
            "validation_checks": [
                "baseline_comparison",
                "score_band_direction",
                "risk_band_direction",
                "monthly_stability",
                "concentration_limits",
            ],
            "pass_criteria": [
                "Validation checks can fail intentionally on malformed sample plans.",
                "Validation checks can pass on a well-formed plan fixture.",
            ],
            "rejection_criteria": [
                "Checks are only descriptive and cannot block promotion.",
                "Checks require research outputs as inputs.",
            ],
            "expected_outputs": [
                "validation_harness_spec.md",
                "validation_harness_checks.csv",
            ],
            "risk_notes": [
                "This improves control but does not directly train a model.",
                "Use it if the next implementation risk is governance rather than modeling.",
            ],
        },
        {
            "id": "episode_and_risk_focus_plan",
            "hypothesis": "Focusing on repeated signals and failure separation may reduce duplicate wave recommendations before the full model route is trained.",
            "why_now": "Previous research suggested repeated signals and weak failure separation caused confusion, but this should be integrated rather than exposed as another branch.",
            "target_label": [
                "episode_start",
                "failure_risk",
                "10_day_success",
            ],
            "allowed_inputs": allowed_inputs,
            "feature_changes": [
                "Build episode grouping features from same-stock signal spacing.",
                "Build drawdown and failed-window labels only from completed future windows.",
            ],
            "model_changes": [
                "Train only the episode and risk heads as a contained research plan.",
                "Do not write formal candidates.",
            ],
            "validation_checks": [
                "episode_duplicate_reduction",
                "risk_band_direction",
                "same_day_market_baseline",
                "monthly_stability",
            ],
            "pass_criteria": [
                "Episode grouping reduces duplicate candidates without lowering holdout quality.",
                "Higher risk bands have higher actual failure rate.",
            ],
            "rejection_criteria": [
                "Duplicate reduction removes useful early signals.",
                "Risk bands do not separate outcomes.",
            ],
            "expected_outputs": [
                "episode_risk_plan_summary.md",
                "episode_risk_validation.csv",
            ],
            "risk_notes": [
                "This is narrower than the recommended plan.",
                "It must not become a separate user-facing branch.",
            ],
        },
    ]
    return {
        "plan_id": f"model_plan_{data_latest_date.replace('-', '')}",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_latest_date": data_latest_date,
        "planning_status": "waiting_for_user_confirmation",
        "problem_statement": "The data learnability review found stable signal in the three CSV inputs, but the full-feature risk-adjusted model failed holdout. The next step is feature-screened retraining of the single deep-learning main model.",
        "recommended_experiment_id": "risk_adjusted_main_model_training_plan",
        "confirmation_required": True,
        "experiment_candidates": candidates,
    }


def check_plan_shape(plan: dict) -> None:
    required = {
        "plan_id",
        "generated_at",
        "data_latest_date",
        "planning_status",
        "problem_statement",
        "recommended_experiment_id",
        "confirmation_required",
        "experiment_candidates",
    }
    missing = required - set(plan)
    if missing:
        fail(f"planning output missing keys: {sorted(missing)}")
    candidates = plan["experiment_candidates"]
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= 3:
        fail("planning output must contain 1 to 3 experiment candidates")
    if plan["confirmation_required"] is not True:
        fail("planning output must require confirmation")
    ids = {candidate.get("id") for candidate in candidates}
    if plan["recommended_experiment_id"] not in ids:
        fail("recommended_experiment_id must match an experiment candidate")
    for candidate in candidates:
        missing_candidate = set(REQUIRED_CANDIDATE_KEYS) - set(candidate)
        if missing_candidate:
            fail(f"candidate {candidate.get('id')} missing keys: {sorted(missing_candidate)}")


def write_plan_markdown(plan: dict) -> None:
    recommended = next(
        c for c in plan["experiment_candidates"] if c["id"] == plan["recommended_experiment_id"]
    )
    lines = [
        "# Current Model Plan",
        "",
        f"- Generated: {plan['generated_at']}",
        f"- Data latest date: {plan['data_latest_date']}",
        f"- Status: {plan['planning_status']}",
        "- Confirmation required: true",
        "",
        "## Problem",
        "",
        plan["problem_statement"],
        "",
        "## Recommended Next Step",
        "",
        f"- Experiment id: `{recommended['id']}`",
        f"- Hypothesis: {recommended['hypothesis']}",
        f"- Why now: {recommended['why_now']}",
        "",
        "This is recommended because the data learnability review found stable clues, while the full-feature model diluted them. The main question is whether screened features can improve the existing integrated deep-learning route without creating another model branch.",
        "",
        "## Candidate Experiments",
        "",
    ]
    for candidate in plan["experiment_candidates"]:
        lines.extend(
            [
                f"### {candidate['id']}",
                "",
                f"- Hypothesis: {candidate['hypothesis']}",
                f"- Why now: {candidate['why_now']}",
                f"- Target labels: `{', '.join(candidate['target_label'])}`",
                f"- Allowed inputs: `{', '.join(candidate['allowed_inputs'])}`",
                f"- Expected outputs: `{', '.join(candidate['expected_outputs'])}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Boundaries",
            "",
            "- This plan does not select stocks.",
            "- This plan does not train a model.",
            "- This plan does not update formal output.",
            "- Raw model scores must not be called success rates unless calibration passes.",
            "",
        ]
    )
    PLAN_MD_PATH.write_text("\n".join(lines), encoding="utf-8")


def write_audit(sources: dict[str, str], data_latest_date: str, plan: dict) -> None:
    lines = [
        "# Planning Audit",
        "",
        f"- Generated: {plan['generated_at']}",
        f"- Data latest date: {data_latest_date}",
        f"- Recommended experiment: `{plan['recommended_experiment_id']}`",
        "- Planning output writes only to `planning_layer`.",
        "- Research outputs were intentionally not used as decision inputs.",
        "- Formal output was intentionally not written.",
        "",
        "## Sources Used",
        "",
    ]
    for source in sources:
        lines.append(f"- `{source}`")
    lines.extend(
        [
            "",
            "## Sources Not Used",
            "",
            "- `research_layer`: preserved as research history only.",
            "- `formal_layer`: not modified by planning.",
            "",
            "## Candidate Count",
            "",
            f"- {len(plan['experiment_candidates'])} candidate experiments generated.",
            "- 1 recommended experiment selected.",
            "",
        ]
    )
    AUDIT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    config = load_config()
    paths = validate_inputs(config)
    stock_latest = csv_latest_date(paths["stock_daily_all"])
    market_latest = csv_latest_date(paths["market_daily"])
    if stock_latest != market_latest:
        fail(f"stock and market latest dates differ: {stock_latest} vs {market_latest}")
    sources = read_contract_sources()
    plan = build_plan(config, stock_latest)
    check_plan_shape(plan)
    PLAN_JSON_PATH.write_text(
        json.dumps(plan, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_plan_markdown(plan)
    write_audit(sources, stock_latest, plan)
    print("OK: planning pipeline completed")
    print(f"PLAN_MD: {PLAN_MD_PATH}")
    print(f"PLAN_JSON: {PLAN_JSON_PATH}")
    print(f"AUDIT: {AUDIT_PATH}")


if __name__ == "__main__":
    main()
