from __future__ import annotations

import csv
import itertools
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from run_deep_learning_calibrated_candidate_experiment import split_masks  # noqa: E402
from run_deep_learning_selection_experiment import fmt_pct  # noqa: E402
from run_integrated_deep_learning_main_model import (  # noqa: E402
    FOURTH_ROUND_SUCCESS_RATE,
    apply_score,
    band_table,
    fourth_round_summary,
    prepare_data,
    score_multitask,
    select_top3_episode_dedup,
    summary_row,
)
from run_integrated_deep_learning_main_model_v2 import day_feature_frame, v1_summary  # noqa: E402


EXPERIMENT_DIR = PROJECT_ROOT / "research_layer"
FORMAL_DIR = PROJECT_ROOT / "research_layer" / "formal_write_disabled"

DECISION_PATH = EXPERIMENT_DIR / "integrated_main_model_v3_decision.md"
BACKTEST_PATH = EXPERIMENT_DIR / "integrated_main_model_v3_backtest.csv"
GATE_PATH = EXPERIMENT_DIR / "integrated_main_model_v3_gate_analysis.csv"
SEARCH_PATH = EXPERIMENT_DIR / "integrated_main_model_v3_condition_search.csv"
CALIBRATION_PATH = EXPERIMENT_DIR / "integrated_main_model_v3_calibration.csv"
FAILURE_PATH = EXPERIMENT_DIR / "integrated_main_model_v3_failure_learning.csv"
LATEST_PATH = EXPERIMENT_DIR / "integrated_main_model_v3_latest.csv"
FORMAL_STATUS_PATH = FORMAL_DIR / "formal_status.md"
FORMAL_CANDIDATES_PATH = FORMAL_DIR / "formal_candidates.csv"


CONDITION_COLUMNS = [
    "top1_score",
    "top3_score_mean",
    "top3_score_gap",
    "top3_risk_head_mean",
    "top3_rank_head_mean",
]


def load_best_weights() -> tuple[float, float, float, float]:
    path = EXPERIMENT_DIR / "integrated_main_model_weight_search.csv"
    if not path.exists():
        return (0.8, 1.0, 0.2, 0.4)
    table = pd.read_csv(path, encoding="utf-8-sig")
    if table.empty or "weights" not in table.columns:
        return (0.8, 1.0, 0.2, 0.4)
    row = table.sort_values(["success_lift", "return_lift"], ascending=False).iloc[0]
    return tuple(float(x) for x in str(row["weights"]).split(","))  # type: ignore[return-value]


def condition_pass(days: pd.DataFrame, condition: dict[str, float]) -> pd.Series:
    return (
        (days["top1_score"] >= condition["top1_min"])
        & (days["top3_score_mean"] >= condition["top3_mean_min"])
        & (days["top3_score_gap"] >= condition["top3_gap_min"])
        & (days["top3_risk_head_mean"] <= condition["risk_mean_max"])
        & (days["top3_rank_head_mean"] >= condition["rank_mean_min"])
    )


def select_with_condition(scored: pd.DataFrame, days: pd.DataFrame, condition: dict[str, float]) -> pd.DataFrame:
    allowed = set(days.loc[condition_pass(days, condition), "日期"])
    return select_top3_episode_dedup(scored[scored["日期"].isin(allowed)])


def condition_label(condition: dict[str, float]) -> str:
    return (
        f"top1>={condition['top1_min']:.6f};"
        f"top3mean>={condition['top3_mean_min']:.6f};"
        f"gap>={condition['top3_gap_min']:.6f};"
        f"risk<={condition['risk_mean_max']:.6f};"
        f"rank>={condition['rank_mean_min']:.6f}"
    )


def tune_conditions(dev_scored: pd.DataFrame, dev_days: pd.DataFrame) -> tuple[dict[str, float], pd.DataFrame]:
    quantiles = {
        "top1_min": ("top1_score", [0.00, 0.15, 0.25, 0.35]),
        "top3_mean_min": ("top3_score_mean", [0.00, 0.15, 0.25, 0.35]),
        "top3_gap_min": ("top3_score_gap", [0.00, 0.20, 0.35]),
        "risk_mean_max": ("top3_risk_head_mean", [0.55, 0.70, 0.85, 1.00]),
        "rank_mean_min": ("top3_rank_head_mean", [0.00, 0.20, 0.35]),
    }
    rows = []
    for q_top1, q_mean, q_gap, q_risk, q_rank in itertools.product(
        quantiles["top1_min"][1],
        quantiles["top3_mean_min"][1],
        quantiles["top3_gap_min"][1],
        quantiles["risk_mean_max"][1],
        quantiles["rank_mean_min"][1],
    ):
        condition = {
            "top1_min": float(dev_days["top1_score"].quantile(q_top1)),
            "top3_mean_min": float(dev_days["top3_score_mean"].quantile(q_mean)),
            "top3_gap_min": float(dev_days["top3_score_gap"].quantile(q_gap)),
            "risk_mean_max": float(dev_days["top3_risk_head_mean"].quantile(q_risk)),
            "rank_mean_min": float(dev_days["top3_rank_head_mean"].quantile(q_rank)),
        }
        picked = select_with_condition(dev_scored, dev_days, condition)
        row = summary_row(dev_scored, picked, "development", "integrated_v3_hard_gate_top3", condition_label(condition))
        row.update(
            {
                "top1_q": q_top1,
                "top3_mean_q": q_mean,
                "gap_q": q_gap,
                "risk_q": q_risk,
                "rank_q": q_rank,
                **condition,
            }
        )
        rows.append(row)
    table = pd.DataFrame(rows)
    feasible = table[
        (table["days"] >= 36)
        & (table["success_rate"] >= FOURTH_ROUND_SUCCESS_RATE - 0.03)
        & (table["success_lift"] >= 0.08)
        & (table["return_lift"] > 0)
        & (table["top_stock_share"] <= 0.20)
        & (table["top_industry_share"] <= 0.50)
    ].copy()
    if feasible.empty:
        feasible = table.copy()
    feasible = feasible.sort_values(["success_lift", "return_lift", "success_rate", "days"], ascending=False)
    best = feasible.iloc[0]
    condition = {
        "top1_min": float(best["top1_min"]),
        "top3_mean_min": float(best["top3_mean_min"]),
        "top3_gap_min": float(best["top3_gap_min"]),
        "risk_mean_max": float(best["risk_mean_max"]),
        "rank_mean_min": float(best["rank_mean_min"]),
    }
    return condition, table


def hard_gate_analysis(days: pd.DataFrame, condition: dict[str, float], split: str) -> pd.DataFrame:
    out = days.copy()
    out["gate_pass"] = condition_pass(out, condition)
    rows = []
    for passed, part in out.groupby("gate_pass"):
        rows.append(
            {
                "split": split,
                "gate_pass": bool(passed),
                "days": len(part),
                "actual_gate_label_rate": part["gate_label"].mean(),
                "top3_success_rate": part["top3_success_rate"].mean(),
                "same_day_success_rate": part["day_success_mean"].mean(),
                "success_lift": (part["top3_success_rate"] - part["day_success_mean"]).mean(),
                "top3_return": part["top3_return"].mean(),
                "same_day_return": part["day_return_mean"].mean(),
                "return_lift": (part["top3_return"] - part["day_return_mean"]).mean(),
                "avg_top1_score": part["top1_score"].mean(),
                "avg_top3_score_mean": part["top3_score_mean"].mean(),
                "avg_risk_mean": part["top3_risk_head_mean"].mean(),
                "avg_rank_mean": part["top3_rank_head_mean"].mean(),
            }
        )
    return pd.DataFrame(rows)


def previous_summary(path_name: str, strategy_name: str) -> dict:
    path = EXPERIMENT_DIR / path_name
    if not path.exists():
        return {"split": "holdout", "strategy": strategy_name, "weights": "missing"}
    table = pd.read_csv(path, encoding="utf-8-sig")
    row = table[(table["split"] == "holdout") & (table["strategy"].str.contains(strategy_name.split("_")[1], na=False))]
    if row.empty:
        row = table[table["split"] == "holdout"].head(1)
    if row.empty:
        return {"split": "holdout", "strategy": strategy_name, "weights": "missing"}
    out = row.iloc[0].to_dict()
    out["strategy"] = strategy_name
    return out


def latest_candidates(scored: pd.DataFrame, days: pd.DataFrame, condition: dict[str, float], passed: bool) -> pd.DataFrame:
    latest_date = scored["日期"].max()
    latest_day = days[days["日期"] == latest_date]
    if latest_day.empty or not bool(condition_pass(latest_day, condition).iloc[0]):
        return pd.DataFrame(
            columns=[
                "signal_date",
                "stock_id",
                "stock_name",
                "research_score",
                "success_head",
                "risk_head",
                "episode_head",
                "rank_head",
                "status",
                "formal_candidate",
            ]
        )
    picked = select_top3_episode_dedup(scored[scored["日期"] == latest_date]).copy()
    picked["signal_date"] = picked["日期"].dt.strftime("%Y-%m-%d")
    picked["stock_id"] = picked["股票代號"]
    picked["stock_name"] = picked["股票名稱"]
    picked["research_score"] = picked["integrated_decision_score"]
    picked["status"] = "追蹤中"
    picked["formal_candidate"] = bool(passed)
    return picked[
        [
            "signal_date",
            "stock_id",
            "stock_name",
            "research_score",
            "success_head",
            "risk_head",
            "episode_head",
            "rank_head",
            "status",
            "formal_candidate",
        ]
    ]


def write_formal(passed: bool, reason: str, latest: pd.DataFrame) -> None:
    result = "deep learning Top 3 formal strategy remains enabled"
    if passed:
        result = "integrated deep learning v3 hard-gated Top 3 formal strategy enabled"
    if latest.empty:
        result += "; current latest date has no candidate"
    FORMAL_STATUS_PATH.write_text(
        "\n".join(["# Formal Status", "", "- Status: active", f"- Result: {result}", f"- Reason: {reason}", ""]),
        encoding="utf-8",
    )
    with FORMAL_CANDIDATES_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "date",
                "stock_id",
                "stock_name",
                "estimated_success_probability",
                "calibration_sample_count",
                "actual_hit_rate",
                "avg_10d_high_close_return",
                "main_basis",
                "main_risk",
            ]
        )
        if not passed:
            return
        for _, row in latest.iterrows():
            writer.writerow(
                [
                    row["signal_date"],
                    row["stock_id"],
                    row["stock_name"],
                    "",
                    "",
                    "",
                    "",
                    "integrated_v3_hard_gate_score",
                    "tracking only until 10 trading days complete",
                ]
            )


def main() -> None:
    EXPERIMENT_DIR.mkdir(exist_ok=True)
    FORMAL_DIR.mkdir(exist_ok=True)
    config, train_data, train_features, scoring_data, scoring_features = prepare_data()
    train_scored, scoring_scored, loss = score_multitask(train_data, train_features, scoring_data, scoring_features, config)
    masks = split_masks(train_scored, config)
    dev = train_scored[masks["development"]].copy()
    holdout = train_scored[masks["holdout"]].copy()
    holdout_start = pd.Timestamp(config["time_split"]["holdout_start"])

    weights = load_best_weights()
    weights_text = ",".join(map(str, weights))
    dev_scored = apply_score(dev, weights)
    holdout_scored = apply_score(holdout, weights)
    scoring_scored = apply_score(scoring_scored, weights)
    dev_days = day_feature_frame(dev_scored, weights)
    holdout_days = day_feature_frame(holdout_scored, weights)
    scoring_days = day_feature_frame(scoring_scored, weights)

    condition, search = tune_conditions(dev_scored, dev_days)
    search.to_csv(SEARCH_PATH, index=False, encoding="utf-8-sig")
    dev_picked = select_with_condition(dev_scored, dev_days, condition)
    holdout_picked = select_with_condition(holdout_scored, holdout_days, condition)
    v3_dev = summary_row(dev_scored, dev_picked, "development", "integrated_v3_hard_gate_top3", condition_label(condition))
    v3_hold = summary_row(holdout_scored, holdout_picked, "holdout", "integrated_v3_hard_gate_top3", condition_label(condition))
    v2 = previous_summary("integrated_main_model_v2_backtest.csv", "integrated_v2_gated_top3")
    v1 = v1_summary(config)
    fourth = fourth_round_summary(config, holdout_start)
    backtest = pd.DataFrame([v3_dev, v3_hold, v2, v1, fourth])
    backtest.to_csv(BACKTEST_PATH, index=False, encoding="utf-8-sig")

    gate_table = pd.concat(
        [
            hard_gate_analysis(dev_days, condition, "development"),
            hard_gate_analysis(holdout_days, condition, "holdout"),
        ],
        ignore_index=True,
    )
    gate_table.to_csv(GATE_PATH, index=False, encoding="utf-8-sig")
    calibration = pd.concat(
        [
            band_table(holdout_scored, "integrated_decision_score", "target_success", "holdout"),
            band_table(holdout_scored, "success_head", "target_success", "holdout"),
        ],
        ignore_index=True,
    )
    calibration.to_csv(CALIBRATION_PATH, index=False, encoding="utf-8-sig")
    failure = band_table(holdout_scored, "risk_head", "risk_label", "holdout")
    failure.to_csv(FAILURE_PATH, index=False, encoding="utf-8-sig")

    selected = v3_hold
    gate_hold = gate_table[(gate_table["split"] == "holdout") & (gate_table["gate_pass"] == True)]
    gate_fail = gate_table[(gate_table["split"] == "holdout") & (gate_table["gate_pass"] == False)]
    gate_ok = bool(
        not gate_hold.empty
        and not gate_fail.empty
        and gate_hold.iloc[0]["success_lift"] > gate_fail.iloc[0]["success_lift"]
        and gate_hold.iloc[0]["return_lift"] > gate_fail.iloc[0]["return_lift"]
    )
    cal_sorted = calibration[calibration["score_col"] == "integrated_decision_score"].sort_values("band")
    fail_sorted = failure.sort_values("band")
    calibration_ok = bool(len(cal_sorted) >= 2 and cal_sorted.iloc[-1]["actual_rate"] >= cal_sorted.iloc[0]["actual_rate"])
    risk_ok = bool(len(fail_sorted) >= 2 and fail_sorted.iloc[-1]["actual_rate"] >= fail_sorted.iloc[0]["actual_rate"])
    month_count = holdout_picked["日期"].dt.strftime("%Y-%m").nunique() if not holdout_picked.empty else 0
    passed = bool(
        selected["days"] >= 36
        and selected["success_rate"] >= FOURTH_ROUND_SUCCESS_RATE - 0.03
        and selected["success_lift"] >= 0.08
        and selected["return_lift"] > 0
        and gate_ok
        and calibration_ok
        and risk_ok
        and selected["top_stock_share"] <= 0.20
        and selected["top_industry_share"] <= 0.50
        and month_count >= 2
    )
    reason = "integrated v3 passed all gates and replaces fourth-round Top 3"
    if not passed:
        reason = "integrated v3 did not pass all gates; keeping fourth-round Top 3"

    latest = latest_candidates(scoring_scored, scoring_days, condition, passed)
    latest.to_csv(LATEST_PATH, index=False, encoding="utf-8-sig")
    write_formal(passed, reason, latest)

    lines = [
        "# Integrated Main Model v3 Decision",
        "",
        f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "- Data sources: three allowed CSV inputs only",
        "- Goal: replace learned action gate with stable hard conditions",
        f"- Status: {'integrated_v3_pass' if passed else 'keep_fourth_round'}",
        f"- Selected weights: {weights_text}",
        f"- Hard condition: {condition_label(condition)}",
        f"- Training loss: {loss[0]:.6f} -> {loss[-1]:.6f}",
        "",
        "## Holdout Comparison",
        "",
        f"- v3 days: {int(selected['days'])}",
        f"- v3 success rate: {fmt_pct(selected['success_rate'])}",
        f"- v3 success lift: {fmt_pct(selected['success_lift'])}",
        f"- v3 return lift: {fmt_pct(selected['return_lift'])}",
        "",
        "## Gates",
        "",
        f"- Hard gate valid: {gate_ok}",
        f"- Calibration ordering valid: {calibration_ok}",
        f"- Failure risk ordering valid: {risk_ok}",
        f"- Top stock share: {fmt_pct(selected['top_stock_share'])}",
        f"- Top industry share: {fmt_pct(selected['top_industry_share'])}",
        f"- Holdout active months: {month_count}",
        "",
        "## Decision",
        "",
        f"- {reason}.",
        "",
        "## Output Files",
        "",
        f"- Backtest: `{BACKTEST_PATH}`",
        f"- Gate analysis: `{GATE_PATH}`",
        f"- Condition search: `{SEARCH_PATH}`",
        f"- Calibration: `{CALIBRATION_PATH}`",
        f"- Failure learning: `{FAILURE_PATH}`",
        f"- Latest: `{LATEST_PATH}`",
        f"- Formal status: `{FORMAL_STATUS_PATH}`",
    ]
    DECISION_PATH.write_text("\n".join(lines), encoding="utf-8")
    print("OK: integrated deep learning main model v3 completed")
    print(f"STATUS: {'integrated_v3_pass' if passed else 'keep_fourth_round'}")
    print(f"DECISION: {DECISION_PATH}")


if __name__ == "__main__":
    main()

