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

from run_deep_learning_frequency_expansion_experiment import (  # noqa: E402
    score_datasets,
    strategy_candidates,
)
from run_deep_learning_selection_experiment import (  # noqa: E402
    fmt_pct,
    predict_mlp,
    standardize,
    train_mlp,
)
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
from run_deep_learning_calibrated_candidate_experiment import split_masks  # noqa: E402


EXPERIMENT_DIR = PROJECT_ROOT / "research_layer"
FORMAL_DIR = PROJECT_ROOT / "research_layer" / "formal_write_disabled"

DECISION_PATH = EXPERIMENT_DIR / "integrated_main_model_v2_decision.md"
BACKTEST_PATH = EXPERIMENT_DIR / "integrated_main_model_v2_backtest.csv"
GATE_PATH = EXPERIMENT_DIR / "integrated_main_model_v2_gate_analysis.csv"
CALIBRATION_PATH = EXPERIMENT_DIR / "integrated_main_model_v2_calibration.csv"
FAILURE_PATH = EXPERIMENT_DIR / "integrated_main_model_v2_failure_learning.csv"
LATEST_PATH = EXPERIMENT_DIR / "integrated_main_model_v2_latest.csv"
FORMAL_STATUS_PATH = FORMAL_DIR / "formal_status.md"
FORMAL_CANDIDATES_PATH = FORMAL_DIR / "formal_candidates.csv"


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def day_feature_frame(scored: pd.DataFrame, weights: tuple[float, float, float, float]) -> pd.DataFrame:
    work = apply_score(scored, weights)
    rows = []
    for date, day in work.groupby("日期"):
        ranked = day.sort_values("integrated_decision_score", ascending=False)
        top3 = ranked.head(3)
        rows.append(
            {
                "日期": date,
                "day_success_mean": day["target_success"].mean(),
                "day_return_mean": day["future_10d_high_close_return"].mean(),
                "day_market_strength": day["market_strength_score"].mean(),
                "top1_score": ranked["integrated_decision_score"].iloc[0],
                "top3_score_mean": top3["integrated_decision_score"].mean(),
                "top3_score_gap": ranked["integrated_decision_score"].iloc[0]
                - ranked["integrated_decision_score"].iloc[min(2, len(ranked) - 1)],
                "top3_success_head_mean": top3["success_head"].mean(),
                "top3_rank_head_mean": top3["rank_head"].mean(),
                "top3_risk_head_mean": top3["risk_head"].mean(),
                "top3_episode_head_mean": top3["episode_head"].mean(),
                "top3_success_rate": top3["target_success"].mean(),
                "top3_return": top3["future_10d_high_close_return"].mean(),
                "gate_label": int(
                    (top3["target_success"].mean() - day["target_success"].mean() >= 0.08)
                    and (top3["future_10d_high_close_return"].mean() > day["future_10d_high_close_return"].mean())
                ),
            }
        )
    return pd.DataFrame(rows)


def train_gate_model(dev_days: pd.DataFrame, holdout_days: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    feature_cols = [
        "day_success_mean",
        "day_return_mean",
        "day_market_strength",
        "top1_score",
        "top3_score_mean",
        "top3_score_gap",
        "top3_success_head_mean",
        "top3_rank_head_mean",
        "top3_risk_head_mean",
        "top3_episode_head_mean",
    ]
    if dev_days.empty or len(dev_days["gate_label"].unique()) < 2:
        dev_days = dev_days.copy()
        holdout_days = holdout_days.copy()
        dev_days["gate_score"] = dev_days["top3_score_mean"].rank(pct=True)
        holdout_days["gate_score"] = holdout_days["top3_score_mean"].rank(pct=True)
        return dev_days, holdout_days, "fallback_top3_score_rank"
    X_dev = dev_days[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    X_hold = holdout_days[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    train_mask = np.ones(len(dev_days), dtype=bool)
    X_dev_std, mean, std = standardize(X_dev, train_mask)
    X_hold_std = ((X_hold - mean) / std).astype(np.float32)
    y = dev_days["gate_label"].to_numpy(dtype=np.float32).reshape(-1, 1)
    model, _ = train_mlp(X_dev_std, y, hidden=8, epochs=120, seed=1002)
    dev_out = dev_days.copy()
    holdout_out = holdout_days.copy()
    dev_out["gate_score"] = predict_mlp(X_dev_std, model)[:, 0]
    holdout_out["gate_score"] = predict_mlp(X_hold_std, model)[:, 0]
    return dev_out, holdout_out, "trained_gate_head"


def tune_gate(dev_scored: pd.DataFrame, dev_days: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    rows = []
    for q in [0.20, 0.30, 0.40, 0.50, 0.60]:
        threshold = dev_days["gate_score"].quantile(q)
        allowed = set(dev_days[dev_days["gate_score"] >= threshold]["日期"])
        picked = select_top3_episode_dedup(dev_scored[dev_scored["日期"].isin(allowed)])
        row = summary_row(dev_scored, picked, "development", "integrated_v2_top3", f"gate_q{q}")
        row["gate_quantile"] = q
        row["gate_threshold"] = threshold
        rows.append(row)
    table = pd.DataFrame(rows)
    feasible = table[
        (table["days"] >= 36)
        & (table["success_lift"] >= 0.08)
        & (table["return_lift"] > 0)
        & (table["top_stock_share"] <= 0.20)
        & (table["top_industry_share"] <= 0.50)
    ].copy()
    if feasible.empty:
        feasible = table.copy()
    feasible = feasible.sort_values(["success_lift", "return_lift", "success_rate", "days"], ascending=False)
    return float(feasible.iloc[0]["gate_threshold"]), table


def select_with_gate(scored: pd.DataFrame, gate_days: pd.DataFrame, threshold: float) -> pd.DataFrame:
    allowed = set(gate_days[gate_days["gate_score"] >= threshold]["日期"])
    return select_top3_episode_dedup(scored[scored["日期"].isin(allowed)])


def gate_analysis(days: pd.DataFrame, threshold: float, split: str) -> pd.DataFrame:
    out = days.copy()
    out["gate_pass"] = out["gate_score"] >= threshold
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
                "avg_gate_score": part["gate_score"].mean(),
            }
        )
    return pd.DataFrame(rows)


def v1_summary(config: dict) -> dict:
    path = EXPERIMENT_DIR / "integrated_main_model_backtest.csv"
    if path.exists():
        table = pd.read_csv(path, encoding="utf-8-sig")
        row = table[(table["split"] == "holdout") & (table["strategy"] == "integrated_top3")]
        if not row.empty:
            out = row.iloc[0].to_dict()
            out["strategy"] = "integrated_v1_top3"
            return out
    return {"split": "holdout", "strategy": "integrated_v1_top3", "weights": "missing"}


def latest_candidates(scored: pd.DataFrame, gate_days: pd.DataFrame, threshold: float, passed: bool) -> pd.DataFrame:
    latest_date = scored["日期"].max()
    gate_row = gate_days[gate_days["日期"] == latest_date]
    if gate_row.empty or float(gate_row.iloc[0]["gate_score"]) < threshold:
        return pd.DataFrame(
            columns=[
                "signal_date",
                "stock_id",
                "stock_name",
                "research_score",
                "gate_score",
                "success_head",
                "risk_head",
                "episode_head",
                "rank_head",
                "status",
                "formal_candidate",
            ]
        )
    latest = scored[scored["日期"] == latest_date].copy()
    picked = select_top3_episode_dedup(latest).copy()
    if picked.empty:
        return pd.DataFrame()
    picked["signal_date"] = picked["日期"].dt.strftime("%Y-%m-%d")
    picked["stock_id"] = picked["股票代號"]
    picked["stock_name"] = picked["股票名稱"]
    picked["research_score"] = picked["integrated_decision_score"]
    picked["gate_score"] = float(gate_row.iloc[0]["gate_score"])
    picked["status"] = "追蹤中"
    picked["formal_candidate"] = bool(passed)
    return picked[
        [
            "signal_date",
            "stock_id",
            "stock_name",
            "research_score",
            "gate_score",
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
        result = "integrated deep learning v2 gated Top 3 formal strategy enabled"
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
                    "integrated_v2_gated_score",
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

    weight_search = EXPERIMENT_DIR / "integrated_main_model_weight_search.csv"
    weights = (0.8, 1.0, 0.2, 0.4)
    if weight_search.exists():
        table = pd.read_csv(weight_search, encoding="utf-8-sig")
        if not table.empty and "weights" in table.columns:
            parts = str(table.sort_values(["success_lift", "return_lift"], ascending=False).iloc[0]["weights"]).split(",")
            weights = tuple(float(x) for x in parts)  # type: ignore[assignment]
    weights_text = ",".join(map(str, weights))

    dev_scored = apply_score(dev, weights)
    holdout_scored = apply_score(holdout, weights)
    scoring_scored = apply_score(scoring_scored, weights)
    dev_days_raw = day_feature_frame(dev_scored, weights)
    holdout_days_raw = day_feature_frame(holdout_scored, weights)
    dev_days, holdout_days, gate_status = train_gate_model(dev_days_raw, holdout_days_raw)
    gate_threshold, gate_tuning = tune_gate(dev_scored, dev_days)
    gate_tuning.to_csv(EXPERIMENT_DIR / "integrated_main_model_v2_gate_tuning.csv", index=False, encoding="utf-8-sig")

    dev_picked = select_with_gate(dev_scored, dev_days, gate_threshold)
    holdout_picked = select_with_gate(holdout_scored, holdout_days, gate_threshold)
    fourth = fourth_round_summary(config, holdout_start)
    v1 = v1_summary(config)
    backtest = pd.DataFrame(
        [
            summary_row(dev_scored, dev_picked, "development", "integrated_v2_gated_top3", f"{weights_text};gate={gate_threshold:.6f}"),
            summary_row(holdout_scored, holdout_picked, "holdout", "integrated_v2_gated_top3", f"{weights_text};gate={gate_threshold:.6f}"),
            v1,
            fourth,
        ]
    )
    backtest.to_csv(BACKTEST_PATH, index=False, encoding="utf-8-sig")

    gate_table = pd.concat(
        [
            gate_analysis(dev_days, gate_threshold, "development"),
            gate_analysis(holdout_days, gate_threshold, "holdout"),
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

    selected = backtest[(backtest["split"] == "holdout") & (backtest["strategy"] == "integrated_v2_gated_top3")].iloc[0]
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
    reason = "integrated v2 passed all gates and replaces fourth-round Top 3"
    if not passed:
        reason = "integrated v2 did not pass all gates; keeping fourth-round Top 3"

    scoring_days_raw = day_feature_frame(scoring_scored, weights)
    _, scoring_days, _ = train_gate_model(dev_days_raw, scoring_days_raw)
    latest = latest_candidates(scoring_scored, scoring_days, gate_threshold, passed)
    latest.to_csv(LATEST_PATH, index=False, encoding="utf-8-sig")
    write_formal(passed, reason, latest)

    lines = [
        "# Integrated Main Model v2 Decision",
        "",
        f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "- Data sources: three allowed CSV inputs only",
        "- Goal: add an action gate before integrated Top 3 selection",
        f"- Status: {'integrated_v2_pass' if passed else 'keep_fourth_round'}",
        f"- Gate status: {gate_status}",
        f"- Selected weights: {weights_text}",
        f"- Gate threshold: {gate_threshold:.6f}",
        f"- Training loss: {loss[0]:.6f} -> {loss[-1]:.6f}",
        "",
        "## Holdout Comparison",
        "",
        f"- v2 days: {int(selected['days'])}",
        f"- v2 success rate: {fmt_pct(selected['success_rate'])}",
        f"- v2 success lift: {fmt_pct(selected['success_lift'])}",
        f"- v2 return lift: {fmt_pct(selected['return_lift'])}",
        "",
        "## Gates",
        "",
        f"- Action gate valid: {gate_ok}",
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
        f"- Calibration: `{CALIBRATION_PATH}`",
        f"- Failure learning: `{FAILURE_PATH}`",
        f"- Latest: `{LATEST_PATH}`",
        f"- Formal status: `{FORMAL_STATUS_PATH}`",
    ]
    DECISION_PATH.write_text("\n".join(lines), encoding="utf-8")
    print("OK: integrated deep learning main model v2 completed")
    print(f"STATUS: {'integrated_v2_pass' if passed else 'keep_fourth_round'}")
    print(f"DECISION: {DECISION_PATH}")


if __name__ == "__main__":
    main()

