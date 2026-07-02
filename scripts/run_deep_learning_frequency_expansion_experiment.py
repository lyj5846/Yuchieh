from __future__ import annotations

import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from run_deep_learning_calibrated_candidate_experiment import (  # noqa: E402
    assign_bands,
    build_scoring_dataset,
    dev_band_edges,
    load_config,
    read_csv,
    split_masks,
)
from run_deep_learning_selection_experiment import (  # noqa: E402
    build_dataset,
    fmt_pct,
    sequence_features,
    standardize,
    train_mlp,
    validate_inputs,
)


EXPERIMENT_DIR = PROJECT_ROOT / "research_layer"
FORMAL_DIR = PROJECT_ROOT / "research_layer" / "formal_write_disabled"

DECISION_PATH = EXPERIMENT_DIR / "deep_learning_frequency_expansion_decision.md"
BACKTEST_PATH = EXPERIMENT_DIR / "deep_learning_frequency_expansion_backtest.csv"
BANDS_PATH = EXPERIMENT_DIR / "deep_learning_frequency_expansion_bands.csv"
LATEST_PATH = EXPERIMENT_DIR / "deep_learning_frequency_expansion_latest.csv"
FORMAL_STATUS_PATH = FORMAL_DIR / "formal_status.md"
FORMAL_CANDIDATES_PATH = FORMAL_DIR / "formal_candidates.csv"


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def score_datasets(config: dict) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, dict[str, float]]:
    paths = validate_inputs(config)
    stock = read_csv(paths["stock_daily_all"])
    market = read_csv(paths["market_daily"])
    theme = read_csv(paths["theme_group"])

    train_data = build_dataset(stock, market, theme)
    scoring_data = build_scoring_dataset(stock, market, theme)

    train_features, _ = sequence_features(train_data, lookback=20)
    train_data = train_data[train_features["has_full_20d_history"] == 1].copy()
    train_features = train_features.loc[train_data.index].drop(columns=["has_full_20d_history"])
    scoring_features, _ = sequence_features(scoring_data, lookback=20)
    scoring_data = scoring_data[scoring_features["has_full_20d_history"] == 1].copy()
    scoring_features = scoring_features.loc[scoring_data.index].drop(columns=["has_full_20d_history"])
    scoring_features = scoring_features.reindex(columns=train_features.columns, fill_value=0.0)

    masks = split_masks(train_data, config)
    train_mask = masks["train"].to_numpy()
    x_all, mean, std = standardize(train_features.to_numpy(dtype=np.float32), train_mask)
    y_abs = train_data["target_success"].to_numpy(dtype=np.float32).reshape(-1, 1)
    model, loss = train_mlp(x_all[train_mask], y_abs[train_mask], seed=11)

    train_data["dl_score"] = sigmoid(
        (np.maximum(x_all @ model["w1"] + model["b1"], 0.0) @ model["w2"] + model["b2"])[:, 0]
    )
    scoring_x = ((scoring_features.to_numpy(dtype=np.float32) - mean) / std).astype(np.float32)
    scoring_data["dl_score"] = sigmoid(
        (np.maximum(scoring_x @ model["w1"] + model["b1"], 0.0) @ model["w2"] + model["b2"])[:, 0]
    )
    return train_data, scoring_data, loss, {"feature_count": float(train_features.shape[1])}


def add_day_ranks(df: pd.DataFrame, score_col: str = "dl_score") -> pd.DataFrame:
    out = df.copy()
    out["day_rank"] = out.groupby("日期")[score_col].rank(method="first", ascending=False)
    out["day_rank_pct"] = out.groupby("日期")[score_col].rank(pct=True, method="average")
    out["day_top3_score"] = out.groupby("日期")[score_col].transform(
        lambda s: s.sort_values(ascending=False).iloc[min(2, len(s) - 1)]
    )
    out["score_gap_to_top3"] = out[score_col] - out["day_top3_score"]
    return out


def strategy_candidates(df: pd.DataFrame, strategy: str, band4_min: float, band3_min: float) -> pd.DataFrame:
    work = add_day_ranks(df)
    if strategy == "strict_gate_top3":
        return work[work["market_strength_score"] >= 0.55].sort_values(["日期", "dl_score"], ascending=[True, False]).groupby("日期").head(3)
    if strategy == "strict_gate_top5":
        return work[work["market_strength_score"] >= 0.55].sort_values(["日期", "dl_score"], ascending=[True, False]).groupby("日期").head(5)
    if strategy == "soft_gate_top3":
        eligible = work[(work["market_strength_score"] >= 0.45) & (work["dl_score"] >= band4_min)]
        return eligible.sort_values(["日期", "dl_score"], ascending=[True, False]).groupby("日期").head(3)
    if strategy == "no_gate_exception_top1":
        eligible = work[(work["market_strength_score"] < 0.55) & (work["dl_score"] >= band4_min)]
        return eligible.sort_values(["日期", "dl_score"], ascending=[True, False]).groupby("日期").head(1)
    if strategy == "adaptive_gate_top3":
        strong = work[work["market_strength_score"] >= 0.55].sort_values(["日期", "dl_score"], ascending=[True, False]).groupby("日期").head(3)
        weak_exception = work[(work["market_strength_score"] < 0.55) & (work["dl_score"] >= band4_min)].sort_values(["日期", "dl_score"], ascending=[True, False]).groupby("日期").head(1)
        return pd.concat([strong, weak_exception], ignore_index=False)
    if strategy == "adaptive_gate_top5":
        strong5 = work[(work["market_strength_score"] >= 0.65) & (work["dl_score"] >= band3_min)].sort_values(["日期", "dl_score"], ascending=[True, False]).groupby("日期").head(5)
        normal3 = work[(work["market_strength_score"] >= 0.55) & (work["market_strength_score"] < 0.65)].sort_values(["日期", "dl_score"], ascending=[True, False]).groupby("日期").head(3)
        weak_exception = work[(work["market_strength_score"] < 0.55) & (work["dl_score"] >= band4_min)].sort_values(["日期", "dl_score"], ascending=[True, False]).groupby("日期").head(1)
        return pd.concat([strong5, normal3, weak_exception], ignore_index=False)
    raise ValueError(f"unknown strategy: {strategy}")


def strategy_summary(df: pd.DataFrame, picked: pd.DataFrame, split: str, strategy: str) -> dict:
    if picked.empty:
        return {
            "split": split,
            "strategy": strategy,
            "days": 0,
            "rows": 0,
            "avg_daily_candidates": 0,
            "success_rate": float("nan"),
            "same_day_baseline_success_rate": float("nan"),
            "success_lift": float("nan"),
            "avg_10d_high_close_return": float("nan"),
            "same_day_baseline_avg_return": float("nan"),
            "return_lift": float("nan"),
            "top_stock_share": float("nan"),
            "top_industry_share": float("nan"),
            "added_days_vs_strict": 0,
        }
    base = df.groupby("日期").agg(
        day_success=("target_success", "mean"),
        day_return=("future_10d_high_close_return", "mean"),
    )
    chosen = picked.groupby("日期").agg(
        pick_success=("target_success", "mean"),
        pick_return=("future_10d_high_close_return", "mean"),
    )
    joined = chosen.join(base, how="inner")
    strict_days = set(strategy_candidates(df, "strict_gate_top3", np.inf, np.inf)["日期"].unique())
    picked_days = set(picked["日期"].unique())
    return {
        "split": split,
        "strategy": strategy,
        "days": picked["日期"].nunique(),
        "rows": len(picked),
        "avg_daily_candidates": len(picked) / max(picked["日期"].nunique(), 1),
        "success_rate": picked["target_success"].mean(),
        "same_day_baseline_success_rate": joined["day_success"].mean(),
        "success_lift": joined["pick_success"].mean() - joined["day_success"].mean(),
        "avg_10d_high_close_return": picked["future_10d_high_close_return"].mean(),
        "same_day_baseline_avg_return": joined["day_return"].mean(),
        "return_lift": joined["pick_return"].mean() - joined["day_return"].mean(),
        "top_stock_share": picked["股票代號"].value_counts(normalize=True).iloc[0],
        "top_industry_share": picked["主分類"].fillna("unknown").value_counts(normalize=True).iloc[0],
        "added_days_vs_strict": len(picked_days - strict_days),
    }


def added_candidate_summary(df: pd.DataFrame, candidate: pd.DataFrame, strict: pd.DataFrame, split: str, strategy: str) -> dict:
    if candidate.empty:
        return {
            "split": split,
            "strategy": f"{strategy}_added_only",
            "days": 0,
            "rows": 0,
            "success_rate": float("nan"),
            "same_day_baseline_success_rate": float("nan"),
            "success_lift": float("nan"),
            "avg_10d_high_close_return": float("nan"),
            "same_day_baseline_avg_return": float("nan"),
            "return_lift": float("nan"),
        }
    key_cols = ["日期", "股票代號"]
    strict_keys = set(map(tuple, strict[key_cols].to_numpy()))
    added = candidate[~candidate[key_cols].apply(tuple, axis=1).isin(strict_keys)].copy()
    if added.empty:
        return {
            "split": split,
            "strategy": f"{strategy}_added_only",
            "days": 0,
            "rows": 0,
            "success_rate": float("nan"),
            "same_day_baseline_success_rate": float("nan"),
            "success_lift": float("nan"),
            "avg_10d_high_close_return": float("nan"),
            "same_day_baseline_avg_return": float("nan"),
            "return_lift": float("nan"),
        }
    base = df.groupby("日期").agg(
        day_success=("target_success", "mean"),
        day_return=("future_10d_high_close_return", "mean"),
    )
    chosen = added.groupby("日期").agg(
        pick_success=("target_success", "mean"),
        pick_return=("future_10d_high_close_return", "mean"),
    )
    joined = chosen.join(base, how="inner")
    return {
        "split": split,
        "strategy": f"{strategy}_added_only",
        "days": added["日期"].nunique(),
        "rows": len(added),
        "success_rate": added["target_success"].mean(),
        "same_day_baseline_success_rate": joined["day_success"].mean(),
        "success_lift": joined["pick_success"].mean() - joined["day_success"].mean(),
        "avg_10d_high_close_return": added["future_10d_high_close_return"].mean(),
        "same_day_baseline_avg_return": joined["day_return"].mean(),
        "return_lift": joined["pick_return"].mean() - joined["day_return"].mean(),
    }


def band_table(picked: pd.DataFrame, split: str, strategy: str, edges: np.ndarray) -> pd.DataFrame:
    if picked.empty:
        return pd.DataFrame()
    work = picked.copy()
    work["band"] = assign_bands(work["dl_score"], edges)
    rows = []
    for band, part in work.groupby("band", observed=False):
        if part.empty:
            continue
        rows.append(
            {
                "split": split,
                "strategy": strategy,
                "band": str(band),
                "rows": len(part),
                "avg_score": part["dl_score"].mean(),
                "actual_success_rate": part["target_success"].mean(),
                "avg_10d_high_close_return": part["future_10d_high_close_return"].mean(),
            }
        )
    return pd.DataFrame(rows)


def latest_candidates(scored: pd.DataFrame, strategy: str, band4_min: float, band3_min: float, passed_strategy: str | None) -> pd.DataFrame:
    latest_date = scored["日期"].max()
    latest = scored[scored["日期"] == latest_date].copy()
    picked = strategy_candidates(latest, strategy, band4_min, band3_min).copy()
    if picked.empty:
        return pd.DataFrame(
            columns=[
                "signal_date",
                "stock_id",
                "stock_name",
                "candidate_type",
                "research_score",
                "market_strength_score",
                "status",
                "formal_candidate",
            ]
        )
    picked = add_day_ranks(picked)
    picked["signal_date"] = picked["日期"].dt.strftime("%Y-%m-%d")
    picked["stock_id"] = picked["股票代號"]
    picked["stock_name"] = picked["股票名稱"]
    picked["candidate_type"] = np.where(picked["day_rank"] <= 3, "core", "extension")
    picked["research_score"] = picked["dl_score"]
    picked["status"] = "追蹤中"
    picked["formal_candidate"] = strategy == passed_strategy
    return picked[
        [
            "signal_date",
            "stock_id",
            "stock_name",
            "candidate_type",
            "research_score",
            "market_strength_score",
            "status",
            "formal_candidate",
        ]
    ]


def write_formal(config: dict, status: str, reason: str, tracking: pd.DataFrame) -> None:
    active = status in {"frequency_expansion_pass", "quantity_expansion_pass", "keep_strict_top3"}
    result = {
        "frequency_expansion_pass": "deep learning frequency-expanded formal strategy enabled",
        "quantity_expansion_pass": "deep learning quantity-expanded formal strategy enabled",
        "keep_strict_top3": "deep learning Top 3 formal strategy remains enabled",
        "rejected": config["formal_candidate_default"],
    }.get(status, config["formal_candidate_default"])
    if tracking.empty and active:
        result += "; current latest date has no candidate"
    FORMAL_STATUS_PATH.write_text(
        "\n".join(
            [
                "# Formal Status",
                "",
                f"- Status: {'active' if active else 'not active'}",
                f"- Result: {result}",
                f"- Reason: {reason}",
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
                "estimated_success_probability",
                "calibration_sample_count",
                "actual_hit_rate",
                "avg_10d_high_close_return",
                "main_basis",
                "main_risk",
            ]
        )
        if not active:
            return
        for _, row in tracking.iterrows():
            writer.writerow(
                [
                    row["signal_date"],
                    row["stock_id"],
                    row["stock_name"],
                    "",
                    "",
                    "",
                    "",
                    row["candidate_type"],
                    "tracking only until 10 trading days complete",
                ]
            )


def main() -> None:
    EXPERIMENT_DIR.mkdir(exist_ok=True)
    FORMAL_DIR.mkdir(exist_ok=True)
    config = load_config()
    train_data, scoring_data, loss, meta = score_datasets(config)
    masks = split_masks(train_data, config)
    dev = train_data.loc[masks["development"]].copy()
    holdout = train_data.loc[masks["holdout"]].copy()

    edges = dev_band_edges(dev["dl_score"], bins=4)
    band4_min = float(edges[-2])
    band3_min = float(edges[-3])
    strategies = [
        "strict_gate_top3",
        "strict_gate_top5",
        "soft_gate_top3",
        "no_gate_exception_top1",
        "adaptive_gate_top3",
        "adaptive_gate_top5",
    ]

    summary_rows = []
    band_rows = []
    for split, part in [("development", dev), ("holdout", holdout)]:
        strict = strategy_candidates(part, "strict_gate_top3", band4_min, band3_min)
        for strategy in strategies:
            picked = strategy_candidates(part, strategy, band4_min, band3_min)
            summary_rows.append(strategy_summary(part, picked, split, strategy))
            if strategy != "strict_gate_top3":
                summary_rows.append(added_candidate_summary(part, picked, strict, split, strategy))
            bt = band_table(picked, split, strategy, edges)
            if not bt.empty:
                band_rows.append(bt)

    backtest = pd.DataFrame(summary_rows)
    bands = pd.concat(band_rows, ignore_index=True) if band_rows else pd.DataFrame()
    backtest.to_csv(BACKTEST_PATH, index=False, encoding="utf-8-sig")
    bands.to_csv(BANDS_PATH, index=False, encoding="utf-8-sig")

    holdout_summary = backtest[(backtest["split"] == "holdout") & (~backtest["strategy"].str.endswith("_added_only"))].copy()
    strict_row = holdout_summary[holdout_summary["strategy"] == "strict_gate_top3"].iloc[0]
    dev_summary = backtest[(backtest["split"] == "development") & (~backtest["strategy"].str.endswith("_added_only"))].copy()

    candidates = []
    for _, row in holdout_summary.iterrows():
        strategy = row["strategy"]
        if strategy == "strict_gate_top3":
            continue
        dev_row = dev_summary[dev_summary["strategy"] == strategy]
        if dev_row.empty:
            continue
        dev_row = dev_row.iloc[0]
        added = backtest[(backtest["split"] == "holdout") & (backtest["strategy"] == f"{strategy}_added_only")]
        added_ok = True
        if not added.empty and int(added.iloc[0]["rows"]) > 0:
            added_ok = bool(added.iloc[0]["success_lift"] >= 0.05 and added.iloc[0]["return_lift"] > 0)
        pass_quality = bool(
            row["days"] > strict_row["days"]
            and row["success_rate"] >= strict_row["success_rate"] - 0.03
            and row["success_lift"] >= 0.05
            and row["return_lift"] > 0
            and row["top_stock_share"] <= 0.20
            and row["top_industry_share"] <= 0.50
            and dev_row["success_lift"] >= 0.05
            and dev_row["return_lift"] > 0
            and added_ok
        )
        if pass_quality:
            candidates.append(row.to_dict())

    chosen_strategy = None
    status = "keep_strict_top3"
    reason = "frequency and quantity expansion did not improve enough; keeping fourth round Top 3 strategy"
    if candidates:
        chosen = pd.DataFrame(candidates).sort_values(["days", "success_lift", "return_lift"], ascending=False).iloc[0]
        chosen_strategy = str(chosen["strategy"])
        if chosen_strategy in {"strict_gate_top5", "adaptive_gate_top5"}:
            status = "quantity_expansion_pass"
        else:
            status = "frequency_expansion_pass"
        reason = f"{chosen_strategy} passed holdout expansion gates without materially degrading quality"
    elif (
        holdout_summary[holdout_summary["strategy"] == "adaptive_gate_top3"]["success_lift"].iloc[0] >= 0.05
        and holdout_summary[holdout_summary["strategy"] == "adaptive_gate_top3"]["return_lift"].iloc[0] > 0
    ):
        reason = (
            "adaptive_gate_top3 looked strong on holdout, but its added candidates failed development gates; "
            "keeping fourth round Top 3 to avoid holdout-driven overfit"
        )

    latest_strategy = chosen_strategy or "strict_gate_top3"
    tracking = latest_candidates(scoring_data, latest_strategy, band4_min, band3_min, latest_strategy)
    tracking.to_csv(LATEST_PATH, index=False, encoding="utf-8-sig")
    write_formal(config, status, reason, tracking)

    best_holdout = holdout_summary.sort_values(["days", "success_lift", "return_lift"], ascending=False).head(8)
    lines = [
        "# Deep Learning Frequency Expansion Decision",
        "",
        f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "- Data sources: three allowed CSV inputs only",
        "- Goal: increase action frequency and candidate count without lowering reliability",
        f"- Training loss: {loss[0]:.6f} -> {loss[-1]:.6f}",
        f"- Status: {status}",
        f"- Selected strategy: {latest_strategy}",
        f"- Latest candidate count: {len(tracking)}",
        "",
        "## Fourth Round Baseline",
        "",
        f"- Days: {int(strict_row['days'])}",
        f"- Rows: {int(strict_row['rows'])}",
        f"- Success rate: {fmt_pct(strict_row['success_rate'])}",
        f"- Success lift: {fmt_pct(strict_row['success_lift'])}",
        f"- Avg 10d high close return: {fmt_pct(strict_row['avg_10d_high_close_return'])}",
        f"- Return lift: {fmt_pct(strict_row['return_lift'])}",
        "",
        "## Best Holdout Expansion Strategies",
        "",
    ]
    for _, row in best_holdout.iterrows():
        lines.append(
            f"- {row['strategy']}: days {int(row['days'])}, rows {int(row['rows'])}, "
            f"success {fmt_pct(row['success_rate'])}, lift {fmt_pct(row['success_lift'])}, "
            f"return lift {fmt_pct(row['return_lift'])}, candidates/day {row['avg_daily_candidates']:.2f}"
        )
    lines.extend(
        [
            "",
        "## Decision",
        "",
        f"- {reason}.",
            "",
            "## Output Files",
            "",
            f"- Backtest: `{BACKTEST_PATH}`",
            f"- Bands: `{BANDS_PATH}`",
            f"- Latest: `{LATEST_PATH}`",
            f"- Formal status: `{FORMAL_STATUS_PATH}`",
            f"- Formal candidates: `{FORMAL_CANDIDATES_PATH}`",
        ]
    )
    DECISION_PATH.write_text("\n".join(lines), encoding="utf-8")
    print("OK: deep learning frequency expansion experiment completed")
    print(f"STATUS: {status}")
    print(f"STRATEGY: {latest_strategy}")
    print(f"DECISION: {DECISION_PATH}")


if __name__ == "__main__":
    main()

