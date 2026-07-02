from __future__ import annotations

import csv
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from run_deep_learning_exception_gate_experiment import (  # noqa: E402
    build_rolling_scores,
    fourth_round_baseline,
    load_clean_data,
    pick_exception,
    summarize,
)
from run_deep_learning_selection_experiment import (  # noqa: E402
    fmt_pct,
    standardize,
    train_mlp,
)


EXPERIMENT_DIR = PROJECT_ROOT / "research_layer"
FORMAL_DIR = PROJECT_ROOT / "research_layer" / "formal_write_disabled"

DECISION_PATH = EXPERIMENT_DIR / "deep_learning_episode_event_decision.md"
BACKTEST_PATH = EXPERIMENT_DIR / "deep_learning_episode_event_backtest.csv"
FAILURE_PATH = EXPERIMENT_DIR / "deep_learning_episode_event_failure_learning.csv"
CANDIDATES_PATH = EXPERIMENT_DIR / "deep_learning_episode_event_candidates.csv"
LATEST_PATH = EXPERIMENT_DIR / "deep_learning_episode_event_latest.csv"
FORMAL_STATUS_PATH = FORMAL_DIR / "formal_status.md"
FORMAL_CANDIDATES_PATH = FORMAL_DIR / "formal_candidates.csv"

EPISODE_GAP_TRADING_DAYS = 10
MIN_EPISODES_FOR_FORMAL = 12


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def add_future_low_return(scoring_data: pd.DataFrame) -> pd.DataFrame:
    work = scoring_data.sort_values(["股票代號", "日期"]).copy()
    g = work.groupby("股票代號", sort=False)
    future_close = pd.concat([g["收盤價"].shift(-i) for i in range(1, 11)], axis=1)
    work["future_10d_low_close"] = future_close.min(axis=1)
    work["future_10d_low_close_return"] = work["future_10d_low_close"] / work["buy_open_next"] - 1.0
    work["stock_trading_index"] = g.cumcount()
    return work


def rebuild_exception_candidates(
    full: pd.DataFrame,
    fourth_days: set[pd.Timestamp],
) -> pd.DataFrame:
    exception_universe = full[~full["日期"].isin(fourth_days)].copy()
    strategies = [
        "exception_6m_top1",
        "exception_9m_top1",
        "exception_12m_top1",
        "exception_ensemble_mean_top1",
        "exception_ensemble_all_high_top1",
    ]
    parts = []
    for strategy in strategies:
        picked = pick_exception(exception_universe, strategy).copy()
        if picked.empty:
            continue
        picked["strategy"] = strategy
        parts.append(picked)
    if not parts:
        return pd.DataFrame()
    raw = pd.concat(parts, ignore_index=True)
    raw = raw.sort_values(["日期", "股票代號", "exception_score"], ascending=[True, True, False])
    raw["strategy_list"] = raw.groupby(["日期", "股票代號"])["strategy"].transform(lambda s: "|".join(sorted(set(s))))
    raw["strategy_count"] = raw.groupby(["日期", "股票代號"])["strategy"].transform("nunique")
    raw["max_exception_score"] = raw.groupby(["日期", "股票代號"])["exception_score"].transform("max")
    return raw.drop_duplicates(["日期", "股票代號"], keep="first").copy()


def build_episodes(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame()
    rows = []
    for stock_id, part in candidates.sort_values(["股票代號", "日期"]).groupby("股票代號", sort=False):
        episode_rows = []
        last_idx = None
        episode_id = 0
        for _, row in part.iterrows():
            current_idx = int(row["stock_trading_index"])
            if last_idx is None or current_idx - last_idx > EPISODE_GAP_TRADING_DAYS:
                if episode_rows:
                    rows.append(make_episode_row(stock_id, episode_id, episode_rows))
                episode_id += 1
                episode_rows = [row]
            else:
                episode_rows.append(row)
            last_idx = current_idx
        if episode_rows:
            rows.append(make_episode_row(stock_id, episode_id, episode_rows))
    episodes = pd.DataFrame(rows)
    if episodes.empty:
        return episodes
    episodes = episodes.sort_values(["first_signal_date", "股票代號"]).reset_index(drop=True)
    episodes["episode_global_id"] = np.arange(1, len(episodes) + 1)
    return episodes


def make_episode_row(stock_id: str, episode_id: int, episode_rows: list[pd.Series]) -> dict:
    first = episode_rows[0].copy()
    strategies = sorted(
        {
            item
            for r in episode_rows
            for item in str(r["strategy_list"]).split("|")
            if item
        }
    )
    signal_count = len(episode_rows)
    first["episode_stock_id"] = f"{stock_id}_{episode_id}"
    first["first_signal_date"] = first["日期"]
    first["last_signal_date"] = episode_rows[-1]["日期"]
    first["episode_signal_count"] = signal_count
    first["episode_strategy_count"] = len(strategies)
    first["episode_strategy_list"] = "|".join(strategies)
    first["success_label"] = int(first["target_success"])
    first["failure_label"] = int((first["target_success"] == 0) or (first["future_10d_low_close_return"] <= -0.03))
    return first.to_dict()


def add_episode_context(episodes: pd.DataFrame) -> pd.DataFrame:
    work = episodes.sort_values(["股票代號", "first_signal_date"]).copy()
    work["prev_episode_date"] = work.groupby("股票代號")["first_signal_date"].shift(1)
    work["prev_episode_index"] = work.groupby("股票代號")["stock_trading_index"].shift(1)
    work["days_since_prev_episode"] = work["stock_trading_index"] - work["prev_episode_index"]
    work["days_since_prev_episode"] = work["days_since_prev_episode"].fillna(999.0)
    work["is_first_episode_for_stock"] = work["prev_episode_date"].isna().astype(float)
    return work


def feature_frame(episodes: pd.DataFrame) -> pd.DataFrame:
    preferred = [
        "score_6m",
        "score_9m",
        "score_12m",
        "ensemble_score_mean",
        "ensemble_score_min",
        "ensemble_agree_count",
        "max_exception_score",
        "strategy_count",
        "market_strength_score",
        "daily_market_success_rate",
        "daily_market_avg_return",
        "ret_1d",
        "ret_5d",
        "ret_10d",
        "ret_20d",
        "vol_ratio_5d",
        "vol_ratio_10d",
        "vol_ratio_20d",
        "ma_gap_5d",
        "ma_gap_10d",
        "ma_gap_20d",
        "intraday_return",
        "intraday_range",
        "close_position",
        "days_since_prev_episode",
        "is_first_episode_for_stock",
    ]
    numeric = [c for c in preferred if c in episodes.columns and pd.api.types.is_numeric_dtype(episodes[c])]
    return episodes[numeric].replace([np.inf, -np.inf], np.nan).fillna(0.0)


def train_episode_models(episodes: pd.DataFrame, features: pd.DataFrame, train_mask: pd.Series) -> tuple[pd.DataFrame, str]:
    out = episodes.copy()
    train_count = int(train_mask.sum())
    if train_count < 8:
        out["success_score"] = np.nan
        out["failure_risk_score"] = np.nan
        return out, "insufficient pre-holdout episodes for deep learning training"
    y_success = out["success_label"].to_numpy(dtype=np.float32).reshape(-1, 1)
    y_failure = out["failure_label"].to_numpy(dtype=np.float32).reshape(-1, 1)
    if len(np.unique(y_success[train_mask.to_numpy()])) < 2 or len(np.unique(y_failure[train_mask.to_numpy()])) < 2:
        out["success_score"] = np.nan
        out["failure_risk_score"] = np.nan
        return out, "pre-holdout episodes do not contain both success and failure classes"
    X, _, _ = standardize(features.to_numpy(dtype=np.float32), train_mask.to_numpy())
    success_model, _ = train_mlp(X[train_mask.to_numpy()], y_success[train_mask.to_numpy()], hidden=8, epochs=120, seed=801)
    failure_model, _ = train_mlp(X[train_mask.to_numpy()], y_failure[train_mask.to_numpy()], hidden=8, epochs=120, seed=802)
    out["success_score"] = sigmoid(
        (np.maximum(X @ success_model["w1"] + success_model["b1"], 0.0) @ success_model["w2"] + success_model["b2"])[:, 0]
    )
    out["failure_risk_score"] = sigmoid(
        (np.maximum(X @ failure_model["w1"] + failure_model["b1"], 0.0) @ failure_model["w2"] + failure_model["b2"])[:, 0]
    )
    return out, "trained"


def band_rows(scored: pd.DataFrame, split_name: str, score_col: str, target_col: str) -> list[dict]:
    rows = []
    if scored.empty or scored[score_col].isna().all():
        return rows
    work = scored.copy()
    work["band"] = pd.qcut(work[score_col].rank(method="first"), q=min(3, len(work)), labels=False, duplicates="drop")
    for band, part in work.groupby("band"):
        rows.append(
            {
                "split": split_name,
                "score_col": score_col,
                "band": int(band),
                "rows": len(part),
                "avg_score": part[score_col].mean(),
                "actual_rate": part[target_col].mean(),
                "avg_10d_high_close_return": part["future_10d_high_close_return"].mean(),
            }
        )
    return rows


def select_model_candidates(scored: pd.DataFrame, train_mask: pd.Series, eval_mask: pd.Series) -> tuple[pd.DataFrame, str]:
    if scored["success_score"].isna().all() or scored["failure_risk_score"].isna().all():
        return pd.DataFrame(), "no trained scores"
    train = scored[train_mask].copy()
    if train.empty:
        return pd.DataFrame(), "no pre-holdout episodes to set score thresholds"
    success_min = train["success_score"].quantile(0.60)
    failure_max = train["failure_risk_score"].quantile(0.50)
    picked = scored[eval_mask & (scored["success_score"] >= success_min) & (scored["failure_risk_score"] <= failure_max)].copy()
    return picked, f"success_score >= train p60 and failure_risk_score <= train p50"


def write_formal(config: dict, passed: bool, reason: str, latest: pd.DataFrame) -> None:
    result = "deep learning Top 3 formal strategy remains enabled"
    if passed:
        result = "deep learning Top 3 plus episode exception formal strategy enabled"
    if latest.empty:
        result += "; current latest date has no episode candidate"
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
                    row["first_signal_date"].strftime("%Y-%m-%d"),
                    row["股票代號"],
                    row["股票名稱"],
                    "",
                    "",
                    "",
                    "",
                    "episode_exception",
                    "tracking only until 10 trading days complete",
                ]
            )


def main() -> None:
    EXPERIMENT_DIR.mkdir(exist_ok=True)
    FORMAL_DIR.mkdir(exist_ok=True)
    config, data, features, scoring_data, _ = load_clean_data()
    dev_start = pd.Timestamp(config["time_split"]["dev_start"])
    dev_end = pd.Timestamp(config["time_split"]["dev_end"])
    holdout_start = pd.Timestamp(config["time_split"]["holdout_start"])

    scoring_data = add_future_low_return(scoring_data)
    rolling_scores = build_rolling_scores(data, features, dev_start)
    fourth_full, fourth_picked = fourth_round_baseline(config, dev_start)
    fourth_days = set(fourth_picked["日期"].unique())
    full = scoring_data[scoring_data["日期"] >= dev_start].merge(rolling_scores, on=["日期", "股票代號"], how="inner")

    daily_candidates = rebuild_exception_candidates(full, fourth_days)
    episodes = add_episode_context(build_episodes(daily_candidates))
    feature_data = feature_frame(episodes)
    train_mask = episodes["first_signal_date"] <= dev_end
    eval_mask = episodes["first_signal_date"] >= holdout_start
    scored, model_status = train_episode_models(episodes, feature_data, train_mask)
    selected, select_rule = select_model_candidates(scored, train_mask, eval_mask)

    scored.to_csv(CANDIDATES_PATH, index=False, encoding="utf-8-sig")
    selected.to_csv(BACKTEST_PATH, index=False, encoding="utf-8-sig")

    failure_rows = []
    failure_rows.extend(band_rows(scored[train_mask], "pre_holdout", "success_score", "success_label"))
    failure_rows.extend(band_rows(scored[eval_mask], "holdout", "success_score", "success_label"))
    failure_rows.extend(band_rows(scored[train_mask], "pre_holdout", "failure_risk_score", "failure_label"))
    failure_rows.extend(band_rows(scored[eval_mask], "holdout", "failure_risk_score", "failure_label"))
    failure_table = pd.DataFrame(failure_rows)
    failure_table.to_csv(FAILURE_PATH, index=False, encoding="utf-8-sig")

    summary = summarize(scored[eval_mask], selected, "episode_success_high_failure_low")
    high_success_ok = False
    high_failure_ok = False
    holdout_bands = failure_table[failure_table["split"] == "holdout"].copy()
    success_bands = holdout_bands[holdout_bands["score_col"] == "success_score"].sort_values("band")
    failure_bands = holdout_bands[holdout_bands["score_col"] == "failure_risk_score"].sort_values("band")
    if len(success_bands) >= 2:
        high_success_ok = bool(success_bands.iloc[-1]["actual_rate"] >= success_bands.iloc[0]["actual_rate"])
    if len(failure_bands) >= 2:
        high_failure_ok = bool(failure_bands.iloc[-1]["actual_rate"] >= failure_bands.iloc[0]["actual_rate"])
    month_counts = selected["first_signal_date"].dt.strftime("%Y-%m").nunique() if not selected.empty else 0
    passes = bool(
        model_status == "trained"
        and int(summary["rows"]) >= MIN_EPISODES_FOR_FORMAL
        and high_success_ok
        and high_failure_ok
        and summary["success_lift"] >= 0.08
        and summary["return_lift"] > 0
        and summary["top_stock_share"] <= 0.20
        and summary["top_industry_share"] <= 0.50
        and month_counts >= 2
    )
    reason = "episode model passed all gates and can be added to fourth-round Top 3"
    if not passes:
        reason = "episode model did not pass all gates; keeping fourth-round Top 3"

    latest = selected[selected["first_signal_date"] == scored["first_signal_date"].max()].copy() if not selected.empty else pd.DataFrame()
    latest.to_csv(LATEST_PATH, index=False, encoding="utf-8-sig")
    write_formal(config, passes, reason, latest)

    lines = [
        "# Deep Learning Episode Event Decision",
        "",
        f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "- Data sources: three allowed CSV inputs only",
        "- Goal: collapse exception daily signals into stock episodes and learn both success and failure",
        f"- Status: {'episode_exception_pass' if passes else 'keep_fourth_round'}",
        f"- Model status: {model_status}",
        f"- Selection rule: {select_rule}",
        "",
        "## Episode Summary",
        "",
        f"- Daily exception candidates after dedupe: {len(daily_candidates)}",
        f"- Episode events: {len(scored)}",
        f"- Pre-holdout episodes: {int(train_mask.sum())}",
        f"- Holdout episodes: {int(eval_mask.sum())}",
        f"- Selected holdout episodes: {int(summary['rows'])}",
        f"- Selected success rate: {fmt_pct(summary['success_rate'])}",
        f"- Selected success lift: {fmt_pct(summary['success_lift'])}",
        f"- Selected return lift: {fmt_pct(summary['return_lift'])}",
        f"- Top stock share: {fmt_pct(summary['top_stock_share'])}",
        f"- Top industry share: {fmt_pct(summary['top_industry_share'])}",
        f"- Success score ordering valid: {high_success_ok}",
        f"- Failure risk ordering valid: {high_failure_ok}",
        "",
        "## Decision",
        "",
        f"- {reason}.",
        "",
        "## Output Files",
        "",
        f"- Backtest: `{BACKTEST_PATH}`",
        f"- Failure learning: `{FAILURE_PATH}`",
        f"- Episode candidates: `{CANDIDATES_PATH}`",
        f"- Latest: `{LATEST_PATH}`",
        f"- Formal status: `{FORMAL_STATUS_PATH}`",
    ]
    DECISION_PATH.write_text("\n".join(lines), encoding="utf-8")
    print("OK: deep learning episode event experiment completed")
    print(f"STATUS: {'episode_exception_pass' if passes else 'keep_fourth_round'}")
    print(f"DECISION: {DECISION_PATH}")


if __name__ == "__main__":
    main()

