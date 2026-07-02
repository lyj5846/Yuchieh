from __future__ import annotations

import csv
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "project_config.json"
EXPERIMENT_DIR = PROJECT_ROOT / "research_layer"
FORMAL_DIR = PROJECT_ROOT / "research_layer" / "formal_write_disabled"

DECISION_PATH = EXPERIMENT_DIR / "relative_selection_decision.md"
METRICS_PATH = EXPERIMENT_DIR / "relative_selection_metrics.csv"
TOPK_PATH = EXPERIMENT_DIR / "relative_selection_topk.csv"
CALIBRATION_PATH = EXPERIMENT_DIR / "relative_selection_calibration.csv"
FAILURE_PATH = EXPERIMENT_DIR / "relative_selection_failure_analysis.csv"
LATEST_SCORES_PATH = EXPERIMENT_DIR / "relative_selection_latest_scores.csv"
FORMAL_STATUS_PATH = FORMAL_DIR / "formal_status.md"
FORMAL_CANDIDATES_PATH = FORMAL_DIR / "formal_candidates.csv"


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def fail(message: str) -> None:
    raise SystemExit(f"FAIL: {message}")


def validate_inputs(config: dict) -> dict[str, Path]:
    paths = {name: Path(value) for name, value in config["allowed_inputs"].items()}
    marker = "stock" + "_raw_only" + "_project"
    for name, path in paths.items():
        if not path.exists():
            fail(f"missing input {name}: {path}")
        if marker in str(path):
            fail(f"forbidden old project path in {name}")
        if name == "theme_group" and PROJECT_ROOT not in path.parents:
            fail("theme_group must live inside this clean project")
    return paths


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def build_dataset(stock: pd.DataFrame, market: pd.DataFrame, theme: pd.DataFrame) -> pd.DataFrame:
    stock = stock.copy()
    market = market.copy()
    theme = theme.copy()
    stock["日期"] = pd.to_datetime(stock["日期"])
    market["日期"] = pd.to_datetime(market["日期"])
    stock["股票代號"] = stock["股票代號"].astype(str)
    theme["股票代號"] = theme["股票代號"].astype(str)

    stock = stock.sort_values(["股票代號", "日期"]).reset_index(drop=True)
    g = stock.groupby("股票代號", sort=False)
    stock["buy_open_next"] = g["開盤價"].shift(-1)
    future_close = pd.concat([g["收盤價"].shift(-i) for i in range(1, 11)], axis=1)
    stock["future_10d_high_close"] = future_close.max(axis=1)
    stock["future_10d_high_close_return"] = stock["future_10d_high_close"] / stock["buy_open_next"] - 1.0
    stock["has_full_10d_window"] = (
        future_close.notna().all(axis=1)
        & stock["buy_open_next"].notna()
        & (stock["buy_open_next"] > 0)
        & np.isfinite(stock["future_10d_high_close_return"])
    )
    stock["target_success"] = (
        stock["future_10d_high_close"] >= stock["buy_open_next"] * 1.03
    ).astype(int)

    for window in [3, 5, 10, 20, 60]:
        stock[f"ret_{window}d"] = g["收盤價"].pct_change(window)
        stock[f"vol_ratio_{window}d"] = stock["成交量(張)"] / g["成交量(張)"].transform(
            lambda s: s.rolling(window, min_periods=max(2, window // 2)).mean()
        )
        stock[f"ma_gap_{window}d"] = stock["收盤價"] / g["收盤價"].transform(
            lambda s: s.rolling(window, min_periods=max(2, window // 2)).mean()
        ) - 1.0
    stock["intraday_range"] = (stock["最高價"] - stock["最低價"]) / stock["收盤價"].replace(0, np.nan)
    stock["close_position"] = (stock["收盤價"] - stock["最低價"]) / (stock["最高價"] - stock["最低價"]).replace(0, np.nan)
    for col in ["三大法人合計買賣超(張)", "融資買進", "融資賣出", "融資餘額"]:
        if col in stock.columns:
            stock[f"{col}_5d_sum"] = g[col].transform(lambda s: s.rolling(5, min_periods=2).sum())

    market = market.sort_values("日期").reset_index(drop=True)
    for col in ["加權指數收盤", "電子指數收盤", "櫃買指數收盤"]:
        if col in market.columns:
            market[f"{col}_ret_5d"] = market[col].pct_change(5)
            market[f"{col}_ma_gap_20d"] = market[col] / market[col].rolling(20, min_periods=10).mean() - 1.0
    if {"上漲家數", "下跌家數"}.issubset(market.columns):
        market["market_breadth"] = market["上漲家數"] / (market["上漲家數"] + market["下跌家數"]).replace(0, np.nan)

    merged = stock.merge(market, on="日期", how="left", suffixes=("", "_market"))
    merged = merged.merge(
        theme[["股票代號", "股票名稱", "主分類", "子分類"]],
        on="股票代號",
        how="left",
    )
    usable = merged[merged["has_full_10d_window"]].copy()
    usable["daily_market_success_rate"] = usable.groupby("日期")["target_success"].transform("mean")
    usable["daily_market_avg_return"] = usable.groupby("日期")["future_10d_high_close_return"].transform("mean")
    usable["relative_return"] = usable["future_10d_high_close_return"] - usable["daily_market_avg_return"]
    usable["relative_top20_label"] = (
        usable.groupby("日期")["future_10d_high_close_return"].rank(pct=True, method="average") >= 0.80
    ).astype(int)
    return usable


def feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    forbidden = {
        "日期",
        "股票代號",
        "股票名稱",
        "buy_open_next",
        "future_10d_high_close",
        "future_10d_high_close_return",
        "has_full_10d_window",
        "target_success",
        "relative_top20_label",
        "relative_return",
    }
    numeric_cols = [
        c
        for c in df.columns
        if c not in forbidden and pd.api.types.is_numeric_dtype(df[c])
    ]
    numeric = df[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    category = pd.get_dummies(
        df[["主分類", "子分類"]].fillna("unknown"),
        prefix=["main", "sub"],
        dtype=float,
    )
    out = pd.concat([numeric, category], axis=1)
    return out, list(out.columns)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def train_logistic(X: np.ndarray, y: np.ndarray, iterations: int = 180) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    Xs = (X - mean) / std
    Xs = np.column_stack([np.ones(len(Xs)), Xs])
    w = np.zeros(Xs.shape[1], dtype=float)
    lr = 0.04
    reg = 0.002
    for _ in range(iterations):
        p = sigmoid(Xs @ w)
        grad = (Xs.T @ (p - y)) / len(y)
        grad[1:] += reg * w[1:]
        w -= lr * grad
    return w, mean, std


def predict_logistic(X: np.ndarray, w: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    Xs = (X - mean) / std
    Xs = np.column_stack([np.ones(len(Xs)), Xs])
    return sigmoid(Xs @ w)


def auc_score(y: np.ndarray, p: np.ndarray) -> float:
    ranks = pd.Series(p).rank(method="average").to_numpy()
    pos = y == 1
    n_pos = float(pos.sum())
    n_neg = float((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def calibration_rows(df: pd.DataFrame, split: str, score_col: str, label_col: str) -> list[dict]:
    if df.empty:
        return []
    bins = pd.qcut(df[score_col], q=5, duplicates="drop")
    rows = []
    for band, part in df.groupby(bins, observed=False):
        rows.append(
            {
                "split": split,
                "score": score_col,
                "band": str(band),
                "rows": len(part),
                "avg_score": part[score_col].mean(),
                "actual_success_rate": part["target_success"].mean(),
                "actual_relative_top20_rate": part[label_col].mean(),
                "avg_10d_high_close_return": part["future_10d_high_close_return"].mean(),
            }
        )
    return rows


def split_masks(df: pd.DataFrame, config: dict) -> dict[str, pd.Series]:
    cfg = config["time_split"]
    return {
        "train": df["日期"] <= pd.Timestamp(cfg["train_end"]),
        "development": (df["日期"] >= pd.Timestamp(cfg["dev_start"])) & (df["日期"] <= pd.Timestamp(cfg["dev_end"])),
        "holdout": df["日期"] >= pd.Timestamp(cfg["holdout_start"]),
    }


def topk_evaluate(df: pd.DataFrame, split: str, strategy: str, score_col: str, k: int, market_gate: float | None = None) -> dict:
    work = df.copy()
    if market_gate is not None:
        work = work[work["daily_market_success_rate"] >= market_gate]
    if work.empty:
        return {
            "split": split,
            "strategy": strategy,
            "top_k": k,
            "market_gate": market_gate if market_gate is not None else "",
            "rows": 0,
            "days": 0,
            "topk_success_rate": float("nan"),
            "same_day_baseline_success_rate": float("nan"),
            "success_lift": float("nan"),
            "topk_avg_return": float("nan"),
            "same_day_baseline_avg_return": float("nan"),
            "return_lift": float("nan"),
            "unique_stocks": 0,
            "top_stock_share": float("nan"),
            "top_industry_share": float("nan"),
        }
    picked = work.sort_values(["日期", score_col], ascending=[True, False]).groupby("日期").head(k)
    baseline_by_day = work.groupby("日期").agg(
        day_success=("target_success", "mean"),
        day_return=("future_10d_high_close_return", "mean"),
    )
    picked_by_day = picked.groupby("日期").agg(
        pick_success=("target_success", "mean"),
        pick_return=("future_10d_high_close_return", "mean"),
    )
    joined = picked_by_day.join(baseline_by_day, how="inner")
    stock_share = picked["股票代號"].value_counts(normalize=True).iloc[0] if not picked.empty else float("nan")
    industry_share = picked["主分類"].fillna("unknown").value_counts(normalize=True).iloc[0] if not picked.empty else float("nan")
    return {
        "split": split,
        "strategy": strategy,
        "top_k": k,
        "market_gate": market_gate if market_gate is not None else "",
        "rows": len(picked),
        "days": picked["日期"].nunique(),
        "topk_success_rate": picked["target_success"].mean(),
        "same_day_baseline_success_rate": joined["day_success"].mean(),
        "success_lift": joined["pick_success"].mean() - joined["day_success"].mean(),
        "topk_avg_return": picked["future_10d_high_close_return"].mean(),
        "same_day_baseline_avg_return": joined["day_return"].mean(),
        "return_lift": joined["pick_return"].mean() - joined["day_return"].mean(),
        "unique_stocks": picked["股票代號"].nunique(),
        "top_stock_share": stock_share,
        "top_industry_share": industry_share,
    }


def failure_analysis(df: pd.DataFrame, split: str, score_col: str) -> list[dict]:
    rows = []
    picked = df.sort_values(["日期", score_col], ascending=[True, False]).groupby("日期").head(5).copy()
    if picked.empty:
        return rows
    picked["failed"] = 1 - picked["target_success"]
    for col in ["主分類", "子分類"]:
        for key, part in picked.groupby(col, dropna=False):
            if len(part) < 5:
                continue
            rows.append(
                {
                    "split": split,
                    "dimension": col,
                    "value": key,
                    "rows": len(part),
                    "failure_rate": part["failed"].mean(),
                    "avg_score": part[score_col].mean(),
                    "avg_return": part["future_10d_high_close_return"].mean(),
                }
            )
    return rows


def fmt_pct(value: float) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    return f"{value:.2%}"


def write_formal_default(config: dict, reason: str) -> None:
    FORMAL_STATUS_PATH.write_text(
        "\n".join(
            [
                "# Formal Status",
                "",
                "- Status: not active",
                f"- Result: {config['formal_candidate_default']}",
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


def main() -> None:
    EXPERIMENT_DIR.mkdir(exist_ok=True)
    FORMAL_DIR.mkdir(exist_ok=True)
    config = load_config()
    paths = validate_inputs(config)
    stock = read_csv(paths["stock_daily_all"])
    market = read_csv(paths["market_daily"])
    theme = read_csv(paths["theme_group"])

    data = build_dataset(stock, market, theme)
    masks = split_masks(data, config)
    features, feature_names = feature_matrix(data)
    X = features.to_numpy(dtype=float)
    abs_y = data["target_success"].to_numpy(dtype=float)
    rel_y = data["relative_top20_label"].to_numpy(dtype=float)

    train_ids = masks["train"].to_numpy()
    if train_ids.sum() == 0:
        fail("no train rows")
    abs_model = train_logistic(X[train_ids], abs_y[train_ids])
    rel_model = train_logistic(X[train_ids], rel_y[train_ids])
    data["absolute_probability_model"] = predict_logistic(X, *abs_model)
    data["relative_rank_model"] = predict_logistic(X, *rel_model)
    data["combined_rank_score"] = (
        data["absolute_probability_model"].rank(pct=True)
        + data["relative_rank_model"].rank(pct=True)
    ) / 2.0

    metrics = []
    calibration = []
    topk = []
    failure = []
    for split, mask in masks.items():
        part = data.loc[mask].copy()
        for score_col, label_col in [
            ("absolute_probability_model", "target_success"),
            ("relative_rank_model", "relative_top20_label"),
            ("combined_rank_score", "relative_top20_label"),
        ]:
            y = part[label_col].to_numpy(dtype=float)
            p = part[score_col].to_numpy(dtype=float)
            metrics.append(
                {
                    "split": split,
                    "model": score_col,
                    "rows": len(part),
                    "auc": auc_score(y, p),
                    "brier": float(np.mean((np.clip(p, 0, 1) - y) ** 2)),
                    "label_rate": float(y.mean()) if len(y) else float("nan"),
                    "avg_10d_high_close_return": float(part["future_10d_high_close_return"].mean()) if len(part) else float("nan"),
                    "same_day_market_success_rate": float(part["daily_market_success_rate"].mean()) if len(part) else float("nan"),
                }
            )
            calibration.extend(calibration_rows(part, split, score_col, label_col))
        for score_col in ["absolute_probability_model", "relative_rank_model", "combined_rank_score"]:
            for k in [3, 5, 10]:
                topk.append(topk_evaluate(part, split, score_col, score_col, k))
                for gate in [0.55, 0.65, 0.75]:
                    topk.append(topk_evaluate(part, split, f"{score_col}_market_gate", score_col, k, gate))
        failure.extend(failure_analysis(part, split, "combined_rank_score"))

    metric_df = pd.DataFrame(metrics)
    topk_df = pd.DataFrame(topk)
    calibration_df = pd.DataFrame(calibration)
    failure_df = pd.DataFrame(failure)
    metric_df.to_csv(METRICS_PATH, index=False, encoding="utf-8-sig")
    topk_df.to_csv(TOPK_PATH, index=False, encoding="utf-8-sig")
    calibration_df.to_csv(CALIBRATION_PATH, index=False, encoding="utf-8-sig")
    failure_df.to_csv(FAILURE_PATH, index=False, encoding="utf-8-sig")

    latest_date = data["日期"].max()
    latest = data[data["日期"] == latest_date].copy()
    latest.sort_values("combined_rank_score", ascending=False).head(50).to_csv(
        LATEST_SCORES_PATH, index=False, encoding="utf-8-sig"
    )

    dev_candidates = topk_df[
        (topk_df["split"] == "development")
        & (topk_df["days"] >= 20)
        & (topk_df["success_lift"] >= 0.05)
        & (topk_df["return_lift"] > 0)
        & (topk_df["top_stock_share"] <= 0.20)
        & (topk_df["top_industry_share"] <= 0.50)
    ].copy()
    holdout = topk_df[topk_df["split"] == "holdout"].copy()
    passed_rows = []
    for _, row in dev_candidates.iterrows():
        h = holdout[
            (holdout["strategy"] == row["strategy"])
            & (holdout["top_k"] == row["top_k"])
            & (holdout["market_gate"].fillna("").astype(str) == str(row["market_gate"]))
        ]
        if h.empty:
            continue
        hrow = h.iloc[0]
        if (
            hrow["days"] >= 20
            and hrow["success_lift"] >= 0.05
            and hrow["return_lift"] > 0
            and hrow["top_stock_share"] <= 0.20
            and hrow["top_industry_share"] <= 0.50
        ):
            passed_rows.append(hrow.to_dict())

    status = "rejected"
    reason = "second round relative selection did not pass holdout gates"
    if passed_rows:
        status = "research_pass_needs_manual_review"
        reason = "at least one relative Top-K strategy passed holdout gates; formal promotion still requires calibration review"
    write_formal_default(config, reason)

    best_dev = dev_candidates.sort_values(["success_lift", "return_lift"], ascending=False).head(5)
    best_holdout = holdout.sort_values(["success_lift", "return_lift"], ascending=False).head(5)
    lines = [
        "# Relative Selection Experiment Decision",
        "",
        f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "- Data sources: three allowed CSV inputs only",
        "- Goal: compare stocks within the same signal day, not only absolute +3% probability",
        f"- Rows: {len(data)}",
        f"- Features: {len(feature_names)}",
        f"- Latest scored date: {latest_date.strftime('%Y-%m-%d')}",
        f"- Status: {status}",
        f"- Formal output: {config['formal_candidate_default']}",
        "",
        "## Holdout Baseline Context",
        "",
        f"- Holdout average same-day market success rate: {fmt_pct(data.loc[masks['holdout'], 'daily_market_success_rate'].mean())}",
        f"- Holdout average same-day market 10d high return: {fmt_pct(data.loc[masks['holdout'], 'daily_market_avg_return'].mean())}",
        "",
        "## Best Development Strategies",
        "",
    ]
    if best_dev.empty:
        lines.append("- None passed development gates.")
    else:
        for _, row in best_dev.iterrows():
            lines.append(
                f"- {row['strategy']} Top {int(row['top_k'])}, gate={row['market_gate']}: "
                f"success lift {fmt_pct(row['success_lift'])}, return lift {fmt_pct(row['return_lift'])}, days {int(row['days'])}"
            )
    lines.extend(["", "## Best Holdout Strategies", ""])
    for _, row in best_holdout.iterrows():
        lines.append(
            f"- {row['strategy']} Top {int(row['top_k'])}, gate={row['market_gate']}: "
            f"success lift {fmt_pct(row['success_lift'])}, return lift {fmt_pct(row['return_lift'])}, days {int(row['days'])}, "
            f"top stock share {fmt_pct(row['top_stock_share'])}, industry share {fmt_pct(row['top_industry_share'])}"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- {reason}.",
            "- Formal candidates remain empty unless the model passes all promotion gates.",
            "",
            "## Output Files",
            "",
            f"- Metrics: `{METRICS_PATH}`",
            f"- Top-K: `{TOPK_PATH}`",
            f"- Calibration: `{CALIBRATION_PATH}`",
            f"- Failure analysis: `{FAILURE_PATH}`",
            f"- Latest research scores: `{LATEST_SCORES_PATH}`",
        ]
    )
    DECISION_PATH.write_text("\n".join(lines), encoding="utf-8")
    print("OK: relative selection experiment completed")
    print(f"STATUS: {status}")
    print(f"DECISION: {DECISION_PATH}")


if __name__ == "__main__":
    main()

