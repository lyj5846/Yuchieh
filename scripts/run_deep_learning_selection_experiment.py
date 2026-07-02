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

DECISION_PATH = EXPERIMENT_DIR / "deep_learning_selection_decision.md"
METRICS_PATH = EXPERIMENT_DIR / "deep_learning_metrics.csv"
TOPK_PATH = EXPERIMENT_DIR / "deep_learning_topk.csv"
CALIBRATION_PATH = EXPERIMENT_DIR / "deep_learning_calibration.csv"
FAILURE_PATH = EXPERIMENT_DIR / "deep_learning_failure_analysis.csv"
LATEST_SCORES_PATH = EXPERIMENT_DIR / "deep_learning_latest_scores.csv"
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


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def auc_score(y: np.ndarray, p: np.ndarray) -> float:
    ranks = pd.Series(p).rank(method="average").to_numpy()
    pos = y == 1
    n_pos = float(pos.sum())
    n_neg = float((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def fmt_pct(value: float) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    return f"{value:.2%}"


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

    stock["ret_1d"] = g["收盤價"].pct_change(1)
    stock["open_gap"] = stock["開盤價"] / g["收盤價"].shift(1) - 1.0
    stock["intraday_return"] = stock["收盤價"] / stock["開盤價"].replace(0, np.nan) - 1.0
    stock["intraday_range"] = (stock["最高價"] - stock["最低價"]) / stock["收盤價"].replace(0, np.nan)
    stock["close_position"] = (stock["收盤價"] - stock["最低價"]) / (stock["最高價"] - stock["最低價"]).replace(0, np.nan)

    for window in [5, 10, 20]:
        stock[f"ret_{window}d"] = g["收盤價"].pct_change(window)
        stock[f"vol_ratio_{window}d"] = stock["成交量(張)"] / g["成交量(張)"].transform(
            lambda s: s.rolling(window, min_periods=max(2, window // 2)).mean()
        )
        stock[f"ma_gap_{window}d"] = stock["收盤價"] / g["收盤價"].transform(
            lambda s: s.rolling(window, min_periods=max(2, window // 2)).mean()
        ) - 1.0

    for col in ["外資買賣超(張)", "投信買賣超(張)", "自營商買賣超(張)", "三大法人合計買賣超(張)", "融資買進", "融資賣出", "融資餘額"]:
        if col in stock.columns:
            stock[f"{col}_5d_sum"] = g[col].transform(lambda s: s.rolling(5, min_periods=2).sum())

    market = market.sort_values("日期").reset_index(drop=True)
    for col in ["加權指數收盤", "電子指數收盤", "櫃買指數收盤"]:
        if col in market.columns:
            market[f"{col}_ret_5d"] = market[col].pct_change(5)
            market[f"{col}_ma_gap_20d"] = market[col] / market[col].rolling(20, min_periods=10).mean() - 1.0
    if {"上漲家數", "下跌家數"}.issubset(market.columns):
        market["market_breadth"] = market["上漲家數"] / (market["上漲家數"] + market["下跌家數"]).replace(0, np.nan)
    if "大盤成交值(億元)" in market.columns:
        market["market_value_ratio_20d"] = market["大盤成交值(億元)"] / market["大盤成交值(億元)"].rolling(20, min_periods=10).mean()

    strength_cols = [
        c
        for c in ["market_breadth", "加權指數收盤_ret_5d", "電子指數收盤_ret_5d", "櫃買指數收盤_ret_5d", "加權指數收盤_ma_gap_20d", "market_value_ratio_20d"]
        if c in market.columns
    ]
    if strength_cols:
        market["market_strength_score"] = market[strength_cols].replace([np.inf, -np.inf], np.nan).rank(pct=True).mean(axis=1)
    else:
        market["market_strength_score"] = 0.5

    merged = stock.merge(market, on="日期", how="left", suffixes=("", "_market"))
    merged = merged.merge(
        theme[["股票代號", "股票名稱", "主分類", "子分類"]],
        on="股票代號",
        how="left",
    )
    usable = merged[merged["has_full_10d_window"]].copy()
    usable["daily_market_success_rate"] = usable.groupby("日期")["target_success"].transform("mean")
    usable["daily_market_avg_return"] = usable.groupby("日期")["future_10d_high_close_return"].transform("mean")
    usable["relative_top20_label"] = (
        usable.groupby("日期")["future_10d_high_close_return"].rank(pct=True, method="average") >= 0.80
    ).astype(int)
    return usable


def sequence_features(df: pd.DataFrame, lookback: int = 20) -> tuple[pd.DataFrame, list[str]]:
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
    }
    candidate_cols = [
        c
        for c in df.columns
        if c not in forbidden and pd.api.types.is_numeric_dtype(df[c])
    ]
    keep_tokens = [
        "ret_",
        "vol_ratio_",
        "ma_gap_",
        "intraday",
        "close_position",
        "買賣超",
        "融資",
        "market_",
        "成交值",
        "收盤_ret",
        "ma_gap_20d",
    ]
    base_cols = [c for c in candidate_cols if any(t in c for t in keep_tokens)]
    base_cols = base_cols[:32]
    work = df.sort_values(["股票代號", "日期"]).copy()
    g = work.groupby("股票代號", sort=False)
    pieces = []
    names = []
    for lag in range(lookback):
        shifted = g[base_cols].shift(lag)
        shifted.columns = [f"{c}_lag{lag}" for c in base_cols]
        pieces.append(shifted)
        names.extend(list(shifted.columns))
    seq = pd.concat(pieces, axis=1)
    category = pd.get_dummies(
        work[["主分類", "子分類"]].fillna("unknown"),
        prefix=["main", "sub"],
        dtype=float,
    )
    out = pd.concat([seq, category], axis=1).replace([np.inf, -np.inf], np.nan)
    out["has_full_20d_history"] = seq.notna().all(axis=1).astype(int)
    out = out.fillna(0.0)
    return out, list(out.columns)


def split_masks(df: pd.DataFrame, config: dict) -> dict[str, pd.Series]:
    cfg = config["time_split"]
    return {
        "train": df["日期"] <= pd.Timestamp(cfg["train_end"]),
        "development": (df["日期"] >= pd.Timestamp(cfg["dev_start"])) & (df["日期"] <= pd.Timestamp(cfg["dev_end"])),
        "holdout": df["日期"] >= pd.Timestamp(cfg["holdout_start"]),
    }


def standardize(X: np.ndarray, train_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = X[train_mask].mean(axis=0)
    std = X[train_mask].std(axis=0)
    std[std == 0] = 1.0
    Xs = (X - mean) / std
    return Xs.astype(np.float32), mean, std


def train_mlp(
    X: np.ndarray,
    y: np.ndarray,
    hidden: int = 40,
    epochs: int = 70,
    batch_size: int = 4096,
    seed: int = 7,
) -> tuple[dict[str, np.ndarray], list[float]]:
    rng = np.random.default_rng(seed)
    n, d = X.shape
    outputs = y.shape[1]
    params = {
        "w1": (rng.normal(0, 1 / math.sqrt(d), size=(d, hidden))).astype(np.float32),
        "b1": np.zeros(hidden, dtype=np.float32),
        "w2": (rng.normal(0, 1 / math.sqrt(hidden), size=(hidden, outputs))).astype(np.float32),
        "b2": np.zeros(outputs, dtype=np.float32),
    }
    lr = 0.015
    reg = 0.0005
    history: list[float] = []
    for epoch in range(epochs):
        order = rng.permutation(n)
        total_loss = 0.0
        seen = 0
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            xb = X[idx]
            yb = y[idx]
            z1 = xb @ params["w1"] + params["b1"]
            h = relu(z1)
            logits = h @ params["w2"] + params["b2"]
            p = sigmoid(logits)
            loss = -np.mean(yb * np.log(p + 1e-7) + (1 - yb) * np.log(1 - p + 1e-7))
            total_loss += float(loss) * len(idx)
            seen += len(idx)
            grad_logits = (p - yb) / len(idx)
            grad_w2 = h.T @ grad_logits + reg * params["w2"]
            grad_b2 = grad_logits.sum(axis=0)
            grad_h = grad_logits @ params["w2"].T
            grad_z1 = grad_h * (z1 > 0)
            grad_w1 = xb.T @ grad_z1 + reg * params["w1"]
            grad_b1 = grad_z1.sum(axis=0)
            params["w2"] -= lr * grad_w2.astype(np.float32)
            params["b2"] -= lr * grad_b2.astype(np.float32)
            params["w1"] -= lr * grad_w1.astype(np.float32)
            params["b1"] -= lr * grad_b1.astype(np.float32)
        history.append(total_loss / max(seen, 1))
    return params, history


def predict_mlp(X: np.ndarray, params: dict[str, np.ndarray]) -> np.ndarray:
    return sigmoid(relu(X @ params["w1"] + params["b1"]) @ params["w2"] + params["b2"])


def topk_evaluate(df: pd.DataFrame, split: str, strategy: str, score_col: str, k: int, gate: float | None = None) -> dict:
    work = df.copy()
    if gate is not None:
        work = work[work["market_strength_score"] >= gate]
    if work.empty:
        return {
            "split": split,
            "strategy": strategy,
            "top_k": k,
            "market_strength_gate": gate if gate is not None else "",
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
    base = work.groupby("日期").agg(
        day_success=("target_success", "mean"),
        day_return=("future_10d_high_close_return", "mean"),
    )
    chosen = picked.groupby("日期").agg(
        pick_success=("target_success", "mean"),
        pick_return=("future_10d_high_close_return", "mean"),
    )
    joined = chosen.join(base, how="inner")
    return {
        "split": split,
        "strategy": strategy,
        "top_k": k,
        "market_strength_gate": gate if gate is not None else "",
        "rows": len(picked),
        "days": picked["日期"].nunique(),
        "topk_success_rate": picked["target_success"].mean(),
        "same_day_baseline_success_rate": joined["day_success"].mean(),
        "success_lift": joined["pick_success"].mean() - joined["day_success"].mean(),
        "topk_avg_return": picked["future_10d_high_close_return"].mean(),
        "same_day_baseline_avg_return": joined["day_return"].mean(),
        "return_lift": joined["pick_return"].mean() - joined["day_return"].mean(),
        "unique_stocks": picked["股票代號"].nunique(),
        "top_stock_share": picked["股票代號"].value_counts(normalize=True).iloc[0],
        "top_industry_share": picked["主分類"].fillna("unknown").value_counts(normalize=True).iloc[0],
    }


def metric_rows(df: pd.DataFrame, split: str) -> list[dict]:
    rows = []
    specs = [
        ("mlp_absolute_success", "target_success"),
        ("mlp_relative_top20", "relative_top20_label"),
        ("mlp_dual_abs", "target_success"),
        ("mlp_dual_rel", "relative_top20_label"),
        ("mlp_dual_combined", "relative_top20_label"),
    ]
    for score_col, label_col in specs:
        y = df[label_col].to_numpy(dtype=float)
        p = df[score_col].to_numpy(dtype=float)
        rows.append(
            {
                "split": split,
                "model": score_col,
                "rows": len(df),
                "auc": auc_score(y, p),
                "brier": float(np.mean((np.clip(p, 0, 1) - y) ** 2)),
                "label_rate": float(y.mean()),
                "avg_10d_high_close_return": float(df["future_10d_high_close_return"].mean()),
                "same_day_market_success_rate": float(df["daily_market_success_rate"].mean()),
            }
        )
    return rows


def calibration_rows(df: pd.DataFrame, split: str, score_col: str, label_col: str) -> list[dict]:
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


def failure_rows(df: pd.DataFrame, split: str, score_col: str) -> list[dict]:
    picked = df.sort_values(["日期", score_col], ascending=[True, False]).groupby("日期").head(5).copy()
    if picked.empty:
        return []
    picked["failed"] = 1 - picked["target_success"]
    rows = []
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
    data = build_dataset(
        read_csv(paths["stock_daily_all"]),
        read_csv(paths["market_daily"]),
        read_csv(paths["theme_group"]),
    )
    Xdf, feature_names = sequence_features(data, lookback=20)
    data = data[Xdf["has_full_20d_history"] == 1].copy()
    Xdf = Xdf.loc[data.index].drop(columns=["has_full_20d_history"])
    masks = split_masks(data, config)
    train_mask = masks["train"].to_numpy()
    if train_mask.sum() == 0:
        fail("no train rows after 20 day history filter")

    X, _, _ = standardize(Xdf.to_numpy(dtype=np.float32), train_mask)
    y_abs = data["target_success"].to_numpy(dtype=np.float32).reshape(-1, 1)
    y_rel = data["relative_top20_label"].to_numpy(dtype=np.float32).reshape(-1, 1)

    abs_model, abs_loss = train_mlp(X[train_mask], y_abs[train_mask], seed=11)
    rel_model, rel_loss = train_mlp(X[train_mask], y_rel[train_mask], seed=13)
    dual_model, dual_loss = train_mlp(X[train_mask], np.hstack([y_abs[train_mask], y_rel[train_mask]]), seed=17)

    data["mlp_absolute_success"] = predict_mlp(X, abs_model)[:, 0]
    data["mlp_relative_top20"] = predict_mlp(X, rel_model)[:, 0]
    dual_pred = predict_mlp(X, dual_model)
    data["mlp_dual_abs"] = dual_pred[:, 0]
    data["mlp_dual_rel"] = dual_pred[:, 1]
    data["mlp_dual_combined"] = (
        data["mlp_dual_abs"].rank(pct=True) + data["mlp_dual_rel"].rank(pct=True)
    ) / 2.0

    metrics = []
    topk = []
    calibration = []
    failure = []
    for split, mask in masks.items():
        part = data.loc[mask].copy()
        if part.empty:
            continue
        metrics.extend(metric_rows(part, split))
        for score_col, label_col in [
            ("mlp_absolute_success", "target_success"),
            ("mlp_relative_top20", "relative_top20_label"),
            ("mlp_dual_combined", "relative_top20_label"),
        ]:
            calibration.extend(calibration_rows(part, split, score_col, label_col))
        for score_col in ["mlp_absolute_success", "mlp_relative_top20", "mlp_dual_combined"]:
            for k in [3, 5, 10]:
                topk.append(topk_evaluate(part, split, score_col, score_col, k))
                for gate in [0.55, 0.65, 0.75]:
                    topk.append(topk_evaluate(part, split, f"{score_col}_market_strength_gate", score_col, k, gate))
        failure.extend(failure_rows(part, split, "mlp_dual_combined"))

    metrics_df = pd.DataFrame(metrics)
    topk_df = pd.DataFrame(topk)
    calibration_df = pd.DataFrame(calibration)
    failure_df = pd.DataFrame(failure)
    metrics_df.to_csv(METRICS_PATH, index=False, encoding="utf-8-sig")
    topk_df.to_csv(TOPK_PATH, index=False, encoding="utf-8-sig")
    calibration_df.to_csv(CALIBRATION_PATH, index=False, encoding="utf-8-sig")
    failure_df.to_csv(FAILURE_PATH, index=False, encoding="utf-8-sig")

    latest_date = data["日期"].max()
    data[data["日期"] == latest_date].sort_values("mlp_dual_combined", ascending=False).head(50).to_csv(
        LATEST_SCORES_PATH, index=False, encoding="utf-8-sig"
    )

    dev_ok = topk_df[
        (topk_df["split"] == "development")
        & (topk_df["top_k"].isin([3, 5]))
        & (topk_df["days"] >= 20)
        & (topk_df["success_lift"] >= 0.05)
        & (topk_df["return_lift"] > 0)
        & (topk_df["top_stock_share"] <= 0.20)
        & (topk_df["top_industry_share"] <= 0.50)
    ]
    holdout = topk_df[topk_df["split"] == "holdout"].copy()
    passed = []
    for _, row in dev_ok.iterrows():
        h = holdout[
            (holdout["strategy"] == row["strategy"])
            & (holdout["top_k"] == row["top_k"])
            & (holdout["market_strength_gate"].fillna("").astype(str) == str(row["market_strength_gate"]))
        ]
        if h.empty:
            continue
        hr = h.iloc[0]
        if (
            hr["days"] >= 20
            and hr["success_lift"] >= 0.05
            and hr["return_lift"] > 0
            and hr["top_stock_share"] <= 0.20
            and hr["top_industry_share"] <= 0.50
        ):
            passed.append(hr.to_dict())

    status = "rejected"
    reason = "deep learning selection did not pass holdout gates"
    if passed:
        status = "research_pass_needs_manual_review"
        reason = "at least one deep learning Top-K strategy passed holdout gates; calibration still needs review"
    write_formal_default(config, reason)

    best_dev = dev_ok.sort_values(["success_lift", "return_lift"], ascending=False).head(5)
    best_holdout = holdout.sort_values(["success_lift", "return_lift"], ascending=False).head(5)
    lines = [
        "# Deep Learning Selection Decision",
        "",
        f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "- Data sources: three allowed CSV inputs only",
        "- Model: numpy MLP with 20 trading day lookback",
        f"- Rows after 20 day history filter: {len(data)}",
        f"- Feature count: {Xdf.shape[1]}",
        f"- Latest scored date: {latest_date.strftime('%Y-%m-%d')}",
        f"- Status: {status}",
        f"- Formal output: {config['formal_candidate_default']}",
        "",
        "## Training Loss",
        "",
        f"- Absolute model: {abs_loss[0]:.6f} -> {abs_loss[-1]:.6f}",
        f"- Relative model: {rel_loss[0]:.6f} -> {rel_loss[-1]:.6f}",
        f"- Dual model: {dual_loss[0]:.6f} -> {dual_loss[-1]:.6f}",
        "",
        "## Holdout Baseline Context",
        "",
        f"- Holdout same-day market success rate: {fmt_pct(data.loc[masks['holdout'], 'daily_market_success_rate'].mean())}",
        f"- Holdout same-day market 10d high return: {fmt_pct(data.loc[masks['holdout'], 'daily_market_avg_return'].mean())}",
        "",
        "## Best Development Strategies",
        "",
    ]
    if best_dev.empty:
        lines.append("- None passed development gates.")
    else:
        for _, row in best_dev.iterrows():
            lines.append(
                f"- {row['strategy']} Top {int(row['top_k'])}, gate={row['market_strength_gate']}: "
                f"success lift {fmt_pct(row['success_lift'])}, return lift {fmt_pct(row['return_lift'])}, days {int(row['days'])}"
            )
    lines.extend(["", "## Best Holdout Strategies", ""])
    for _, row in best_holdout.iterrows():
        lines.append(
            f"- {row['strategy']} Top {int(row['top_k'])}, gate={row['market_strength_gate']}: "
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
    print("OK: deep learning selection experiment completed")
    print(f"STATUS: {status}")
    print(f"DECISION: {DECISION_PATH}")


if __name__ == "__main__":
    main()

