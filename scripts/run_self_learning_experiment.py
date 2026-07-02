from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "project_config.json"
EXPERIMENT_DIR = PROJECT_ROOT / "research_layer"
FORMAL_DIR = PROJECT_ROOT / "research_layer" / "formal_write_disabled"
LOG_PATH = EXPERIMENT_DIR / "self_learning_experiment_log.md"
METRICS_PATH = EXPERIMENT_DIR / "self_learning_metrics.csv"
CALIBRATION_PATH = EXPERIMENT_DIR / "self_learning_calibration.csv"
TOPK_PATH = EXPERIMENT_DIR / "self_learning_topk.csv"
FORMAL_STATUS_PATH = FORMAL_DIR / "formal_status.md"
FORMAL_CANDIDATES_PATH = FORMAL_DIR / "formal_candidates.csv"


@dataclass
class EvalResult:
    name: str
    split: str
    rows: int
    success_rate: float
    auc: float
    brier: float
    avg_high_return: float


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def fail(message: str) -> None:
    raise SystemExit(f"FAIL: {message}")


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def validate_inputs(config: dict) -> dict[str, Path]:
    paths = {name: Path(value) for name, value in config["allowed_inputs"].items()}
    for name, path in paths.items():
        if not path.exists():
            fail(f"missing input {name}: {path}")
        if name == "theme_group" and PROJECT_ROOT not in path.parents:
            fail("theme_group must live inside this clean project")
    return paths


def build_label_dataset(stock: pd.DataFrame, market: pd.DataFrame, theme: pd.DataFrame) -> pd.DataFrame:
    stock = stock.copy()
    market = market.copy()
    theme = theme.copy()

    stock["日期"] = pd.to_datetime(stock["日期"])
    market["日期"] = pd.to_datetime(market["日期"])
    stock["股票代號"] = stock["股票代號"].astype(str)
    theme["股票代號"] = theme["股票代號"].astype(str)

    stock = stock.sort_values(["股票代號", "日期"]).reset_index(drop=True)
    grouped = stock.groupby("股票代號", sort=False)
    stock["buy_open_next"] = grouped["開盤價"].shift(-1)

    future_closes = [grouped["收盤價"].shift(-i) for i in range(1, 11)]
    future_close_frame = pd.concat(future_closes, axis=1)
    stock["future_10d_high_close"] = future_close_frame.max(axis=1)
    stock["has_full_10d_window"] = future_close_frame.notna().all(axis=1) & stock["buy_open_next"].notna()
    stock["has_full_10d_window"] = stock["has_full_10d_window"] & (stock["buy_open_next"] > 0)
    stock["target_success"] = (
        stock["future_10d_high_close"] >= stock["buy_open_next"] * 1.03
    ).astype(int)
    stock["future_10d_high_close_return"] = stock["future_10d_high_close"] / stock["buy_open_next"] - 1.0
    stock["has_full_10d_window"] = stock["has_full_10d_window"] & np.isfinite(
        stock["future_10d_high_close_return"]
    )

    merged = stock.merge(market, on="日期", how="left", suffixes=("", "_market"))
    merged = merged.merge(
        theme[["股票代號", "股票名稱", "主分類", "子分類"]],
        on="股票代號",
        how="left",
    )
    return merged


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    forbidden = {
        "日期",
        "股票代號",
        "股票名稱",
        "buy_open_next",
        "future_10d_high_close",
        "has_full_10d_window",
        "target_success",
        "future_10d_high_close_return",
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
    features = pd.concat([numeric, category], axis=1)
    return features, list(features.columns)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def train_logistic(X: np.ndarray, y: np.ndarray, iterations: int = 220) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    Xs = (X - mean) / std
    Xs = np.column_stack([np.ones(len(Xs)), Xs])
    w = np.zeros(Xs.shape[1], dtype=float)
    lr = 0.05
    reg = 0.001
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


def evaluate(name: str, split: str, y: np.ndarray, p: np.ndarray, returns: np.ndarray) -> EvalResult:
    return EvalResult(
        name=name,
        split=split,
        rows=len(y),
        success_rate=float(y.mean()) if len(y) else float("nan"),
        auc=auc_score(y, p) if len(y) else float("nan"),
        brier=float(np.mean((p - y) ** 2)) if len(y) else float("nan"),
        avg_high_return=float(np.nanmean(returns)) if len(y) else float("nan"),
    )


def calibration_rows(split: str, y: np.ndarray, p: np.ndarray) -> list[dict]:
    if len(y) == 0:
        return []
    bins = pd.qcut(pd.Series(p), q=5, duplicates="drop")
    out = []
    for band, idx in pd.Series(range(len(y))).groupby(bins):
        ids = idx.to_numpy()
        out.append(
            {
                "split": split,
                "band": str(band),
                "rows": len(ids),
                "estimated_probability": float(p[ids].mean()),
                "actual_hit_rate": float(y[ids].mean()),
                "calibration_error": float(abs(p[ids].mean() - y[ids].mean())),
            }
        )
    return out


def topk_rows(split_df: pd.DataFrame, split: str, prob_col: str) -> list[dict]:
    rows = []
    for k in [3, 5, 10]:
        picked = (
            split_df.sort_values(["日期", prob_col], ascending=[True, False])
            .groupby("日期")
            .head(k)
        )
        if picked.empty:
            continue
        rows.append(
            {
                "split": split,
                "top_k": k,
                "rows": len(picked),
                "days": picked["日期"].nunique(),
                "success_rate": picked["target_success"].mean(),
                "avg_10d_high_close_return": picked["future_10d_high_close_return"].mean(),
                "avg_daily_candidates": len(picked) / picked["日期"].nunique(),
            }
        )
    return rows


def fmt_pct(value: float) -> str:
    if value is None or math.isnan(value):
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
    data = build_label_dataset(stock, market, theme)
    usable = data[data["has_full_10d_window"]].copy()

    split_cfg = config["time_split"]
    train_mask = usable["日期"] <= pd.Timestamp(split_cfg["train_end"])
    dev_mask = (usable["日期"] >= pd.Timestamp(split_cfg["dev_start"])) & (
        usable["日期"] <= pd.Timestamp(split_cfg["dev_end"])
    )
    holdout_mask = usable["日期"] >= pd.Timestamp(split_cfg["holdout_start"])

    features, feature_names = prepare_features(usable)
    X = features.to_numpy(dtype=float)
    y = usable["target_success"].to_numpy(dtype=float)
    returns = usable["future_10d_high_close_return"].to_numpy(dtype=float)

    X_train = X[train_mask.to_numpy()]
    y_train = y[train_mask.to_numpy()]
    if len(y_train) == 0:
        fail("no training rows after split")

    baseline_prob = float(y_train.mean())
    w, mean, std = train_logistic(X_train, y_train)
    usable["baseline_probability"] = baseline_prob
    usable["model_probability"] = predict_logistic(X, w, mean, std)

    metrics = []
    calibration = []
    topk = []
    for split, mask in [
        ("train", train_mask),
        ("development", dev_mask),
        ("holdout", holdout_mask),
    ]:
        ids = mask.to_numpy()
        split_y = y[ids]
        split_returns = returns[ids]
        split_df = usable.loc[mask].copy()
        for model_name, prob_col in [
            ("baseline", "baseline_probability"),
            ("clean_logistic", "model_probability"),
        ]:
            p = split_df[prob_col].to_numpy(dtype=float)
            metrics.append(evaluate(model_name, split, split_y, p, split_returns).__dict__)
            calibration.extend(calibration_rows(f"{split}:{model_name}", split_y, p))
        topk.extend(topk_rows(split_df, split, "model_probability"))

    pd.DataFrame(metrics).to_csv(METRICS_PATH, index=False, encoding="utf-8-sig")
    pd.DataFrame(calibration).to_csv(CALIBRATION_PATH, index=False, encoding="utf-8-sig")
    pd.DataFrame(topk).to_csv(TOPK_PATH, index=False, encoding="utf-8-sig")

    metric_df = pd.DataFrame(metrics)
    holdout_model = metric_df[
        (metric_df["name"] == "clean_logistic") & (metric_df["split"] == "holdout")
    ].iloc[0]
    holdout_base = metric_df[
        (metric_df["name"] == "baseline") & (metric_df["split"] == "holdout")
    ].iloc[0]
    pass_holdout = bool(
        holdout_model["brier"] < holdout_base["brier"]
        and holdout_model["auc"] >= 0.55
    )

    status = "rejected"
    reason = "holdout validation not passed"
    if pass_holdout:
        status = "research_pass_needs_manual_review"
        reason = "holdout metrics passed minimum gate, but formal promotion still requires calibration and concentration review"
    write_formal_default(config, reason)

    lines = [
        "# Self-Learning Experiment Log",
        "",
        f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "- Data sources: three allowed CSV inputs only",
        "- Target: next trading day open buy, next 10 trading days any close +3%",
        f"- Full-window rows: {len(usable)}",
        f"- Feature count: {len(feature_names)}",
        f"- Experiment status: {status}",
        f"- Formal output: {config['formal_candidate_default']}",
        "",
        "## Split Summary",
        "",
        f"- Train rows: {int(train_mask.sum())}",
        f"- Development rows: {int(dev_mask.sum())}",
        f"- Holdout rows: {int(holdout_mask.sum())}",
        "",
        "## Holdout Result",
        "",
        f"- Baseline Brier: {holdout_base['brier']:.6f}",
        f"- Model Brier: {holdout_model['brier']:.6f}",
        f"- Model AUC: {holdout_model['auc']:.4f}",
        f"- Model actual hit rate: {fmt_pct(holdout_model['success_rate'])}",
        f"- Model average 10d high close return: {fmt_pct(holdout_model['avg_high_return'])}",
        "",
        "## Decision",
        "",
        f"- {reason}.",
        "- No formal stock candidate is produced by this run.",
        "",
        "## Output Files",
        "",
        f"- Metrics: `{METRICS_PATH}`",
        f"- Calibration: `{CALIBRATION_PATH}`",
        f"- Top-K: `{TOPK_PATH}`",
        f"- Formal status: `{FORMAL_STATUS_PATH}`",
    ]
    LOG_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"OK: self-learning experiment completed")
    print(f"STATUS: {status}")
    print(f"LOG: {LOG_PATH}")


if __name__ == "__main__":
    main()

