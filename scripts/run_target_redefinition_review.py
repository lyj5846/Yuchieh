from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "project_config.json"
CONTRACT_PATH = PROJECT_ROOT / "label_layer" / "target_redefinition_contract.md"
BASELINE_PATH = PROJECT_ROOT / "validation_layer" / "target_redefinition_baseline.csv"
DECISION_PATH = PROJECT_ROOT / "decision_layer" / "target_redefinition_decision.md"

PROFIT_THRESHOLD = 0.03
ADVERSE_THRESHOLD = -0.03
LOOKAHEAD_DAYS = 10


def fail(message: str) -> None:
    raise SystemExit(f"FAIL: {message}")


def read_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        config = json.load(f)
    allowed = config.get("allowed_inputs", {})
    if set(allowed) != {"stock_daily_all", "market_daily", "theme_group"}:
        fail("allowed_inputs must contain exactly stock_daily_all, market_daily, theme_group")
    return config


def allowed_paths(config: dict) -> dict[str, Path]:
    paths = {name: Path(value) for name, value in config["allowed_inputs"].items()}
    for name, path in paths.items():
        if not path.exists():
            fail(f"missing allowed input {name}: {path}")
    return paths


def load_inputs(paths: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stock = pd.read_csv(paths["stock_daily_all"], encoding="utf-8-sig")
    market = pd.read_csv(paths["market_daily"], encoding="utf-8-sig")
    theme = pd.read_csv(paths["theme_group"], encoding="utf-8-sig")

    required_stock = {"日期", "股票代號", "開盤價", "最低價", "收盤價"}
    missing_stock = sorted(required_stock - set(stock.columns))
    if missing_stock:
        fail("stock_daily_all missing columns: " + ", ".join(missing_stock))

    if "日期" not in market.columns:
        fail("market_daily missing column: 日期")
    missing_theme = sorted({"股票代號", "主分類", "股票名稱"} - set(theme.columns))
    if missing_theme:
        fail("theme_group missing columns: " + ", ".join(missing_theme))

    for frame in [stock, market]:
        frame["日期"] = pd.to_datetime(frame["日期"], errors="coerce")
        if frame["日期"].isna().any():
            fail("date parsing failed in an allowed input")

    stock["股票代號"] = stock["股票代號"].astype(str).str.strip()
    theme["股票代號"] = theme["股票代號"].astype(str).str.strip()
    for col in ["開盤價", "最低價", "收盤價"]:
        stock[col] = pd.to_numeric(stock[col], errors="coerce")

    return stock, market, theme


def first_hit_day(values: pd.DataFrame, threshold: float, direction: str) -> pd.Series:
    if direction == "ge":
        hits = values.ge(threshold)
    elif direction == "le":
        hits = values.le(threshold)
    else:
        fail(f"unsupported hit direction: {direction}")

    days = np.arange(1, LOOKAHEAD_DAYS + 1, dtype=float)
    hit_matrix = hits.to_numpy(dtype=bool)
    first = np.full(hit_matrix.shape[0], np.nan, dtype=float)
    has_hit = hit_matrix.any(axis=1)
    first[has_hit] = np.argmax(hit_matrix[has_hit], axis=1) + 1
    return pd.Series(first, index=values.index)


def add_target_labels(stock: pd.DataFrame) -> pd.DataFrame:
    out = stock.sort_values(["股票代號", "日期"]).reset_index(drop=True).copy()
    group = out.groupby("股票代號", sort=False)
    out["buy_open_next"] = group["開盤價"].shift(-1)

    future_closes = pd.concat([group["收盤價"].shift(-i) for i in range(1, LOOKAHEAD_DAYS + 1)], axis=1)
    future_lows = pd.concat([group["最低價"].shift(-i) for i in range(1, LOOKAHEAD_DAYS + 1)], axis=1)
    future_closes.columns = [f"future_close_{i}" for i in range(1, LOOKAHEAD_DAYS + 1)]
    future_lows.columns = [f"future_low_{i}" for i in range(1, LOOKAHEAD_DAYS + 1)]

    out["future_window_count"] = future_closes.notna().sum(axis=1)
    out["future_low_window_count"] = future_lows.notna().sum(axis=1)
    out["label_complete"] = (
        (out["future_window_count"] == LOOKAHEAD_DAYS)
        & (out["future_low_window_count"] == LOOKAHEAD_DAYS)
        & out["buy_open_next"].notna()
        & (out["buy_open_next"] > 0)
    )

    close_returns = future_closes.div(out["buy_open_next"], axis=0) - 1.0
    low_returns = future_lows.div(out["buy_open_next"], axis=0) - 1.0
    close_returns = close_returns.replace([np.inf, -np.inf], np.nan)
    low_returns = low_returns.replace([np.inf, -np.inf], np.nan)

    out["future_10d_high_close_return"] = close_returns.max(axis=1)
    out["future_10d_day10_close_return"] = close_returns.iloc[:, -1]
    out["max_adverse_return"] = low_returns.min(axis=1)
    out["target_success"] = (
        out["label_complete"] & (out["future_10d_high_close_return"] >= PROFIT_THRESHOLD)
    ).astype(int)

    success_day = first_hit_day(close_returns, PROFIT_THRESHOLD, "ge")
    adverse_day = first_hit_day(low_returns, ADVERSE_THRESHOLD, "le")
    out["profit_event_day"] = success_day
    out["adverse_event_day"] = adverse_day

    has_success = out["profit_event_day"].notna()
    has_adverse = out["adverse_event_day"].notna()
    success_first = has_success & (~has_adverse | (out["profit_event_day"] < out["adverse_event_day"]))
    adverse_first = has_adverse & (~has_success | (out["adverse_event_day"] <= out["profit_event_day"]))

    out["first_event_day"] = np.nan
    out.loc[success_first, "first_event_day"] = out.loc[success_first, "profit_event_day"]
    out.loc[adverse_first, "first_event_day"] = out.loc[adverse_first, "adverse_event_day"]
    out["first_event_type"] = "none"
    out.loc[success_first, "first_event_type"] = "profit_first"
    out.loc[adverse_first, "first_event_type"] = "adverse_first"

    out["risk_adjusted_10d_success"] = (
        out["label_complete"] & success_first
    ).astype(int)
    out["old_success_but_risk_failed"] = (
        (out["target_success"] == 1) & (out["risk_adjusted_10d_success"] == 0)
    ).astype(int)
    out["same_day_both_event"] = (
        out["label_complete"]
        & has_success
        & has_adverse
        & out["profit_event_day"].eq(out["adverse_event_day"])
    ).astype(int)

    out["realized_10d_trade_return"] = out["future_10d_day10_close_return"]
    out.loc[success_first, "realized_10d_trade_return"] = PROFIT_THRESHOLD
    out.loc[adverse_first, "realized_10d_trade_return"] = ADVERSE_THRESHOLD
    out.loc[~out["label_complete"], "realized_10d_trade_return"] = np.nan

    return out


def add_split(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    out = df.copy()
    train_end = pd.Timestamp(config["time_split"]["train_end"])
    dev_start = pd.Timestamp(config["time_split"]["dev_start"])
    dev_end = pd.Timestamp(config["time_split"]["dev_end"])
    holdout_start = pd.Timestamp(config["time_split"]["holdout_start"])
    out["split"] = "tracking"
    complete = out["label_complete"]
    out.loc[complete & (out["日期"] <= train_end), "split"] = "train"
    out.loc[complete & (out["日期"] >= dev_start) & (out["日期"] <= dev_end), "split"] = "development"
    out.loc[complete & (out["日期"] >= holdout_start), "split"] = "holdout"
    return out


def enrich_theme(df: pd.DataFrame, theme: pd.DataFrame) -> pd.DataFrame:
    keep = [c for c in ["股票代號", "股票名稱", "主分類", "子分類"] if c in theme.columns]
    theme_small = theme[keep].drop_duplicates("股票代號")
    out = df.merge(theme_small, on="股票代號", how="left")
    out["股票名稱"] = out["股票名稱"].fillna("")
    out["主分類"] = out["主分類"].fillna("unknown")
    out["子分類"] = out["子分類"].fillna("unknown")
    return out


def safe_mean(series: pd.Series) -> float:
    value = series.mean()
    return float(value) if pd.notna(value) else math.nan


def safe_share(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return math.nan
    counts = frame[column].fillna("unknown").value_counts(normalize=True)
    if counts.empty:
        return math.nan
    return float(counts.iloc[0])


def summary_record(section: str, split: str, group: str, frame: pd.DataFrame) -> dict:
    completed = frame[frame["label_complete"]].copy()
    old_success = int(completed["target_success"].sum()) if not completed.empty else 0
    risk_success = int(completed["risk_adjusted_10d_success"].sum()) if not completed.empty else 0
    removed = int(completed["old_success_but_risk_failed"].sum()) if not completed.empty else 0
    removed_among_old = removed / old_success if old_success else math.nan
    return {
        "section": section,
        "split": split,
        "group": group,
        "rows": int(len(completed)),
        "days": int(completed["日期"].nunique()) if not completed.empty else 0,
        "stocks": int(completed["股票代號"].nunique()) if not completed.empty else 0,
        "old_success_count": old_success,
        "risk_adjusted_success_count": risk_success,
        "old_success_rate": safe_mean(completed["target_success"]) if not completed.empty else math.nan,
        "risk_adjusted_success_rate": safe_mean(completed["risk_adjusted_10d_success"]) if not completed.empty else math.nan,
        "success_rate_delta": (
            safe_mean(completed["risk_adjusted_10d_success"]) - safe_mean(completed["target_success"])
            if not completed.empty
            else math.nan
        ),
        "old_success_but_risk_failed_count": removed,
        "old_success_but_risk_failed_rate": safe_mean(completed["old_success_but_risk_failed"]) if not completed.empty else math.nan,
        "old_success_but_risk_failed_among_old_success": removed_among_old,
        "same_day_both_event_count": int(completed["same_day_both_event"].sum()) if not completed.empty else 0,
        "avg_10d_high_close_return": safe_mean(completed["future_10d_high_close_return"]) if not completed.empty else math.nan,
        "avg_max_adverse_return": safe_mean(completed["max_adverse_return"]) if not completed.empty else math.nan,
        "avg_realized_10d_trade_return": safe_mean(completed["realized_10d_trade_return"]) if not completed.empty else math.nan,
        "top_stock_share": safe_share(completed, "股票代號"),
        "top_industry_share": safe_share(completed, "主分類"),
    }


def build_baseline(df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []
    completed = df[df["label_complete"]].copy()

    for split in ["train", "development", "holdout"]:
        records.append(summary_record("overall", split, split, completed[completed["split"].eq(split)]))

    month_frame = completed.copy()
    month_frame["month"] = month_frame["日期"].dt.strftime("%Y-%m")
    for (split, month), part in month_frame.groupby(["split", "month"], sort=True):
        records.append(summary_record("monthly", split, month, part))

    for split in ["train", "development", "holdout"]:
        split_frame = completed[completed["split"].eq(split)]
        for industry, part in split_frame.groupby("主分類", sort=True):
            records.append(summary_record("industry", split, str(industry), part))

    for split in ["train", "development", "holdout"]:
        split_frame = completed[completed["split"].eq(split)]
        top_stocks = split_frame["股票代號"].value_counts().head(10).index
        for stock_code in top_stocks:
            part = split_frame[split_frame["股票代號"].eq(stock_code)]
            stock_name = part["股票名稱"].dropna().astype(str).head(1)
            label = f"{stock_code} {stock_name.iloc[0] if not stock_name.empty else ''}".strip()
            records.append(summary_record("top_stock", split, label, part))

    return pd.DataFrame(records)


def pct(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value:.2%}"


def format_number(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    if isinstance(value, (int, np.integer)):
        return str(value)
    return f"{value:.4f}"


def target_viability(overall: pd.DataFrame) -> tuple[str, list[str]]:
    reasons: list[str] = []
    holdout = overall[(overall["section"].eq("overall")) & (overall["split"].eq("holdout"))]
    train = overall[(overall["section"].eq("overall")) & (overall["split"].eq("train"))]
    dev = overall[(overall["section"].eq("overall")) & (overall["split"].eq("development"))]
    if holdout.empty or train.empty or dev.empty:
        return "not_viable_missing_split", ["missing train, development, or holdout summary"]

    split_rows = {
        "train": train.iloc[0],
        "development": dev.iloc[0],
        "holdout": holdout.iloc[0],
    }
    for split, row in split_rows.items():
        if row["rows"] < 1000:
            reasons.append(f"{split} has fewer than 1000 completed rows")
        if row["risk_adjusted_success_count"] < 100:
            reasons.append(f"{split} has fewer than 100 risk-adjusted success rows")

    removed_among_old = float(holdout.iloc[0]["old_success_but_risk_failed_among_old_success"])
    if pd.isna(removed_among_old) or removed_among_old <= 0:
        reasons.append("risk-adjusted target did not remove any old holdout successes")

    if reasons:
        return "needs_review_before_training", reasons
    return "label_viable_for_training_review", [
        "all splits have enough completed rows",
        "all splits retain enough risk-adjusted success rows",
        "holdout has old successes filtered by the adverse-first rule",
    ]


def write_contract(data_latest: str) -> None:
    lines = [
        "# Target Redefinition Contract",
        "",
        f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Data latest date: {data_latest}",
        "- Scope: label-only review; no model training; no formal candidates.",
        "",
        "## Old Target",
        "",
        "- Signal time: after signal-day close.",
        "- Buy assumption: next trading day open.",
        "- Success: any close in the next 1 to 10 trading days is at least +3% above buy open.",
        "- Unfinished: missing next-day open or incomplete 10 trading day window is tracking-only.",
        "",
        "## Proposed Risk-Adjusted Target",
        "",
        "- Buy assumption stays the same: next trading day open.",
        "- Profit event: any close in the next 1 to 10 trading days is at least +3% above buy open.",
        "- Adverse event: any low in the next 1 to 10 trading days is at least -3% below buy open.",
        "- Success: the profit event happens before the adverse event.",
        "- Failure: the adverse event happens before the profit event, no profit event happens, or both happen on the same day.",
        "- Conservative tie rule: if +3% close and -3% low happen on the same day, adverse event wins.",
        "- This target is a label candidate, not a calibrated probability and not a formal stock recommendation.",
        "",
        "## Derived Fields",
        "",
        "- `risk_adjusted_10d_success`: proposed success label.",
        "- `max_adverse_return`: worst low return within the 10 trading day window.",
        "- `first_event_day`: first profit/adverse event day, 1 to 10.",
        "- `realized_10d_trade_return`: +3% on profit-first, -3% on adverse-first, otherwise day-10 close return.",
        "- `old_success_but_risk_failed`: old +3% success that fails the adverse-first rule.",
        "",
        "## Boundary",
        "",
        "- This contract does not modify `target_success` in the current main model.",
        "- This contract does not write formal output.",
        "- If this target is viable, the next step is a separate main-label contract change and model retraining.",
        "",
    ]
    CONTRACT_PATH.write_text("\n".join(lines), encoding="utf-8")


def write_decision(data_latest: str, baseline: pd.DataFrame, status: str, reasons: list[str]) -> None:
    overall = baseline[baseline["section"].eq("overall")].copy()
    by_split = {row["split"]: row for _, row in overall.iterrows()}
    lines = [
        "# Target Redefinition Decision",
        "",
        f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Data latest date: {data_latest}",
        f"- Decision status: `{status}`",
        "- Formal output: unchanged by this review.",
        "",
        "## 白話結論",
        "",
    ]
    if status == "label_viable_for_training_review":
        lines.extend(
            [
                "新目標可以進入下一步訓練審查，但還不是正式模型。",
                "",
                "意思是：先收盤 +3% 才算乾淨成功；如果先跌到 -3%，即使後面又漲到 +3%，這筆不再算成功。",
            ]
        )
    else:
        lines.extend(
            [
                "新目標目前只完成審查，還不能直接拿去訓練正式模型。",
                "",
                "原因是樣本量或風險過濾效果還需要再確認。",
            ]
        )
    lines.extend(["", "## Split Summary", ""])

    for split in ["train", "development", "holdout"]:
        row = by_split.get(split)
        if row is None:
            continue
        lines.extend(
            [
                f"### {split}",
                "",
                f"- Completed rows: {int(row['rows'])}",
                f"- Old success rate: {pct(row['old_success_rate'])}",
                f"- Risk-adjusted success rate: {pct(row['risk_adjusted_success_rate'])}",
                f"- Old successes filtered by risk rule: {int(row['old_success_but_risk_failed_count'])}",
                f"- Filtered among old successes: {pct(row['old_success_but_risk_failed_among_old_success'])}",
                f"- Average high-close return: {pct(row['avg_10d_high_close_return'])}",
                f"- Average adverse low return: {pct(row['avg_max_adverse_return'])}",
                f"- Average realized rule return: {pct(row['avg_realized_10d_trade_return'])}",
                "",
            ]
        )

    lines.extend(["## Decision Reasons", ""])
    for reason in reasons:
        lines.append(f"- {reason}")
    lines.extend(
        [
            "",
            "## Next Step",
            "",
            "If accepted, modify the main label contract to train against `risk_adjusted_10d_success` in a separate step.",
            "Do not update formal candidates from this review.",
            "",
            "## Outputs",
            "",
            f"- `{CONTRACT_PATH.relative_to(PROJECT_ROOT)}`",
            f"- `{BASELINE_PATH.relative_to(PROJECT_ROOT)}`",
            "",
        ]
    )
    DECISION_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    config = read_config()
    paths = allowed_paths(config)
    stock, market, theme = load_inputs(paths)
    data_latest = stock["日期"].max().strftime("%Y-%m-%d")
    market_latest = market["日期"].max().strftime("%Y-%m-%d")
    if data_latest != market_latest:
        fail(f"stock and market latest dates differ: stock={data_latest}, market={market_latest}")

    labeled = add_target_labels(stock)
    labeled = add_split(labeled, config)
    labeled = enrich_theme(labeled, theme)
    baseline = build_baseline(labeled)
    status, reasons = target_viability(baseline)

    write_contract(data_latest)
    baseline.to_csv(BASELINE_PATH, index=False, encoding="utf-8-sig")
    write_decision(data_latest, baseline, status, reasons)

    print("OK: target redefinition review completed")
    print(f"STATUS: {status}")
    print(f"CONTRACT: {CONTRACT_PATH}")
    print(f"BASELINE: {BASELINE_PATH}")
    print(f"DECISION: {DECISION_PATH}")


if __name__ == "__main__":
    main()
