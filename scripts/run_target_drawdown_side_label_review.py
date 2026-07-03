from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "project_config.json"
CONTRACT_PATH = PROJECT_ROOT / "label_layer" / "target_drawdown_side_label_contract.md"
BASELINE_PATH = PROJECT_ROOT / "validation_layer" / "target_drawdown_side_label_baseline.csv"
DECISION_MD_PATH = PROJECT_ROOT / "decision_layer" / "target_drawdown_side_label_decision.md"
DECISION_JSON_PATH = PROJECT_ROOT / "decision_layer" / "target_drawdown_side_label_decision.json"

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

    hit_matrix = hits.to_numpy(dtype=bool)
    first = np.full(hit_matrix.shape[0], np.nan, dtype=float)
    has_hit = hit_matrix.any(axis=1)
    first[has_hit] = np.argmax(hit_matrix[has_hit], axis=1) + 1
    return pd.Series(first, index=values.index)


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


def add_drawdown_side_labels(stock: pd.DataFrame, config: dict, theme: pd.DataFrame) -> pd.DataFrame:
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

    profit_day = first_hit_day(close_returns, PROFIT_THRESHOLD, "ge")
    adverse_day = first_hit_day(low_returns, ADVERSE_THRESHOLD, "le")
    has_profit = profit_day.notna()
    has_adverse = adverse_day.notna()

    adverse_before_or_same_profit = has_profit & has_adverse & adverse_day.le(profit_day)
    adverse_before_profit = has_profit & has_adverse & adverse_day.lt(profit_day)
    same_day_profit_and_adverse = has_profit & has_adverse & adverse_day.eq(profit_day)
    profit_before_adverse = has_profit & (~has_adverse | profit_day.lt(adverse_day))

    out["profit_event_day"] = profit_day
    out["adverse_event_day"] = adverse_day
    out["future_10d_high_close_return"] = close_returns.max(axis=1)
    out["future_10d_day10_close_return"] = close_returns.iloc[:, -1]
    out["max_adverse_return_10d"] = low_returns.min(axis=1)
    out["old_touch_3pct_success"] = (out["label_complete"] & has_profit).astype(int)
    out["risk_adjusted_3pct_before_minus3pct_success"] = (
        out["label_complete"] & profit_before_adverse
    ).astype(int)
    out["drawdown_minus3_before_or_same_success"] = (
        out["label_complete"] & adverse_before_or_same_profit
    ).astype(int)
    out["drawdown_minus3_before_success"] = (
        out["label_complete"] & adverse_before_profit
    ).astype(int)
    out["same_day_profit_and_drawdown_minus3"] = (
        out["label_complete"] & same_day_profit_and_adverse
    ).astype(int)
    out["hit_minus3_low_anytime_10d"] = (out["label_complete"] & has_adverse).astype(int)
    out["clean_success_label"] = (
        out["label_complete"] & has_profit & ~adverse_before_or_same_profit
    ).astype(int)
    out["painful_success_label"] = (
        out["label_complete"] & has_profit & adverse_before_or_same_profit
    ).astype(int)
    out["failed_without_success_label"] = (
        out["label_complete"] & ~has_profit
    ).astype(int)
    out["drawdown_risk_side_label"] = out["hit_minus3_low_anytime_10d"]

    out = add_split(out, config)
    out = enrich_theme(out, theme)
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
    old_success_count = int(completed["old_touch_3pct_success"].sum()) if not completed.empty else 0
    painful_success_count = int(completed["painful_success_label"].sum()) if not completed.empty else 0
    clean_success_count = int(completed["clean_success_label"].sum()) if not completed.empty else 0
    return {
        "section": section,
        "split": split,
        "group": group,
        "rows": int(len(completed)),
        "days": int(completed["日期"].nunique()) if not completed.empty else 0,
        "stocks": int(completed["股票代號"].nunique()) if not completed.empty else 0,
        "old_touch_success_count": old_success_count,
        "old_touch_success_rate": safe_mean(completed["old_touch_3pct_success"]) if not completed.empty else math.nan,
        "risk_adjusted_hard_success_count": int(completed["risk_adjusted_3pct_before_minus3pct_success"].sum()) if not completed.empty else 0,
        "risk_adjusted_hard_success_rate": safe_mean(completed["risk_adjusted_3pct_before_minus3pct_success"]) if not completed.empty else math.nan,
        "proposed_primary_success_count": old_success_count,
        "proposed_primary_success_rate": safe_mean(completed["old_touch_3pct_success"]) if not completed.empty else math.nan,
        "clean_success_count": clean_success_count,
        "clean_success_rate": safe_mean(completed["clean_success_label"]) if not completed.empty else math.nan,
        "painful_success_count": painful_success_count,
        "painful_success_rate": safe_mean(completed["painful_success_label"]) if not completed.empty else math.nan,
        "painful_success_among_success_rate": painful_success_count / old_success_count if old_success_count else math.nan,
        "minus3_anytime_rate": safe_mean(completed["hit_minus3_low_anytime_10d"]) if not completed.empty else math.nan,
        "minus3_before_or_same_success_rate": safe_mean(completed["drawdown_minus3_before_or_same_success"]) if not completed.empty else math.nan,
        "same_day_profit_and_minus3_count": int(completed["same_day_profit_and_drawdown_minus3"].sum()) if not completed.empty else 0,
        "avg_10d_high_close_return": safe_mean(completed["future_10d_high_close_return"]) if not completed.empty else math.nan,
        "avg_day10_close_return": safe_mean(completed["future_10d_day10_close_return"]) if not completed.empty else math.nan,
        "avg_max_adverse_return_10d": safe_mean(completed["max_adverse_return_10d"]) if not completed.empty else math.nan,
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

    return pd.DataFrame(records)


def pct(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value:.2%}"


def target_viability(baseline: pd.DataFrame) -> tuple[str, list[str]]:
    reasons: list[str] = []
    overall = baseline[baseline["section"].eq("overall")]
    by_split = {row["split"]: row for _, row in overall.iterrows()}
    if not {"train", "development", "holdout"}.issubset(by_split):
        return "not_viable_missing_split", ["missing train, development, or holdout summary"]

    for split, row in by_split.items():
        if int(row["rows"]) < 1000:
            reasons.append(f"{split} has fewer than 1000 completed rows")
        if int(row["old_touch_success_count"]) < 100:
            reasons.append(f"{split} has fewer than 100 primary success rows")
        if int(row["painful_success_count"]) <= 0:
            reasons.append(f"{split} has no painful success rows to learn as risk side labels")

    holdout = by_split["holdout"]
    if float(holdout["old_touch_success_rate"]) <= float(holdout["risk_adjusted_hard_success_rate"]):
        reasons.append("holdout side-label primary target should preserve more successes than hard risk-adjusted target")
    painful_share = float(holdout["painful_success_among_success_rate"])
    if pd.isna(painful_share) or painful_share <= 0:
        reasons.append("holdout has no success cases that would be recovered from the hard -3% rule")

    if reasons:
        return "needs_review_before_training", reasons
    return "side_label_target_contract_ready", [
        "primary success remains the 10-day +3% close touch rule",
        "-3% drawdown is retained as a risk side label instead of automatic failure",
        "all splits have enough completed rows and recovered success examples",
    ]


def write_contract(data_latest: str) -> None:
    lines = [
        "# Target Drawdown Side-Label Contract",
        "",
        f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Data latest date: {data_latest}",
        "- Scope: label-only review; no model training; no formal candidates.",
        "",
        "## Primary Target",
        "",
        "- Signal time: after signal-day close.",
        "- Buy assumption: next trading day open.",
        "- `target_success`: any close in the next 1 to 10 trading days is at least +3% above buy open.",
        "- If price first drops to -3% but later reaches +3%, `target_success` still remains success.",
        "- Unfinished: missing next-day open or incomplete 10 trading day window is tracking-only.",
        "",
        "## Drawdown Risk Side Labels",
        "",
        "- `max_adverse_return_10d`: worst low return within the 10 trading day window.",
        "- `hit_minus3_low_anytime_10d`: whether any low in the window reaches -3% below buy open.",
        "- `drawdown_minus3_before_or_same_success`: success path touched -3% before or on the same day as +3% close.",
        "- `clean_success_label`: +3% success without -3% low before or on the success day.",
        "- `painful_success_label`: +3% success after or on the same day as a -3% low.",
        "",
        "## Boundary",
        "",
        "- This contract does not use -3% as an automatic failure.",
        "- This contract does not modify the current main model target.",
        "- This contract does not write formal output.",
        "- If accepted, the next red-light step is to update the main label contract and retrain.",
        "",
    ]
    CONTRACT_PATH.write_text("\n".join(lines), encoding="utf-8")


def write_decision(data_latest: str, baseline: pd.DataFrame, status: str, reasons: list[str]) -> None:
    overall = baseline[baseline["section"].eq("overall")].copy()
    by_split = {row["split"]: row for _, row in overall.iterrows()}
    holdout = by_split.get("holdout")
    lines = [
        "# Target Drawdown Side-Label Decision",
        "",
        f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Data latest date: {data_latest}",
        f"- Decision status: `{status}`",
        "- Formal output: unchanged by this review.",
        "",
        "## 白話結論",
        "",
        "這個審查把 `-3%` 從硬性失敗改成風險旁支標籤。",
        "",
        "也就是：如果 100 買進，先跌到 97，後來 10 日內收盤到 103 以上，主目標仍算成功；但會另外記錄這是一筆先經過 -3% 回撤的高風險成功。",
        "",
    ]
    if holdout is not None:
        lines.extend(
            [
                "## Holdout 重點",
                "",
                f"- 原本 +3% 觸及成功率: {pct(holdout['old_touch_success_rate'])}",
                f"- 硬性不能先 -3% 成功率: {pct(holdout['risk_adjusted_hard_success_rate'])}",
                f"- 旁支標籤主成功率: {pct(holdout['proposed_primary_success_rate'])}",
                f"- 先碰 -3% 但後來 +3% 的成功占成功樣本: {pct(holdout['painful_success_among_success_rate'])}",
                f"- 平均 10 日最高收盤報酬: {pct(holdout['avg_10d_high_close_return'])}",
                f"- 平均最大不利低點: {pct(holdout['avg_max_adverse_return_10d'])}",
                "",
            ]
        )

    lines.extend(["## Split Summary", ""])
    for split in ["train", "development", "holdout"]:
        row = by_split.get(split)
        if row is None:
            continue
        lines.extend(
            [
                f"### {split}",
                "",
                f"- Completed rows: {int(row['rows'])}",
                f"- Primary +3% success rate: {pct(row['proposed_primary_success_rate'])}",
                f"- Hard -3% rule success rate: {pct(row['risk_adjusted_hard_success_rate'])}",
                f"- Clean success rate: {pct(row['clean_success_rate'])}",
                f"- Painful success rate: {pct(row['painful_success_rate'])}",
                f"- Painful among successes: {pct(row['painful_success_among_success_rate'])}",
                f"- Any -3% low rate: {pct(row['minus3_anytime_rate'])}",
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
            "If accepted, update the main model label contract so `target_success` returns to the +3% touch rule and the -3% path becomes a risk side label.",
            "That next step is a red-light change because it changes the model target and requires retraining.",
            "",
            "## Outputs",
            "",
            f"- `{CONTRACT_PATH.relative_to(PROJECT_ROOT)}`",
            f"- `{BASELINE_PATH.relative_to(PROJECT_ROOT)}`",
            f"- `{DECISION_JSON_PATH.relative_to(PROJECT_ROOT)}`",
            "",
        ]
    )
    DECISION_MD_PATH.write_text("\n".join(lines), encoding="utf-8")

    decision = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_latest_date": data_latest,
        "decision_status": status,
        "formal_output_changed": False,
        "recommended_next_step": "update_main_label_contract_to_drawdown_side_labels",
        "red_light_required_for_next_step": True,
        "reasons": reasons,
        "holdout": holdout.to_dict() if holdout is not None else {},
    }
    DECISION_JSON_PATH.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    config = read_config()
    paths = allowed_paths(config)
    stock, market, theme = load_inputs(paths)
    data_latest = stock["日期"].max().strftime("%Y-%m-%d")
    market_latest = market["日期"].max().strftime("%Y-%m-%d")
    if data_latest != market_latest:
        fail(f"stock and market latest dates differ: stock={data_latest}, market={market_latest}")

    labeled = add_drawdown_side_labels(stock, config, theme)
    baseline = build_baseline(labeled)
    status, reasons = target_viability(baseline)

    write_contract(data_latest)
    baseline.to_csv(BASELINE_PATH, index=False, encoding="utf-8-sig")
    write_decision(data_latest, baseline, status, reasons)

    print("OK: target drawdown side-label review completed")
    print(f"STATUS: {status}")
    print(f"CONTRACT: {CONTRACT_PATH}")
    print(f"BASELINE: {BASELINE_PATH}")
    print(f"DECISION: {DECISION_MD_PATH}")


if __name__ == "__main__":
    main()
