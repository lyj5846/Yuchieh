from __future__ import annotations

import csv
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from run_main_model_training_pipeline import (
    CONFIG_PATH,
    PROJECT_ROOT,
    load_json,
    normalize_market,
    normalize_stock,
    normalize_stock_id,
    normalize_theme,
    read_csv,
    split_name,
    validate_inputs,
)


VALIDATION_DIR = PROJECT_ROOT / "validation_layer"
DECISION_DIR = PROJECT_ROOT / "decision_layer"
SCORES_PATH = PROJECT_ROOT / "model_layer" / "main_model_scores.csv"
MAIN_DECISION_PATH = DECISION_DIR / "main_model_decision.json"

REVIEW_MD_PATH = VALIDATION_DIR / "repeat_signal_episode_review.md"
SUMMARY_PATH = VALIDATION_DIR / "repeat_signal_episode_summary.csv"
EVENTS_PATH = VALIDATION_DIR / "repeat_signal_episode_events.csv"
DECISION_JSON_PATH = DECISION_DIR / "repeat_signal_episode_decision.json"

LOOKAHEAD_DAYS = 10
RAW_TOP_LIMIT = 10
MIN_ALL_HOLDOUT_EVENTS = 50
MIN_RESET_HOLDOUT_EVENTS = 20


def fail(message: str) -> None:
    raise SystemExit(f"FAIL: {message}")


def fmt_pct(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.2%}"


def safe_float(value: object) -> float:
    try:
        if value is None or pd.isna(value):
            return math.nan
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def first_hit_day(frame: pd.DataFrame, threshold: float, direction: str) -> pd.Series:
    if direction == "ge":
        hits = frame.ge(threshold)
    elif direction == "le":
        hits = frame.le(threshold)
    else:
        fail(f"unsupported hit direction: {direction}")
    hit_matrix = hits.to_numpy(dtype=bool)
    first = np.full(hit_matrix.shape[0], np.nan, dtype=float)
    has_hit = hit_matrix.any(axis=1)
    first[has_hit] = np.argmax(hit_matrix[has_hit], axis=1) + 1
    return pd.Series(first, index=frame.index)


def date_at_horizon(row: pd.Series, day_value: object) -> str:
    if pd.isna(day_value):
        return ""
    day = int(day_value)
    value = row.get(f"future_date_{day}")
    if pd.isna(value):
        return ""
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def load_base_frame(config: dict) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    paths = validate_inputs(config)
    stock = normalize_stock(read_csv(paths["stock_daily_all"]))
    stock["股票代號"] = normalize_stock_id(stock["股票代號"])
    market = normalize_market(read_csv(paths["market_daily"]))
    theme = normalize_theme(read_csv(paths["theme_group"]))
    theme["股票代號"] = normalize_stock_id(theme["股票代號"])

    if stock["日期"].max() != market["日期"].max():
        fail(
            "stock and market latest dates differ: "
            f"{stock['日期'].max().date()} vs {market['日期'].max().date()}"
        )

    base = stock.sort_values(["股票代號", "日期"]).reset_index(drop=True).copy()
    group = base.groupby("股票代號", sort=False)
    base["stock_trading_index"] = group.cumcount()
    base["buy_open_next"] = group["開盤價"].shift(-1)
    base["buy_date_next"] = group["日期"].shift(-1)
    for horizon in range(1, LOOKAHEAD_DAYS + 1):
        base[f"future_close_{horizon}"] = group["收盤價"].shift(-horizon)
        base[f"future_low_{horizon}"] = group["最低價"].shift(-horizon)
        base[f"future_date_{horizon}"] = group["日期"].shift(-horizon)

    close_cols = [f"future_close_{i}" for i in range(1, LOOKAHEAD_DAYS + 1)]
    low_cols = [f"future_low_{i}" for i in range(1, LOOKAHEAD_DAYS + 1)]
    date_cols = [f"future_date_{i}" for i in range(1, LOOKAHEAD_DAYS + 1)]
    base["label_complete"] = (
        base[close_cols].notna().all(axis=1)
        & base[low_cols].notna().all(axis=1)
        & base[date_cols].notna().all(axis=1)
        & base["buy_open_next"].notna()
        & (base["buy_open_next"] > 0)
    )

    close_returns = base[close_cols].div(base["buy_open_next"], axis=0) - 1.0
    low_returns = base[low_cols].div(base["buy_open_next"], axis=0) - 1.0
    close_returns = close_returns.replace([np.inf, -np.inf], np.nan)
    low_returns = low_returns.replace([np.inf, -np.inf], np.nan)
    base["future_10d_high_close_return"] = close_returns.max(axis=1)
    base["future_10d_min_low_return"] = low_returns.min(axis=1)
    profit_day = first_hit_day(close_returns, 0.03, "ge")
    drawdown_day = first_hit_day(low_returns, -0.03, "le")
    base["target_success"] = (base["label_complete"] & profit_day.notna()).astype(int)
    base["hit_plus3_day"] = profit_day
    base["hit_minus3_day"] = drawdown_day
    base["hit_plus3_date"] = [date_at_horizon(row, profit_day.loc[idx]) for idx, row in base.iterrows()]
    base["hit_minus3_date"] = [date_at_horizon(row, drawdown_day.loc[idx]) for idx, row in base.iterrows()]
    base["split"] = split_name(base["日期"], base["label_complete"], config)

    daily = (
        base[base["label_complete"]]
        .groupby("日期")
        .agg(
            daily_market_success_rate=("target_success", "mean"),
            daily_market_avg_return=("future_10d_high_close_return", "mean"),
        )
        .reset_index()
    )
    base = base.merge(daily, on="日期", how="left")
    base = base.merge(theme[["股票代號", "股票名稱", "主分類"]], on="股票代號", how="left")
    base["股票名稱"] = base["股票名稱"].fillna("")
    base["主分類"] = base["主分類"].fillna("未分類")
    latest_date = stock["日期"].max().strftime("%Y-%m-%d")
    return base, theme, latest_date


def load_scores(base: pd.DataFrame) -> pd.DataFrame:
    if not SCORES_PATH.exists():
        fail("missing model scores")
    scores = pd.read_csv(SCORES_PATH, encoding="utf-8-sig")
    required = {"日期", "股票代號", "股票名稱", "integrated_research_score", "daily_rank"}
    missing = required - set(scores.columns)
    if missing:
        fail("model scores missing columns: " + ", ".join(sorted(missing)))
    scores = scores.copy()
    scores["日期"] = pd.to_datetime(scores["日期"])
    scores["股票代號"] = normalize_stock_id(scores["股票代號"])
    scores["integrated_research_score"] = pd.to_numeric(scores["integrated_research_score"], errors="coerce")
    scores["daily_rank"] = pd.to_numeric(scores["daily_rank"], errors="coerce")
    index_cols = ["日期", "股票代號", "stock_trading_index"]
    scores = scores.merge(base[index_cols], on=["日期", "股票代號"], how="left")
    scores = scores.dropna(subset=["integrated_research_score", "stock_trading_index"]).copy()
    scores["stock_trading_index"] = scores["stock_trading_index"].astype(int)
    scores = scores.sort_values(["日期", "integrated_research_score", "股票代號"], ascending=[True, False, True])
    scores["raw_top_rank"] = scores.groupby("日期").cumcount() + 1
    scores["raw_top10"] = scores["raw_top_rank"] <= RAW_TOP_LIMIT
    return scores


def selected_gate() -> float | None:
    if not MAIN_DECISION_PATH.exists():
        return None
    decision = json.loads(MAIN_DECISION_PATH.read_text(encoding="utf-8"))
    value = decision.get("selected_gate")
    parsed = safe_float(value)
    return None if pd.isna(parsed) else parsed


def simulate_formal_signals(scores: pd.DataFrame, gate: float | None) -> pd.DataFrame:
    selected_indices: list[int] = []
    last_pick_index: dict[str, int] = {}
    for signal_date, day_rows in scores.groupby("日期", sort=True):
        day_rows = day_rows.sort_values(["integrated_research_score", "股票代號"], ascending=[False, True])
        if day_rows.empty:
            continue
        if gate is not None and float(day_rows.iloc[0]["integrated_research_score"]) < gate:
            continue
        selected_today = 0
        for idx, row in day_rows.iterrows():
            stock_id = str(row["股票代號"])
            stock_index = int(row["stock_trading_index"])
            if stock_id in last_pick_index and stock_index - last_pick_index[stock_id] <= LOOKAHEAD_DAYS:
                continue
            last_pick_index[stock_id] = stock_index
            selected_indices.append(idx)
            selected_today += 1
            if selected_today >= 3:
                break
    if not selected_indices:
        return pd.DataFrame(columns=list(scores.columns) + ["formal_pick_rank"])
    formal = scores.loc[selected_indices].copy()
    formal = formal.sort_values(["日期", "integrated_research_score"], ascending=[True, False])
    formal["formal_pick_rank"] = formal.groupby("日期").cumcount() + 1
    return formal.reset_index(drop=True)


def base_lookup(base: pd.DataFrame) -> dict[tuple[str, pd.Timestamp], pd.Series]:
    return {
        (str(row["股票代號"]), pd.Timestamp(row["日期"])): row
        for _, row in base.iterrows()
    }


def prior_status_as_of(prior: pd.Series, event_date: pd.Timestamp) -> tuple[str, str]:
    buy_open = safe_float(prior.get("buy_open_next"))
    if pd.isna(buy_open) or buy_open <= 0:
        return "not_started", ""
    observed = 0
    for horizon in range(1, LOOKAHEAD_DAYS + 1):
        future_date = prior.get(f"future_date_{horizon}")
        close = safe_float(prior.get(f"future_close_{horizon}"))
        if pd.isna(future_date) or pd.Timestamp(future_date) > event_date or pd.isna(close):
            continue
        observed += 1
        if close >= buy_open * 1.03:
            return "success", pd.Timestamp(future_date).strftime("%Y-%m-%d")
    if observed >= LOOKAHEAD_DAYS:
        return "failure", ""
    return "tracking", ""


def build_repeat_events(base: pd.DataFrame, scores: pd.DataFrame, formal: pd.DataFrame) -> pd.DataFrame:
    lookup = base_lookup(base)
    formal_by_stock = {
        stock_id: group.sort_values("stock_trading_index").reset_index(drop=True)
        for stock_id, group in formal.groupby("股票代號", sort=False)
    }
    scores_by_stock = {
        stock_id: group.sort_values("stock_trading_index").reset_index(drop=True)
        for stock_id, group in scores.groupby("股票代號", sort=False)
    }
    events: list[dict] = []
    raw_top = scores[scores["raw_top10"]].sort_values(["股票代號", "日期", "raw_top_rank"])
    for _, event in raw_top.iterrows():
        stock_id = str(event["股票代號"])
        event_date = pd.Timestamp(event["日期"])
        event_index = int(event["stock_trading_index"])
        formals = formal_by_stock.get(stock_id)
        if formals is None or formals.empty:
            continue
        prior_formals = formals[formals["stock_trading_index"] < event_index]
        if prior_formals.empty:
            continue
        prior = prior_formals.iloc[-1]
        prior_date = pd.Timestamp(prior["日期"])
        prior_index = int(prior["stock_trading_index"])
        distance = event_index - prior_index
        score_history = scores_by_stock.get(stock_id, pd.DataFrame())
        between = score_history[
            (score_history["stock_trading_index"] > prior_index)
            & (score_history["stock_trading_index"] < event_index)
        ]
        left_raw_top10 = bool((~between["raw_top10"]).any()) if not between.empty else False

        prior_base = lookup.get((stock_id, prior_date))
        event_base = lookup.get((stock_id, event_date))
        if prior_base is None or event_base is None:
            continue
        prior_status, prior_success_date = prior_status_as_of(prior_base, event_date)
        event_complete = bool(event_base["label_complete"])
        event_high_return = safe_float(event_base.get("future_10d_high_close_return"))
        event_low_return = safe_float(event_base.get("future_10d_min_low_return"))
        market_success = safe_float(event_base.get("daily_market_success_rate"))
        market_return = safe_float(event_base.get("daily_market_avg_return"))
        beat_market = (
            event_complete
            and not pd.isna(event_high_return)
            and not pd.isna(market_return)
            and event_high_return >= market_return
        )

        within_10 = distance <= LOOKAHEAD_DAYS
        within_10_after_success = within_10 and prior_status == "success"
        within_10_not_success = within_10 and prior_status != "success"
        after_10_reappeared = distance > LOOKAHEAD_DAYS
        if within_10_after_success:
            primary_context = "within_10_after_success"
        elif within_10_not_success:
            primary_context = "within_10_not_success"
        elif left_raw_top10:
            primary_context = "returned_after_leaving_top10"
        else:
            primary_context = "after_10_reappeared"

        events.append(
            {
                "event_date": event_date.strftime("%Y-%m-%d"),
                "stock_id": stock_id,
                "stock_name": event.get("股票名稱", ""),
                "main_category": event_base.get("主分類", "未分類"),
                "split": event_base.get("split", "tracking"),
                "month": event_date.strftime("%Y-%m"),
                "raw_top_rank": int(event["raw_top_rank"]),
                "research_score": float(event["integrated_research_score"]),
                "prior_signal_date": prior_date.strftime("%Y-%m-%d"),
                "prior_formal_rank": int(prior["formal_pick_rank"]),
                "days_since_prior_signal": int(distance),
                "prior_status_as_of_event": prior_status,
                "prior_success_date_as_of_event": prior_success_date,
                "left_raw_top10_before_return": int(left_raw_top10),
                "within_10_not_success": int(within_10_not_success),
                "within_10_after_success": int(within_10_after_success),
                "after_10_reappeared": int(after_10_reappeared),
                "returned_after_leaving_top10": int(left_raw_top10),
                "primary_context": primary_context,
                "event_buy_date": pd.Timestamp(event_base["buy_date_next"]).strftime("%Y-%m-%d")
                if pd.notna(event_base["buy_date_next"])
                else "",
                "event_label_status": "completed" if event_complete else "tracking",
                "event_target_success": int(event_base["target_success"]) if event_complete else "",
                "event_hit_plus3_date": event_base.get("hit_plus3_date", "") if event_complete else "",
                "event_hit_minus3_date": event_base.get("hit_minus3_date", "") if event_complete else "",
                "event_future_10d_high_close_return": event_high_return if event_complete else math.nan,
                "event_future_10d_min_low_return": event_low_return if event_complete else math.nan,
                "daily_market_success_rate": market_success if event_complete else math.nan,
                "daily_market_avg_return": market_return if event_complete else math.nan,
                "event_beat_market": int(beat_market) if event_complete else "",
            }
        )
    return pd.DataFrame(events)


def summarize(events: pd.DataFrame) -> pd.DataFrame:
    scenarios = {
        "all_repeat_events": pd.Series(True, index=events.index),
        "within_10_not_success": events["within_10_not_success"].eq(1),
        "within_10_after_success": events["within_10_after_success"].eq(1),
        "after_10_reappeared": events["after_10_reappeared"].eq(1),
        "returned_after_leaving_top10": events["returned_after_leaving_top10"].eq(1),
    }
    rows: list[dict] = []
    for scenario, mask in scenarios.items():
        scenario_events = events[mask].copy()
        for split in ["overall", "train", "development", "holdout"]:
            part = scenario_events if split == "overall" else scenario_events[scenario_events["split"] == split]
            completed = part[part["event_label_status"] == "completed"].copy()
            tracking = part[part["event_label_status"] != "completed"]
            if completed.empty:
                success_rate = math.nan
                avg_return = math.nan
                avg_drawdown = math.nan
                market_success = math.nan
                market_return = math.nan
                success_lift = math.nan
                return_lift = math.nan
                beat_market_rate = math.nan
                max_stock_share = math.nan
                max_industry_share = math.nan
                max_month_share = math.nan
            else:
                success_rate = float(pd.to_numeric(completed["event_target_success"], errors="coerce").mean())
                avg_return = float(completed["event_future_10d_high_close_return"].mean())
                avg_drawdown = float(completed["event_future_10d_min_low_return"].mean())
                market_success = float(completed["daily_market_success_rate"].mean())
                market_return = float(completed["daily_market_avg_return"].mean())
                success_lift = success_rate - market_success
                return_lift = avg_return - market_return
                beat_market_rate = float(pd.to_numeric(completed["event_beat_market"], errors="coerce").mean())
                max_stock_share = float(completed["stock_id"].value_counts(normalize=True).max())
                max_industry_share = float(completed["main_category"].value_counts(normalize=True).max())
                max_month_share = float(completed["month"].value_counts(normalize=True).max())
            rows.append(
                {
                    "scenario": scenario,
                    "split": split,
                    "events": int(len(part)),
                    "completed_events": int(len(completed)),
                    "tracking_events": int(len(tracking)),
                    "stocks": int(completed["stock_id"].nunique()) if not completed.empty else 0,
                    "months": int(completed["month"].nunique()) if not completed.empty else 0,
                    "success_rate": success_rate,
                    "avg_10d_high_close_return": avg_return,
                    "avg_10d_min_low_return": avg_drawdown,
                    "daily_market_success_rate": market_success,
                    "daily_market_avg_return": market_return,
                    "success_lift": success_lift,
                    "return_lift": return_lift,
                    "beat_market_rate": beat_market_rate,
                    "max_stock_share": max_stock_share,
                    "max_industry_share": max_industry_share,
                    "max_month_share": max_month_share,
                }
            )
    return pd.DataFrame(rows)


def metric(summary: pd.DataFrame, scenario: str, split: str) -> dict:
    row = summary[(summary["scenario"] == scenario) & (summary["split"] == split)]
    if row.empty:
        return {}
    return row.iloc[0].to_dict()


def decide(summary: pd.DataFrame) -> dict:
    all_holdout = metric(summary, "all_repeat_events", "holdout")
    reset_holdout = metric(summary, "returned_after_leaving_top10", "holdout")
    if not all_holdout or int(all_holdout.get("completed_events", 0)) < MIN_ALL_HOLDOUT_EVENTS:
        status = "insufficient_signal"
        next_step = "collect_more_completed_repeat_events"
        reason = (
            "Holdout completed repeat-signal samples are below the minimum needed for a re-entry rule review."
        )
    elif (
        reset_holdout
        and int(reset_holdout.get("completed_events", 0)) >= MIN_RESET_HOLDOUT_EVENTS
        and safe_float(reset_holdout.get("success_lift")) >= 0.05
        and safe_float(reset_holdout.get("return_lift")) > 0
        and safe_float(reset_holdout.get("beat_market_rate")) >= 0.50
        and safe_float(reset_holdout.get("max_stock_share")) <= 0.20
        and safe_float(reset_holdout.get("max_industry_share")) <= 0.50
    ):
        status = "allow_reentry_after_reset"
        next_step = "plan_reentry_label_contract"
        reason = (
            "Returned-after-leaving-Top10 repeats passed the holdout lift, return, and concentration checks."
        )
    else:
        status = "keep_tracking_only"
        next_step = "keep_formal_tracking_only"
        reason = (
            "Repeat high-score events do not yet justify opening a new formal buy point; keep them as tracking evidence."
        )
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": status,
        "recommended_next_step": next_step,
        "reason": reason,
        "minimum_all_holdout_events": MIN_ALL_HOLDOUT_EVENTS,
        "minimum_reset_holdout_events": MIN_RESET_HOLDOUT_EVENTS,
        "all_repeat_holdout_completed_events": int(all_holdout.get("completed_events", 0)) if all_holdout else 0,
        "reset_holdout_completed_events": int(reset_holdout.get("completed_events", 0)) if reset_holdout else 0,
        "reset_holdout_success_lift": safe_float(reset_holdout.get("success_lift")) if reset_holdout else None,
        "reset_holdout_return_lift": safe_float(reset_holdout.get("return_lift")) if reset_holdout else None,
    }


def write_review(summary: pd.DataFrame, decision: dict, latest_date: str, gate: float | None) -> None:
    holdout_rows = summary[summary["split"] == "holdout"].copy()
    lines = [
        "# Repeat Signal Episode Review",
        "",
        f"- Generated: {decision['generated_at']}",
        f"- Data latest date: {latest_date}",
        f"- Selected score gate: {gate if gate is not None else 'none'}",
        "- Review type: label-only repeat signal episode review.",
        "- Formal output: unchanged by this review.",
        "- This does not choose stocks.",
        "- This does not train or promote a model.",
        "- research_score is a ranking score, not a probability.",
        "- 10 trading days is a validation window, not a wave boundary.",
        "",
        "## Decision",
        "",
        f"- Status: `{decision['status']}`",
        f"- Recommended next step: `{decision['recommended_next_step']}`",
        f"- Reason: {decision['reason']}",
        "",
        "## Holdout Scenario Summary",
        "",
        "| 情境 | 已結案筆數 | 成功率 | 成功率差 | 平均最高收盤報酬 | 報酬差 | 贏同日市場率 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in holdout_rows.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["scenario"]),
                    str(int(row["completed_events"])),
                    fmt_pct(row["success_rate"]),
                    fmt_pct(row["success_lift"]),
                    fmt_pct(row["avg_10d_high_close_return"]),
                    fmt_pct(row["return_lift"]),
                    fmt_pct(row["beat_market_rate"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `within_10_after_success` means the previous formal signal had already hit +3% before the repeat high-score day.",
            "- `within_10_not_success` means the previous formal signal was still not successful when the repeat high-score day appeared.",
            "- `after_10_reappeared` checks whether day-count alone is enough evidence.",
            "- `returned_after_leaving_top10` checks whether leaving raw Top10 and coming back is a better reset signal.",
            "",
        ]
    )
    REVIEW_MD_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    config = load_json(CONFIG_PATH)
    base, _theme, latest_date = load_base_frame(config)
    scores = load_scores(base)
    gate = selected_gate()
    formal = simulate_formal_signals(scores, gate)
    events = build_repeat_events(base, scores, formal)
    if events.empty:
        fail("repeat signal review produced no events")
    summary = summarize(events)
    decision = decide(summary)

    events.to_csv(EVENTS_PATH, index=False, encoding="utf-8-sig")
    summary.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")
    DECISION_JSON_PATH.write_text(json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_review(summary, decision, latest_date, gate)
    print("OK: repeat signal episode review completed")
    print(f"STATUS: {decision['status']}")
    print(f"NEXT_STEP: {decision['recommended_next_step']}")
    print(f"REVIEW: {REVIEW_MD_PATH}")


if __name__ == "__main__":
    main()
