from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "project_config.json"
TARGET_SENSITIVITY_DECISION_PATH = PROJECT_ROOT / "decision_layer" / "target_sensitivity_decision.json"

VALIDATION_DIR = PROJECT_ROOT / "validation_layer"
DECISION_DIR = PROJECT_ROOT / "decision_layer"

REVIEW_MD_PATH = VALIDATION_DIR / "data_enrichment_review.md"
GAP_MATRIX_PATH = VALIDATION_DIR / "data_enrichment_gap_matrix.csv"
INVENTORY_PATH = VALIDATION_DIR / "data_enrichment_current_inventory.csv"
DECISION_JSON_PATH = DECISION_DIR / "data_enrichment_decision.json"


def fail(message: str) -> None:
    raise SystemExit(f"FAIL: {message}")


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_inputs(config: dict) -> dict[str, Path]:
    allowed = config.get("allowed_inputs", {})
    if set(allowed) != {"stock_daily_all", "market_daily", "theme_group"}:
        fail("allowed_inputs must contain exactly stock_daily_all, market_daily, theme_group")
    old_marker = "stock" + "_raw_only" + "_project"
    paths = {name: Path(value) for name, value in allowed.items()}
    for name, path in paths.items():
        if not path.exists():
            fail(f"missing allowed input {name}: {path}")
        if old_marker in str(path):
            fail(f"old project path is not allowed: {name}")
    return paths


def read_columns(paths: dict[str, Path]) -> dict[str, list[str]]:
    columns: dict[str, list[str]] = {}
    for name, path in paths.items():
        frame = pd.read_csv(path, encoding="utf-8-sig", nrows=3)
        columns[name] = [str(col) for col in frame.columns]
    return columns


def inventory_rows(columns: dict[str, list[str]]) -> list[dict]:
    groups = {
        "price_volume": {
            "source": "stock_daily_all",
            "required": ["開盤價", "最高價", "最低價", "收盤價", "成交量(張)"],
        },
        "institutional_daily": {
            "source": "stock_daily_all",
            "required": ["外資買賣超(張)", "投信買賣超(張)", "自營商買賣超(張)", "三大法人合計買賣超(張)"],
        },
        "margin_short_daily": {
            "source": "stock_daily_all",
            "required": ["融資買進", "融資賣出", "融資增減", "融資餘額", "融券買進", "融券賣出", "融券增減", "融券餘額"],
        },
        "day_trade_daily": {
            "source": "stock_daily_all",
            "required": ["當沖成交量(張)", "當沖比率"],
        },
        "market_state_daily": {
            "source": "market_daily",
            "required": ["加權指數收盤", "電子指數收盤", "櫃買指數收盤", "上漲家數", "下跌家數", "漲停家數", "跌停家數"],
        },
        "static_theme": {
            "source": "theme_group",
            "required": ["股票代號", "股票名稱", "主分類", "子分類"],
        },
    }
    rows: list[dict] = []
    for group_name, spec in groups.items():
        source = str(spec["source"])
        available = set(columns.get(source, []))
        required = list(spec["required"])
        present = [col for col in required if col in available]
        missing = [col for col in required if col not in available]
        rows.append(
            {
                "current_data_group": group_name,
                "source": source,
                "required_columns": " | ".join(required),
                "present_columns": " | ".join(present),
                "missing_columns": " | ".join(missing),
                "coverage_rate": len(present) / len(required) if required else 0.0,
            }
        )
    return rows


def gap_candidates(columns: dict[str, list[str]]) -> list[dict]:
    stock_cols = set(columns.get("stock_daily_all", []))
    market_cols = set(columns.get("market_daily", []))
    theme_cols = set(columns.get("theme_group", []))
    current_summary = {
        "price_volume": " | ".join([col for col in ["開盤價", "最高價", "最低價", "收盤價", "成交量(張)"] if col in stock_cols]),
        "chip": " | ".join([col for col in ["外資買賣超(張)", "投信買賣超(張)", "融資餘額", "融券餘額", "當沖比率"] if col in stock_cols]),
        "market": " | ".join([col for col in ["加權指數收盤", "電子指數收盤", "櫃買指數收盤", "上漲家數", "下跌家數"] if col in market_cols]),
        "theme": " | ".join([col for col in ["主分類", "子分類"] if col in theme_cols]),
    }
    raw_rows = [
        {
            "data_need_id": "event_risk_calendar",
            "data_family": "event_and_warning_flags",
            "risk_question_answered": "訊號日前後是否有重大訊息、停復牌、注意或處置風險，導致先跌 -3%。",
            "current_coverage": "missing",
            "current_columns_used": "",
            "missing_fields": "重大訊息日期 | 停復牌日期 | 注意股 | 處置股 | 法說會 | 除權息與重大公司事件",
            "why_it_may_help": "目前價格與籌碼只能看到結果，缺少事件風險觸發原因；這類欄位最直接對應先跌風險。",
            "no_leakage_rule": "只允許使用訊號日收盤前已公告或當日已知的事件旗標；公告日晚於訊號日者不得回填。",
            "fetch_feasibility": "high",
            "implementation_cost": "medium",
            "expected_frequency": "daily",
            "direct_adverse_relevance": 5,
            "fetchability_score": 4,
            "timeliness_score": 5,
            "current_missing_score": 5,
        },
        {
            "data_need_id": "monthly_revenue_surprise",
            "data_family": "fundamental_momentum",
            "risk_question_answered": "個股基本面是否在訊號日前已出現營收轉弱或低於預期，導致漲不上去先回撤。",
            "current_coverage": "missing",
            "current_columns_used": "",
            "missing_fields": "月營收 | YoY | MoM | 累計營收 | 公告日期 | 產業同群相對營收",
            "why_it_may_help": "目前資料沒有營運變化，只能看價格與籌碼；營收能補上公司基本面方向。",
            "no_leakage_rule": "用公告日期控管，訊號日以前已公告的最近一期營收才可進特徵。",
            "fetch_feasibility": "high",
            "implementation_cost": "medium",
            "expected_frequency": "monthly",
            "direct_adverse_relevance": 4,
            "fetchability_score": 4,
            "timeliness_score": 3,
            "current_missing_score": 5,
        },
        {
            "data_need_id": "external_market_regime",
            "data_family": "global_and_sector_market",
            "risk_question_answered": "台股訊號日隔天是否受到美股、半導體、匯率或期貨風險影響而先跌。",
            "current_coverage": "partial",
            "current_columns_used": current_summary["market"],
            "missing_fields": "NASDAQ | SOX | S&P futures | USD/TWD | 台指期夜盤 | AI/server sector overseas proxy",
            "why_it_may_help": "目前只有台股當日大盤，缺隔夜外部風險；這會影響隔日開盤後是否先跌。",
            "no_leakage_rule": "只使用訊號日台灣收盤後到隔日開盤前已發生且可取得的海外市場資料，並另標記時間窗。",
            "fetch_feasibility": "medium",
            "implementation_cost": "medium",
            "expected_frequency": "daily",
            "direct_adverse_relevance": 4,
            "fetchability_score": 3,
            "timeliness_score": 5,
            "current_missing_score": 4,
        },
        {
            "data_need_id": "securities_lending_and_short_pressure",
            "data_family": "advanced_chip_pressure",
            "risk_question_answered": "是否有借券、融券或空方壓力增加，導致價格先向下測風險線。",
            "current_coverage": "partial",
            "current_columns_used": current_summary["chip"],
            "missing_fields": "借券賣出 | 借券餘額 | 券資比 | 融資使用率 | 當沖買賣分解",
            "why_it_may_help": "目前有融資券餘額，但沒有更細的空方與借券壓力。",
            "no_leakage_rule": "只使用訊號日已公布的餘額與交易資料，不能使用後續補公布資料回填。",
            "fetch_feasibility": "medium",
            "implementation_cost": "medium",
            "expected_frequency": "daily",
            "direct_adverse_relevance": 4,
            "fetchability_score": 3,
            "timeliness_score": 4,
            "current_missing_score": 3,
        },
        {
            "data_need_id": "intraday_closing_pressure",
            "data_family": "intraday_microstructure",
            "risk_question_answered": "訊號日尾盤是否已出現拉高出貨、收盤失真或隔日容易回落的盤中型態。",
            "current_coverage": "partial",
            "current_columns_used": current_summary["price_volume"],
            "missing_fields": "分鐘K | 尾盤成交 | 收盤前30分鐘漲跌 | VWAP | 開高走低型態 | 收盤位置",
            "why_it_may_help": "日線高低收量太粗，無法知道訊號日內部買盤是否乾淨。",
            "no_leakage_rule": "只使用訊號日收盤前的分時資料；不得使用隔日盤中資料。",
            "fetch_feasibility": "low",
            "implementation_cost": "high",
            "expected_frequency": "intraday",
            "direct_adverse_relevance": 5,
            "fetchability_score": 1,
            "timeliness_score": 5,
            "current_missing_score": 4,
        },
        {
            "data_need_id": "quarterly_fundamental_quality",
            "data_family": "financial_statement_quality",
            "risk_question_answered": "個股是否只是題材上漲但財報品質不足，導致追價後先回撤。",
            "current_coverage": "missing",
            "current_columns_used": "",
            "missing_fields": "EPS | 毛利率 | 營益率 | ROE | 現金流 | 存貨 | 財報公告日期",
            "why_it_may_help": "能補上公司品質，但更新頻率低，對短線先跌 -3% 的即時性較弱。",
            "no_leakage_rule": "以財報公告日為準，訊號日以前已公告資料才可使用。",
            "fetch_feasibility": "medium",
            "implementation_cost": "high",
            "expected_frequency": "quarterly",
            "direct_adverse_relevance": 2,
            "fetchability_score": 3,
            "timeliness_score": 2,
            "current_missing_score": 5,
        },
    ]
    for row in raw_rows:
        row["review_score"] = (
            row["direct_adverse_relevance"] * 0.40
            + row["fetchability_score"] * 0.25
            + row["timeliness_score"] * 0.25
            + row["current_missing_score"] * 0.10
        )
    return sorted(raw_rows, key=lambda item: item["review_score"], reverse=True)


def decide(gaps: list[dict], target_decision: dict) -> dict:
    recommended = gaps[0]
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "data_gap_confirmed",
        "recommended_next_step": "build_event_risk_data_collector_spec",
        "recommended_data_need_id": recommended["data_need_id"],
        "recommended_data_family": recommended["data_family"],
        "reason": "目前交易目標本身可用，但模型學不到先跌風險；最直接缺口是事件與注意處置類風險旗標。",
        "target_review_status": target_decision.get("status"),
        "target_review_next_step": target_decision.get("recommended_next_step"),
        "do_not_retrain_yet": True,
        "formal_outputs_unchanged": True,
        "top_gap_ids": [row["data_need_id"] for row in gaps[:3]],
    }


def write_review_md(decision: dict, gaps: list[dict], inventory: list[dict]) -> None:
    recommended = gaps[0]
    lines = [
        "# Data Enrichment Review",
        "",
        f"- Generated: {decision['generated_at']}",
        "- Scope: data gap review only; no model training; no stock candidates.",
        "- Formal output: unchanged by this review.",
        "",
        "## 白話結論",
        "",
        decision["reason"],
        "",
        f"- Review status: `{decision['status']}`",
        f"- Recommended next step: `{decision['recommended_next_step']}`",
        f"- Recommended data need: `{decision['recommended_data_need_id']}`",
        "",
        "## Why This Comes First",
        "",
        f"- Missing fields: {recommended['missing_fields']}",
        f"- Why it may help: {recommended['why_it_may_help']}",
        f"- No-leakage rule: {recommended['no_leakage_rule']}",
        f"- Fetch feasibility: {recommended['fetch_feasibility']}",
        "",
        "## Current Data Coverage",
        "",
        "| group | source | coverage | missing |",
        "|---|---|---:|---|",
    ]
    for row in inventory:
        lines.append(
            f"| {row['current_data_group']} | {row['source']} | {row['coverage_rate']:.0%} | {row['missing_columns'] or 'none'} |"
        )
    lines.extend(
        [
            "",
            "## Gap Ranking",
            "",
            "| data need | family | current coverage | review score |",
            "|---|---|---|---:|",
        ]
    )
    for row in gaps:
        lines.append(
            f"| {row['data_need_id']} | {row['data_family']} | {row['current_coverage']} | {row['review_score']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- This does not add a data source yet.",
            "- This does not train or promote a model.",
            "- This does not choose stocks.",
            "- Any future collector must use announcement dates or observable timestamps to avoid leakage.",
            "",
            "## Outputs",
            "",
            f"- `{GAP_MATRIX_PATH.relative_to(PROJECT_ROOT)}`",
            f"- `{INVENTORY_PATH.relative_to(PROJECT_ROOT)}`",
            f"- `{DECISION_JSON_PATH.relative_to(PROJECT_ROOT)}`",
            "",
        ]
    )
    REVIEW_MD_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    config = load_json(CONFIG_PATH)
    paths = validate_inputs(config)
    if not TARGET_SENSITIVITY_DECISION_PATH.exists():
        fail("target sensitivity decision is required before data enrichment review")
    target_decision = load_json(TARGET_SENSITIVITY_DECISION_PATH)
    if target_decision.get("recommended_next_step") != "review_data_enrichment":
        fail("target sensitivity review must recommend data enrichment before this step")
    columns = read_columns(paths)
    inventory = inventory_rows(columns)
    gaps = gap_candidates(columns)
    decision = decide(gaps, target_decision)

    pd.DataFrame(inventory).to_csv(INVENTORY_PATH, index=False, encoding="utf-8-sig")
    pd.DataFrame(gaps).to_csv(GAP_MATRIX_PATH, index=False, encoding="utf-8-sig")
    DECISION_JSON_PATH.write_text(json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_review_md(decision, gaps, inventory)

    print("OK: data enrichment review completed")
    print(f"STATUS: {decision['status']}")
    print(f"NEXT_STEP: {decision['recommended_next_step']}")
    print(f"REVIEW: {REVIEW_MD_PATH}")


if __name__ == "__main__":
    main()
