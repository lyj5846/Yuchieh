# Data Enrichment Review

- Generated: 2026-07-04 10:08:35
- Scope: data gap review only; no model training; no stock candidates.
- Formal output: unchanged by this review.

## 白話結論

目前交易目標本身可用，但模型學不到先跌風險；最直接缺口是事件與注意處置類風險旗標。

- Review status: `data_gap_confirmed`
- Recommended next step: `build_event_risk_data_collector_spec`
- Recommended data need: `event_risk_calendar`

## Why This Comes First

- Missing fields: 重大訊息日期 | 停復牌日期 | 注意股 | 處置股 | 法說會 | 除權息與重大公司事件
- Why it may help: 目前價格與籌碼只能看到結果，缺少事件風險觸發原因；這類欄位最直接對應先跌風險。
- No-leakage rule: 只允許使用訊號日收盤前已公告或當日已知的事件旗標；公告日晚於訊號日者不得回填。
- Fetch feasibility: high

## Current Data Coverage

| group | source | coverage | missing |
|---|---|---:|---|
| price_volume | stock_daily_all | 100% | none |
| institutional_daily | stock_daily_all | 100% | none |
| margin_short_daily | stock_daily_all | 100% | none |
| day_trade_daily | stock_daily_all | 100% | none |
| market_state_daily | market_daily | 100% | none |
| static_theme | theme_group | 100% | none |

## Gap Ranking

| data need | family | current coverage | review score |
|---|---|---|---:|
| event_risk_calendar | event_and_warning_flags | missing | 4.75 |
| external_market_regime | global_and_sector_market | partial | 4.00 |
| intraday_closing_pressure | intraday_microstructure | partial | 3.90 |
| monthly_revenue_surprise | fundamental_momentum | missing | 3.85 |
| securities_lending_and_short_pressure | advanced_chip_pressure | partial | 3.65 |
| quarterly_fundamental_quality | financial_statement_quality | missing | 2.55 |

## Boundary

- This does not add a data source yet.
- This does not train or promote a model.
- This does not choose stocks.
- Any future collector must use announcement dates or observable timestamps to avoid leakage.

## Outputs

- `validation_layer\data_enrichment_gap_matrix.csv`
- `validation_layer\data_enrichment_current_inventory.csv`
- `decision_layer\data_enrichment_decision.json`
