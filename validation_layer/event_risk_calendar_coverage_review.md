# Event Risk Calendar Coverage Review

- Scope: coverage review only.
- Formal output: unchanged.
- Model training: not executed.
- Event input source: produced but not enabled in `project_config.json`.

## 白話結論

目前事件資料只是官方 API 快照，主要落在最近日期，train/development 幾乎沒有覆蓋；不能拿去重訓。

- Status: `coverage_not_training_ready`
- Recommended next step: `build_historical_event_risk_backfill_collector`
- Usable rows by market latest date: 90
- Event date range: 2026-06-16 to 2026-07-06

## Split Coverage

| split | trading days | event rows | event days | coverage | stocks |
|---|---:|---:|---:|---:|---:|
| train | 485 | 0 | 0 | 0.00% | 0 |
| development | 55 | 0 | 0 | 0.00% | 0 |
| holdout | 65 | 90 | 14 | 21.54% | 74 |

## Event Type Coverage

| event type | rows | stocks | first date | last date |
|---|---:|---:|---|---|
| attention | 11 | 9 | 2026-07-01 | 2026-07-02 |
| disposition | 20 | 20 | 2026-06-16 | 2026-07-01 |
| ex_dividend | 34 | 34 | 2026-06-23 | 2026-07-06 |
| investor_meeting | 3 | 3 | 2026-07-02 | 2026-07-02 |
| material_info | 22 | 18 | 2026-07-01 | 2026-07-02 |

## Decision Boundary

- This is not training-ready unless train and development both have enough historical coverage.
- Current official API snapshot can be kept as the newest event layer, but historical backfill is required before model use.
- No model may use `inputs/event_risk_calendar.csv` until the backfill and feature contract pass.

## Outputs

- `validation_layer\event_risk_calendar_coverage_by_split.csv`
- `validation_layer\event_risk_calendar_coverage_by_month.csv`
- `validation_layer\event_risk_calendar_coverage_by_type.csv`
- `decision_layer\event_risk_calendar_coverage_decision.json`
