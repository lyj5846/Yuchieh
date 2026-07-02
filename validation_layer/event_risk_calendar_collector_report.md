# Event Risk Calendar Collector Report

- Scope: official API snapshot collection.
- Formal output: unchanged.
- Model training: not executed.
- New input source: produced but not enabled in `project_config.json`.

## 白話結論

已產生 `inputs/event_risk_calendar.csv`，共 145 筆事件。這份檔案目前只供覆蓋率與防偷看檢查，尚未允許進主模型。

## Event Counts

| event type | rows |
|---|---:|
| attention | 11 |
| disposition | 20 |
| ex_dividend | 89 |
| investor_meeting | 3 |
| material_info | 22 |

## Source Summary

| source | raw rows | kept events | status |
|---|---:|---:|---|
| TWSE material information | 100 | 19 | ok |
| TWSE attention securities | 1 | 0 | ok |
| TWSE disposition securities | 31 | 12 | ok |
| TWSE trading halt and resumption | 1 | 0 | ok |
| TWSE ex-dividend preview | 339 | 61 | ok |
| TPEx material information | 64 | 7 | ok |
| TPEx attention securities | 50 | 11 | ok |
| TPEx disposition securities | 52 | 8 | ok |
| TPEx trading halt and resumption | 1 | 0 | ok |
| TPEx ex-dividend preview | 301 | 27 | ok |

## Boundaries

- This collector does not modify the approved three model inputs.
- This collector does not train or promote a model.
- Rows announced after signal-day close are retained but marked `post_close_pre_next_open=true`.
- Rows without parseable public announcement date/time are excluded.

## Outputs

- `inputs\event_risk_calendar.csv`
- `validation_layer\event_risk_calendar_source_summary.csv`
- `decision_layer\event_risk_calendar_collector_decision.json`
