# Historical Event Risk Backfill Coverage Review

- Scope: historical backfill coverage review only.
- Formal output: unchanged.
- Model training: not executed.
- Backfilled event input: produced but not enabled in `project_config.json`.

## 白話結論

資料跨 train/development/holdout 已足夠，但事件高度集中於 TWSE 注意/處置；只能先做有限風險特徵，不能宣稱完整事件風險。

- Status: `coverage_ready_for_limited_attention_disposition_features`
- Recommended next step: `prepare_limited_attention_disposition_feature_contract`
- Allowed scope: `attention_disposition_only`
- Usable rows by market latest date: 3780
- Raw rows including future effective dates: 3895
- Future rows excluded from coverage: 115
- Top event type: attention (97.59%)
- Top source: TWSE historical attention (97.59%)

## Split Coverage

| split | event rows | event days | day coverage | stock coverage | attention | disposition | other |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 2304 | 470 | 96.91% | 59.06% | 2243 | 61 | 0 |
| development | 542 | 55 | 100.00% | 32.28% | 542 | 0 | 0 |
| holdout | 934 | 61 | 100.00% | 54.33% | 904 | 19 | 11 |

## Event Type Concentration

| event type | rows | share | stocks |
|---|---:|---:|---:|
| attention | 3689 | 97.59% | 168 |
| disposition | 80 | 2.12% | 51 |
| ex_dividend | 11 | 0.29% | 11 |

## Source Concentration

| source | rows | share | event types |
|---|---:|---:|---|
| TWSE historical attention | 3689 | 97.59% | attention |
| TWSE historical disposition | 61 | 1.61% | disposition |
| TWSE disposition securities | 11 | 0.29% | disposition |
| TPEx ex-dividend preview | 9 | 0.24% | ex_dividend |
| TPEx disposition securities | 8 | 0.21% | disposition |
| TWSE ex-dividend preview | 2 | 0.05% | ex_dividend |

## Decision Boundary

- This review does not approve model training.
- If allowed scope is limited, features must be named as attention/disposition risk only.
- Full event-risk wording is blocked until TPEx, material information, and corporate action history are separately covered.
- `event_risk_calendar_backfilled.csv` remains outside approved model inputs.

## Outputs

- `validation_layer\historical_event_risk_backfill_coverage_by_split.csv`
- `validation_layer\historical_event_risk_backfill_coverage_by_type.csv`
- `validation_layer\historical_event_risk_backfill_coverage_by_source.csv`
- `validation_layer\historical_event_risk_backfill_coverage_by_month.csv`
- `decision_layer\historical_event_risk_backfill_coverage_decision.json`
