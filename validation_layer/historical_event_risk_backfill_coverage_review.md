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
- Usable rows by market latest date: 3840
- Raw rows including future effective dates: 3895
- Future rows excluded from coverage: 55
- Top event type: attention (96.35%)
- Top source: TWSE historical attention (96.07%)

## Split Coverage

| split | event rows | event days | day coverage | stock coverage | attention | disposition | other |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 2304 | 470 | 96.91% | 59.06% | 2243 | 61 | 0 |
| development | 542 | 55 | 100.00% | 32.28% | 542 | 0 | 0 |
| holdout | 994 | 65 | 100.00% | 62.60% | 915 | 20 | 59 |

## Event Type Concentration

| event type | rows | share | stocks |
|---|---:|---:|---:|
| attention | 3700 | 96.35% | 177 |
| disposition | 81 | 2.11% | 52 |
| ex_dividend | 34 | 0.89% | 34 |
| material_info | 22 | 0.57% | 18 |
| investor_meeting | 3 | 0.08% | 3 |

## Source Concentration

| source | rows | share | event types |
|---|---:|---:|---|
| TWSE historical attention | 3689 | 96.07% | attention |
| TWSE historical disposition | 61 | 1.59% | disposition |
| TWSE material information | 19 | 0.49% | investor_meeting | material_info |
| TWSE ex-dividend preview | 18 | 0.47% | ex_dividend |
| TPEx ex-dividend preview | 15 | 0.39% | ex_dividend |
| TWSE disposition securities | 12 | 0.31% | disposition |
| TPEx attention securities | 11 | 0.29% | attention |
| TPEx disposition securities | 8 | 0.21% | disposition |
| TPEx material information | 7 | 0.18% | ex_dividend | material_info |

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
