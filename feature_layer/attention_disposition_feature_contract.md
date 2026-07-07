# Attention / Disposition Feature Contract

- Status: contract only; not enabled for training.
- Scope: limited attention/disposition features only.
- Formal output: unchanged.
- Model training: not executed.
- Event file remains outside `project_config.json` approved inputs.

## 白話結論

歷史回補資料可以先做注意股與處置股的有限風險特徵，但不能寫成完整事件風險，也不能直接拿去重訓。

## Allowed Source Rows

- `event_type` must be `attention` or `disposition`.
- `known_before_signal_close` must be true.
- `post_close_pre_next_open` must be false for main-model features.
- `signal_usable_date` must be on or before the signal date.
- Rows beyond the latest available market date are excluded from coverage and cannot be scored as historical evidence.

## Approved Feature Names

| feature | type | window | meaning |
|---|---|---|---|
| `attention_disposition_known_count_1d` | numeric | 1 trading day | Count of known attention or disposition rows usable on the signal day. |
| `attention_disposition_known_count_3d` | numeric | 3 trading days | Recent density of known attention or disposition rows. |
| `attention_disposition_known_count_10d` | numeric | 10 trading days | Short-term accumulation of exchange warning or disposition pressure. |
| `attention_active_on_signal_date` | binary | active range | Whether attention status is active and already known on the signal date. |
| `disposition_active_on_signal_date` | binary | active range | Whether disposition status is active and already known on the signal date. |
| `days_since_last_attention_disposition` | numeric | historical | Trading days since the latest eligible attention or disposition row. |
| `has_attention_disposition_history_20d` | binary | 20 trading days | Whether the stock recently had any known attention or disposition history. |

## Forbidden Uses

- Do not use event titles or source text as NLP signals in this contract.
- Do not use post-close events unless the decision-time contract is changed first.
- Do not use event rows as success or failure labels.
- Do not call these features complete event-risk features.
- Do not add the event file to approved model inputs without a separate red-light approval step.

## Coverage Basis

- Usable historical rows: 3840
- Attention rows: 3700
- Disposition rows: 81
- Other rows: 59
- Dominant source: TWSE historical attention (96.07%)

## Next Boundary

下一步若要把這些特徵接進主模型，必須先做 feature generation 防偷看檢查；重訓與正式輸出仍不能在本步驟發生。
