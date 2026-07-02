# Historical Event Risk Backfill Report

- Scope: historical backfill collector output.
- Formal output: unchanged.
- Model training: not executed.
- New input source: produced but not enabled in `project_config.json`.

## 白話結論

已建立歷史事件回補檔，但目前回補主要來自 TWSE 歷史注意/處置資料；TPEx 與重大訊息歷史仍需要另外處理，and still need separate source work. 因此這一步仍不允許重訓。

## Split Rows

| split | rows |
|---|---:|
| train | 2304 |
| development | 542 |
| holdout | 1049 |

## Event Counts

| event type | rows |
|---|---:|
| attention | 3700 |
| disposition | 81 |
| ex_dividend | 89 |
| investor_meeting | 3 |
| material_info | 22 |

## Source Summary

| source | raw rows | kept events | status |
|---|---:|---:|---|
| TWSE historical attention | 10170 | 3689 | ok |
| TWSE historical disposition | 366 | 86 | partial_error |
| current official OpenAPI snapshot | 145 | 145 | ok |

## Boundary

- This file is not added to approved model inputs.
- This file needs a separate coverage review before any model feature contract.
- This step keeps current snapshot rows but does not overwrite the snapshot file.
