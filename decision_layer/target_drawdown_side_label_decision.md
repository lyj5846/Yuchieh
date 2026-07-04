# Target Drawdown Side-Label Decision

- Generated: 2026-07-04 08:15:27
- Data latest date: 2026-07-03
- Decision status: `side_label_target_contract_ready`
- Formal output: unchanged by this review.

## 白話結論

這個審查把 `-3%` 從硬性失敗改成風險旁支標籤。

也就是：如果 100 買進，先跌到 97，後來 10 日內收盤到 103 以上，主目標仍算成功；但會另外記錄這是一筆先經過 -3% 回撤的高風險成功。

## Holdout 重點

- 原本 +3% 觸及成功率: 71.84%
- 硬性不能先 -3% 成功率: 39.61%
- 旁支標籤主成功率: 71.84%
- 先碰 -3% 但後來 +3% 的成功占成功樣本: 44.86%
- 平均 10 日最高收盤報酬: 12.24%
- 平均最大不利低點: -8.10%

## Split Summary

### train

- Completed rows: 121749
- Primary +3% success rate: 51.62%
- Hard -3% rule success rate: 36.06%
- Clean success rate: 36.06%
- Painful success rate: 15.57%
- Painful among successes: 30.15%
- Any -3% low rate: 64.27%

### development

- Completed rows: 13893
- Primary +3% success rate: 63.05%
- Hard -3% rule success rate: 35.78%
- Clean success rate: 35.78%
- Painful success rate: 27.27%
- Painful among successes: 43.25%
- Any -3% low rate: 74.17%

### holdout

- Completed rows: 13294
- Primary +3% success rate: 71.84%
- Hard -3% rule success rate: 39.61%
- Clean success rate: 39.61%
- Painful success rate: 32.23%
- Painful among successes: 44.86%
- Any -3% low rate: 71.33%

## Decision Reasons

- primary success remains the 10-day +3% close touch rule
- -3% drawdown is retained as a risk side label instead of automatic failure
- all splits have enough completed rows and recovered success examples

## Next Step

If accepted, update the main model label contract so `target_success` returns to the +3% touch rule and the -3% path becomes a risk side label.
That next step is a red-light change because it changes the model target and requires retraining.

## Outputs

- `label_layer\target_drawdown_side_label_contract.md`
- `validation_layer\target_drawdown_side_label_baseline.csv`
- `decision_layer\target_drawdown_side_label_decision.json`
