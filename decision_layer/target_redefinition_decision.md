# Target Redefinition Decision

- Generated: 2026-07-03 09:27:09
- Data latest date: 2026-06-30
- Decision status: `label_viable_for_training_review`
- Formal output: unchanged by this review.

## 白話結論

新目標可以進入下一步訓練審查，但還不是正式模型。

意思是：先收盤 +3% 才算乾淨成功；如果先跌到 -3%，即使後面又漲到 +3%，這筆不再算成功。

## Split Summary

### train

- Completed rows: 121749
- Old success rate: 51.62%
- Risk-adjusted success rate: 36.06%
- Old successes filtered by risk rule: 18953
- Filtered among old successes: 30.15%
- Average high-close return: 5.59%
- Average adverse low return: -6.17%
- Average realized rule return: -0.60%

### development

- Completed rows: 13893
- Old success rate: 63.05%
- Risk-adjusted success rate: 35.78%
- Old successes filtered by risk rule: 3789
- Filtered among old successes: 43.25%
- Average high-close return: 9.35%
- Average adverse low return: -7.50%
- Average realized rule return: -0.76%

### holdout

- Completed rows: 12847
- Old success rate: 71.93%
- Risk-adjusted success rate: 39.22%
- Old successes filtered by risk rule: 4203
- Filtered among old successes: 45.48%
- Average high-close return: 12.36%
- Average adverse low return: -8.05%
- Average realized rule return: -0.61%

## Decision Reasons

- all splits have enough completed rows
- all splits retain enough risk-adjusted success rows
- holdout has old successes filtered by the adverse-first rule

## Next Step

If accepted, modify the main label contract to train against `risk_adjusted_10d_success` in a separate step.
Do not update formal candidates from this review.

## Outputs

- `label_layer\target_redefinition_contract.md`
- `validation_layer\target_redefinition_baseline.csv`
