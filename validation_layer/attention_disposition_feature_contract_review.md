# Attention / Disposition Feature Contract Review

- Scope: contract review only.
- Formal output: unchanged.
- Model training: not executed.
- New model input: not enabled.

## 白話結論

契約已把事件資料限制在注意/處置用途；它只能描述已知市場警示狀態，不能當作完整事件風險，也不能當成答案標籤。

- Status: `limited_attention_disposition_feature_contract_ready`
- Recommended next step: `build_attention_disposition_feature_generation_check`
- Feature rows approved: 7
- Allowed scope: `attention_disposition_only`
- Coverage source status: `coverage_ready_for_limited_attention_disposition_features`

## Approved Feature Families

- Recent known attention/disposition counts.
- Active attention or disposition status on signal date.
- Days since the latest known attention/disposition row.
- Recent history flag.

## Still Blocked

- Full event-risk wording.
- Material information, suspension, resumption, investor meeting, ex-dividend, and corporate-action features.
- Training with the event file before feature generation leakage checks pass.
- Formal candidate output.
