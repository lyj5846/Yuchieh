# Event Risk Collector Spec Review

- Scope: collector specification only.
- Formal output: unchanged.
- Model training: not executed.
- New input source: not enabled.

## 白話結論

下一步不是重訓，而是先把事件風險資料定義乾淨。若沒有 `announcement_datetime` 與 `signal_usable_date`，這類資料很容易偷看未來，因此先做規格比直接寫抓取程式更重要。

## Required Future Data

- 重大訊息
- 停牌與復牌
- 注意股與處置股
- 法說會
- 除權息與重大公司事件

## Anti-Leakage Decision

- Main model can use only rows known by signal-day close.
- Post-close rows are stored but blocked from the main model until the decision-time contract changes.
- Missing public timestamp means unusable for training.

## Decision

- Status: `collector_spec_ready`
- Recommended next step: `implement_event_risk_calendar_collector`
- Do not retrain yet: `true`

## Outputs

- `data_layer\event_risk_calendar_source_contract.md`
- `data_layer\event_risk_calendar_schema.csv`
- `decision_layer\event_risk_collector_spec_decision.json`
