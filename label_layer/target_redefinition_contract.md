# Target Redefinition Contract

- Generated: 2026-07-03 20:22:41
- Data latest date: 2026-06-30
- Scope: label-only review; no model training; no formal candidates.

## Old Target

- Signal time: after signal-day close.
- Buy assumption: next trading day open.
- Success: any close in the next 1 to 10 trading days is at least +3% above buy open.
- Unfinished: missing next-day open or incomplete 10 trading day window is tracking-only.

## Proposed Risk-Adjusted Target

- Buy assumption stays the same: next trading day open.
- Profit event: any close in the next 1 to 10 trading days is at least +3% above buy open.
- Adverse event: any low in the next 1 to 10 trading days is at least -3% below buy open.
- Success: the profit event happens before the adverse event.
- Failure: the adverse event happens before the profit event, no profit event happens, or both happen on the same day.
- Conservative tie rule: if +3% close and -3% low happen on the same day, adverse event wins.
- This target is a label candidate, not a calibrated probability and not a formal stock recommendation.

## Derived Fields

- `risk_adjusted_10d_success`: proposed success label.
- `max_adverse_return`: worst low return within the 10 trading day window.
- `first_event_day`: first profit/adverse event day, 1 to 10.
- `realized_10d_trade_return`: +3% on profit-first, -3% on adverse-first, otherwise day-10 close return.
- `old_success_but_risk_failed`: old +3% success that fails the adverse-first rule.

## Boundary

- This contract does not modify `target_success` in the current main model.
- This contract does not write formal output.
- If this target is viable, the next step is a separate main-label contract change and model retraining.
