# Target Drawdown Side-Label Contract

- Generated: 2026-07-04 10:25:16
- Data latest date: 2026-07-03
- Scope: label-only review; no model training; no formal candidates.

## Primary Target

- Signal time: after signal-day close.
- Buy assumption: next trading day open.
- `target_success`: any close in the next 1 to 10 trading days is at least +3% above buy open.
- If price first drops to -3% but later reaches +3%, `target_success` still remains success.
- Unfinished: missing next-day open or incomplete 10 trading day window is tracking-only.

## Drawdown Risk Side Labels

- `max_adverse_return_10d`: worst low return within the 10 trading day window.
- `hit_minus3_low_anytime_10d`: whether any low in the window reaches -3% below buy open.
- `drawdown_minus3_before_or_same_success`: success path touched -3% before or on the same day as +3% close.
- `clean_success_label`: +3% success without -3% low before or on the success day.
- `painful_success_label`: +3% success after or on the same day as a -3% low.

## Boundary

- This contract does not use -3% as an automatic failure.
- This contract does not modify the current main model target.
- This contract does not write formal output.
- If accepted, the next red-light step is to update the main label contract and retrain.
