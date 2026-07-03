# Formal Candidate Tracking Replay

- Generated: 2026-07-03 20:35:22
- As-of date: 2026-07-03
- Scope: formal signal ledger.
- No-lookahead rule: signal rows are locked when created; only tracking fields are updated afterward.
- Buy assumption: next trading day open.
- Success rule: within the next 10 trading days, any close reaches buy open +3%.
- Drawdown: -3% low is tracked as risk context, not automatic failure.
- research_score is not a calibrated probability.

## Status Counts

- failure: 2
- not_started: 1
- success: 10
- tracking: 5

## Files

- `formal_layer\formal_candidate_tracking.csv`
