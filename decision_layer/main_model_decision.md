# Main Model Decision

- Generated: 2026-07-04 10:06:50
- Status: passed_holdout_validation
- Formal approved: True
- Reason: main model passed validation and can be considered by the formal entrypoint
- Training loss: 0.592978 -> 0.429557
- Feature screen: selected 48 of 115 features
- Feature screen holdout usage: audit-only, not selection
- Target contract: drawdown_side_label_10d_touch_success
- Holdout primary +3% touch success rate: 71.26%
- Holdout risk-adjusted success rate: 39.55%
- Holdout success with -3% drawdown side risk among all rows: 31.71%
- Holdout success with -3% drawdown side risk among successes: 44.50%
- Holdout clean success rate: 39.55%
- Holdout painful success rate: 31.71%
- Holdout painful success among successes: 44.50%
- Holdout success rate: 83.33%
- Holdout success lift: 14.75%
- Holdout return lift: 5.07%
- Holdout return-ranking probe success lift: 8.98%
- Holdout return-ranking probe return lift: 4.31%
- Score band ordering valid: False
- Score band ordering blocks promotion: False
- Candidate-region validation passed: True
- Advantage head ordering valid: True
- Return-ranking probe ordering valid: True
- Risk band ordering valid: True
- Active holdout months: 2
- Development monthly positive months: 3/3
- Development min monthly success lift: 8.50%
- Development min monthly return lift: 4.30%
- Selected weight stability passed: True
- Selected development score-band passed: True
- Development score-band success delta: 5.83%
- Development score-band advantage delta: 11.90%
- Development score-band return delta: 1.87%
- Selected balanced objective score: 0.173979

Formal output is not updated by this training pipeline.
