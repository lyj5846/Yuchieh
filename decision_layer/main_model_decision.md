# Main Model Decision

- Generated: 2026-07-07 08:49:08
- Status: passed_holdout_validation
- Formal approved: True
- Reason: main model passed validation and can be considered by the formal entrypoint
- Training loss: 0.584196 -> 0.424309
- Feature screen: selected 48 of 165 features
- Feature screen holdout usage: audit-only, not selection
- Target contract: drawdown_side_label_10d_touch_success
- Holdout primary +3% touch success rate: 70.64%
- Holdout risk-adjusted success rate: 38.94%
- Holdout success with -3% drawdown side risk among all rows: 31.70%
- Holdout success with -3% drawdown side risk among successes: 44.88%
- Holdout clean success rate: 38.94%
- Holdout painful success rate: 31.70%
- Holdout painful success among successes: 44.88%
- Holdout success rate: 73.33%
- Holdout success lift: 2.69%
- Holdout return lift: 3.39%
- Holdout return-ranking probe success lift: 2.69%
- Holdout return-ranking probe return lift: 0.73%
- Score band ordering valid: True
- Score band ordering blocks promotion: False
- Candidate-region validation passed: True
- Advantage head ordering valid: True
- Return-ranking probe ordering valid: True
- Risk band ordering valid: True
- Active holdout months: 3
- Development monthly positive months: 3/3
- Development min monthly success lift: 25.99%
- Development min monthly return lift: 9.05%
- Selected weight stability passed: True
- Selected development score-band passed: True
- Development score-band success delta: 18.13%
- Development score-band advantage delta: 24.08%
- Development score-band return delta: 5.93%
- Selected balanced objective score: 0.278673

Formal output is not updated by this training pipeline.
