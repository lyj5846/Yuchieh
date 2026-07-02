# Main Model Training Spec

- Generated: 2026-07-02 22:55:49
- Confirmed plan: `single_main_model_training_plan`
- Data sources: three approved CSV inputs only.
- Model: one hidden-layer numpy MLP with four outputs.
- Training heads: selection_success, same_day_advantage soft target, failure_risk, episode_start.
- Same-day advantage soft target: pure same-day return percentile.
- Uses same-day relative return-ranking features against all stocks, same industry, and market indices.
- same_day_advantage loss weight: 3.0.
- Strategy tuning: selected on development with monthly stability and a balanced success/return objective.
- Development monthly stability requires most active months to have both success lift and return lift above zero.
- Formal target_success is unchanged: next-day open buy, any close within 10 trading days reaches +3%.
- Feature lookback: 20 trading days.
- Episode gap: 10 trading days.
- Selected weights: 1.0, 1.2, 0.0, 0.1
- Selected gate: 1.5101522445678712
- Selected development positive months: 3/3
- Selected balanced objective score: 0.226468
- Raw outputs are research ranking scores, not calibrated success rates.
