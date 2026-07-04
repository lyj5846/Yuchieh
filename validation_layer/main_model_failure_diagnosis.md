# Main Model Failure Diagnosis

- Generated: 2026-07-04 19:44:28
- Data latest date: 2026-07-03
- Main model status: `passed_holdout_validation`
- Formal output: unchanged

## 結論

主模型已通過訓練驗證；但正式輸出仍只能由 `scripts/run_main_pipeline.py` 決定。

唯一建議: `ready_for_formal_review`

回撤旁支標籤主模型已通過訓練驗證；下一步只能由正式入口決定是否更新候選。

## 核心證據

- Target contract: drawdown_side_label_10d_touch_success.
- Holdout success rate 83.33%, above same-day market baseline 68.58%.
- Holdout success lift 14.75%; development success lift 17.15%.
- Integrated score high-low success delta -2.79%.
- Success advantage head high-low success delta -1.85%.
- Same-day advantage head high-low advantage delta 5.44%.
- Same-day advantage head high-low soft target delta 3.48%.
- Stable same-day return ranking feature count 9/10.
- Raw same-day advantage Top3 return lift 4.31%.
- Return-ranking probe holdout success lift 8.98%.
- Same-day advantage contribution share inside integrated score 19.68%.
- Selected weight development monthly stability 3/3; objective score 0.173979.
- Risk head high-low failure delta 0.97%; risk separation is not the primary blocker.

## 白話解讀

- 目前模型會收斂，但收斂到的不是可正式使用的選股排序。
- holdout 同時有成功率優勢與報酬優勢，但仍需確認 head 方向、風險與集中度是否過關。
- 同日報酬排序 head 已轉正，問題已從「head 學反」收斂成「整合分數如何同時保留成功率與報酬優勢」。

## 同日報酬排序根因

- 穩定的同日報酬排序特徵數: 9/10。
- same_day_advantage_head 在 development 的 soft target 高低差: 6.73%。
- same_day_advantage_head 在 holdout 的 soft target 高低差: 3.48%。
- 單看 same_day_advantage_head 的 Top3 return lift: 4.31%。
- 整合分數中的 same_day_advantage 權重貢獻占比: 19.68%。
- holdout 負報酬優勢月份數: 0。

根因判定: `ready_for_formal_review`。回撤旁支標籤主模型已通過訓練驗證；下一步只能由正式入口決定是否更新候選。

## 整合分數權重根因

- development 月度穩定: 3/3 個出手月份同時通過 success lift 與 return lift。
- development 最差月 success lift: 8.50%。
- development 最差月 return lift: 4.30%。
- 選定權重的 balanced objective score: 0.173979。
- 權重穩定檢查通過: True。

## 輸出檔案

- `validation_layer\main_model_failure_diagnosis.csv`
- `validation_layer\main_model_repair_recommendation.json`
