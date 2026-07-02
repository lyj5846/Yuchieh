# Main Model Failure Diagnosis

- Generated: 2026-07-02 22:57:19
- Data latest date: 2026-06-30
- Main model status: `not_promoted`
- Formal output: unchanged

## 結論

主模型沒有升正式，原因不是訓練沒有跑，而是 holdout 沒有同時通過成功率優勢、風險排序與正式驗證檢查。

唯一建議: `redefine_return_target`

權重已通過 development 月度穩定，且報酬 lift 為正，但成功率仍輸同日市場；下一步不該再調權重，應檢討正式交易目標。

## 核心證據

- Holdout success rate 61.90%, below same-day market baseline 69.93%.
- Holdout success lift -8.02%; development success lift 25.71%.
- Integrated score high-low success delta 6.32%.
- Success advantage head high-low success delta -2.96%.
- Same-day advantage head high-low advantage delta 3.24%.
- Same-day advantage head high-low soft target delta 4.01%.
- Stable same-day return ranking feature count 9/10.
- Raw same-day advantage Top3 return lift 2.38%.
- Return-ranking probe holdout success lift -1.34%.
- Same-day advantage contribution share inside integrated score 53.43%.
- Selected weight development monthly stability 3/3; objective score 0.226468.
- Risk head high-low failure delta -2.80%; risk separation is not the primary blocker.

## 白話解讀

- 目前模型會收斂，但收斂到的不是可正式使用的選股排序。
- holdout 報酬優勢已轉正，但成功率輸同日市場，代表報酬排序有線索，整合分數仍不夠平衡。
- 同日報酬排序 head 已轉正，問題已從「head 學反」收斂成「整合分數如何同時保留成功率與報酬優勢」。

## 同日報酬排序根因

- 穩定的同日報酬排序特徵數: 9/10。
- same_day_advantage_head 在 development 的 soft target 高低差: 12.94%。
- same_day_advantage_head 在 holdout 的 soft target 高低差: 4.01%。
- 單看 same_day_advantage_head 的 Top3 return lift: 2.38%。
- 整合分數中的 same_day_advantage 權重貢獻占比: 53.43%。
- holdout 負報酬優勢月份數: 1。

根因判定: `redefine_return_target`。權重已通過 development 月度穩定，且報酬 lift 為正，但成功率仍輸同日市場；下一步不該再調權重，應檢討正式交易目標。

## 整合分數權重根因

- development 月度穩定: 3/3 個出手月份同時通過 success lift 與 return lift。
- development 最差月 success lift: 18.65%。
- development 最差月 return lift: 5.72%。
- 選定權重的 balanced objective score: 0.226468。
- 權重穩定檢查通過: True。

## 輸出檔案

- `validation_layer\main_model_failure_diagnosis.csv`
- `validation_layer\main_model_repair_recommendation.json`
