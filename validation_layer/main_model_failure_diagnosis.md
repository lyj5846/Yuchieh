# Main Model Failure Diagnosis

- Generated: 2026-07-03 08:41:05
- Data latest date: 2026-06-30
- Main model status: `not_promoted`
- Formal output: unchanged

## 結論

主模型沒有升正式，原因不是訓練沒有跑，而是 holdout 沒有同時通過成功率優勢、風險排序與正式驗證檢查。

唯一建議: `review_target_or_data_sufficiency`

風險調整目標重訓後仍未通過正式驗證；不要再補權重或新增分支，下一步應檢討目前三份資料是否足以支撐此交易目標。

## 核心證據

- Target contract: risk_adjusted_10d_success.
- Holdout success rate 31.03%, below same-day market baseline 37.06%.
- Holdout success lift -6.02%; development success lift 11.13%.
- Integrated score high-low success delta -6.57%.
- Success advantage head high-low success delta 6.54%.
- Same-day advantage head high-low advantage delta 15.47%.
- Same-day advantage head high-low soft target delta 7.63%.
- Stable same-day return ranking feature count 9/10.
- Raw same-day advantage Top3 return lift 4.21%.
- Return-ranking probe holdout success lift -3.92%.
- Same-day advantage contribution share inside integrated score 47.96%.
- Selected weight development monthly stability 3/3; objective score 0.202998.
- Risk head high-low failure delta 1.96%; risk separation is not the primary blocker.

## 白話解讀

- 目前模型會收斂，但收斂到的不是可正式使用的選股排序。
- holdout 報酬優勢已轉正，但成功率輸同日市場，代表報酬排序有線索，整合分數仍不夠平衡。
- 同日報酬排序 head 已轉正，問題已從「head 學反」收斂成「整合分數如何同時保留成功率與報酬優勢」。

## 同日報酬排序根因

- 穩定的同日報酬排序特徵數: 9/10。
- same_day_advantage_head 在 development 的 soft target 高低差: 13.77%。
- same_day_advantage_head 在 holdout 的 soft target 高低差: 7.63%。
- 單看 same_day_advantage_head 的 Top3 return lift: 4.21%。
- 整合分數中的 same_day_advantage 權重貢獻占比: 47.96%。
- holdout 負報酬優勢月份數: 0。

根因判定: `review_target_or_data_sufficiency`。風險調整目標重訓後仍未通過正式驗證；不要再補權重或新增分支，下一步應檢討目前三份資料是否足以支撐此交易目標。

## 整合分數權重根因

- development 月度穩定: 3/3 個出手月份同時通過 success lift 與 return lift。
- development 最差月 success lift: 7.60%。
- development 最差月 return lift: 1.61%。
- 選定權重的 balanced objective score: 0.202998。
- 權重穩定檢查通過: True。

## 輸出檔案

- `validation_layer\main_model_failure_diagnosis.csv`
- `validation_layer\main_model_repair_recommendation.json`
