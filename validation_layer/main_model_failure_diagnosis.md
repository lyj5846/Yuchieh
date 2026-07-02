# Main Model Failure Diagnosis

- Generated: 2026-07-02 23:56:18
- Data latest date: 2026-06-30
- Main model status: `not_promoted`
- Formal output: unchanged

## 結論

主模型沒有升正式，原因不是訓練沒有跑，而是 holdout 沒有同時通過成功率優勢、風險排序與正式驗證檢查。

唯一建議: `review_target_or_data_sufficiency`

風險調整目標重訓後仍未通過正式驗證；不要再補權重或新增分支，下一步應檢討目前三份資料是否足以支撐此交易目標。

## 核心證據

- Target contract: risk_adjusted_10d_success.
- Holdout success rate 35.56%, below same-day market baseline 40.55%.
- Holdout success lift -4.99%; development success lift 19.48%.
- Integrated score high-low success delta 6.94%.
- Success advantage head high-low success delta 4.95%.
- Same-day advantage head high-low advantage delta 3.11%.
- Same-day advantage head high-low soft target delta 3.93%.
- Stable same-day return ranking feature count 9/10.
- Raw same-day advantage Top3 return lift 2.30%.
- Return-ranking probe holdout success lift -1.96%.
- Same-day advantage contribution share inside integrated score 46.65%.
- Selected weight development monthly stability 3/3; objective score 0.176313.
- Risk head high-low failure delta 1.03%; risk separation is not the primary blocker.

## 白話解讀

- 目前模型會收斂，但收斂到的不是可正式使用的選股排序。
- holdout 成功率與報酬優勢都沒有通過，主模型不能升正式。
- 同日報酬排序 head 已轉正，問題已從「head 學反」收斂成「整合分數如何同時保留成功率與報酬優勢」。

## 同日報酬排序根因

- 穩定的同日報酬排序特徵數: 9/10。
- same_day_advantage_head 在 development 的 soft target 高低差: 12.86%。
- same_day_advantage_head 在 holdout 的 soft target 高低差: 3.93%。
- 單看 same_day_advantage_head 的 Top3 return lift: 2.30%。
- 整合分數中的 same_day_advantage 權重貢獻占比: 46.65%。
- holdout 負報酬優勢月份數: 3。

根因判定: `review_target_or_data_sufficiency`。風險調整目標重訓後仍未通過正式驗證；不要再補權重或新增分支，下一步應檢討目前三份資料是否足以支撐此交易目標。

## 整合分數權重根因

- development 月度穩定: 3/3 個出手月份同時通過 success lift 與 return lift。
- development 最差月 success lift: 6.72%。
- development 最差月 return lift: 2.57%。
- 選定權重的 balanced objective score: 0.176313。
- 權重穩定檢查通過: True。

## 輸出檔案

- `validation_layer\main_model_failure_diagnosis.csv`
- `validation_layer\main_model_repair_recommendation.json`
