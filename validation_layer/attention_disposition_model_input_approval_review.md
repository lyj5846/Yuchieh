# Attention / Disposition Model Input Approval Review

- Scope: candidate model input approval only.
- Formal output: unchanged.
- Model training: not executed.
- Core allowed inputs remain the original three CSV files.
- Approved only as a limited candidate feature input for the next main-model training run.

## 白話結論

注意/處置事件資料已通過覆蓋率、特徵契約與防偷看生成檢查，因此可以列為下一次主模型重訓的候選特徵輸入；但這一步沒有重訓，也沒有產生正式候選。

- Status: `attention_disposition_candidate_model_input_approved`
- Recommended next step: `wire_attention_disposition_features_into_main_training_pipeline`
- Candidate input key: `attention_disposition_events`
- Scope: `attention_disposition_only`
- Approved feature count: 7

## Boundaries

- The raw event file still cannot be used outside the approved attention/disposition features.
- Event titles and source text are still blocked as NLP inputs.
- Events are still not labels and cannot define success or failure.
- This approval does not update formal candidates.
- This approval does not train the model.

## Next Step

下一步可以修改唯一主模型訓練管線，讓它在重訓時讀取這個候選輸入並產生注意/處置特徵；重訓後仍必須通過 holdout 驗證才可考慮正式輸出。
