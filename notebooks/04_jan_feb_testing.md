# Jan-Feb Dataset Testing Log

## Goal
Track Jan-Feb experiment runs and compare Jan-Feb baseline models against Jan-Feb tuned models.

## Scope
- Data window: January + February (new test path)
- Comparison target: Jan-Feb baseline runs vs Jan-Feb tuned runs
- Keep all notes here so 03_evaluation.md remains baseline-focused

## Step-by-step plan

### Step 1 - Confirm source files
- [x] Confirm both monthly parquet files exist in data folder.
- [x] Note exact file names used for this run.

Files used:
- features.parquet (training/tuning source)
- MLflow experiment: DemandCast_RandomSplits

### Step 2 - Build features for Jan-Feb path
- [x] Run feature build for Jan-Feb data.
- [ ] Record row count and date range from features output.

Feature output notes:
- Rows:
- Min timestamp:
- Max timestamp:
- Null timestamps:

### Step 3 - Split strategy for this path
- [x] Record split method used for this run.
- [x] Record split boundaries or percentages.

Split config used:
- Method: random
- Train: 50%
- Validation: 30%
- Test: 20%

### Step 4 - Train models
- [x] Run training pipeline.
- [x] Capture model run IDs from MLflow.

Training run IDs:
- Linear: (latest not captured here)
- Random Forest: fdbf223f9417432081aade51ced7fa1a
- Gradient Boosting: (latest not captured here)

### Step 5 - Cross-validation
- [x] Run time-series CV.
- [x] Record fold metrics summary.

CV summary:
- MAE mean: 8.4175 (baseline RF CV run)
- MAE std: 1.14 (baseline RF CV run)
- Notes: Baseline CV run name = cv_random_forest_regressor, run_id = 94a74b489f714833b4d779cbc7669edd

### Step 6 - Hyperparameter tuning
- [x] Run tuning.
- [x] Save best params and best validation metrics.

Tuning summary:
- Best trial: optuna_trial_11_20260427T035923Z
- Best params: See MLflow run parameters for run_id be999b37ed2c41e29dcddfcc13e20169
- Best val_mae: 7.754216
- Best val_rmse: 17.773285
- Best val_r2: 0.950351

## Comparison table (Jan-Feb Baseline vs Jan-Feb Tuned)

Metric | Jan-Feb baseline | Jan-Feb tuned | Better run
---|---:|---:|---
val_mae | 7.694700 | 7.754216 | Baseline
val_rmse | 17.778800 | 17.773285 | Jan-Feb tuned
val_r2 | 0.950300 | 0.950351 | Jan-Feb tuned
val_mape | 69.008400 | 70.974166 | Baseline
val_mbe | 0.698500 | 0.721267 | Baseline
cv_mae_mean | 8.417500 | 10.511038 | Baseline
cv_mae_std | 1.140000 | n/a | Baseline (reported)

## Decision notes
- Which path is better overall: Jan-Feb baseline RandomForest is currently better overall.
- Why: Jan-Feb baseline beats Jan-Feb tuned on val_mae, val_mape, val_mbe, and cv_mae_mean; tuned only slightly edges baseline on val_rmse and val_r2.
- Risks/concerns: Tuned trial set appears to overfit fold behavior and did not improve generalization on MAE/MAPE.
- Next action: Keep baseline RF as primary model for this milestone, then expand tuning search space and rerun with consistent CV summary logging.



________________________________________________________________________________________
|    Comparing Split by date and random precetage                                      |
-----------------------------------------------------------------------------------------

### Prepared comparison results (from MLflow)

### Best run by split method

Method | Best run | Experiment | Run ID
---|---|---|---
Date | random_forest_regressor | DemandCast_NewMetrics | a2cfccb85dd340f182072d40f0fd4c7d
Random | optuna_trial_6_20260427T042445Z | DemandCast_RandomSplits | cbcde21d94db43c2a5657c4c35e1f033

### Metric comparison (lower is better for MAE/RMSE/MAPE)

Metric | Date best | Random best | Winner
---|---:|---:|---
val_mae | 7.695 | 6.942 | Random
val_rmse | 17.779 | 16.034 | Random
val_r2 | 0.950 | 0.955 | Random
val_mape | 69.008 | 57.169 | Random
mean_cv_mae | n/a | 7.130 | Random (only value logged)

Conclusion for this section:
- Best random split run is better than best date split run on val_mae, val_rmse, val_r2, and val_mape.
- Improvement in val_mae from date to random: 0.752229 (about 9.78% lower MAE).

