import mlflow

mlflow.set_tracking_uri('http://localhost:5000')

b = mlflow.search_runs(experiment_names=['DemandCast_NewMetrics'])
rf = b[b.get('params.model', '').astype(str).str.contains('RandomForestRegressor', na=False)] if len(b) > 0 else b
print('BASELINE_COUNT', len(rf))
if len(rf) > 0:
    base = rf.sort_values('metrics.val_mae', ascending=True).iloc[0]
    print('BASELINE_RUN_ID', base['run_id'])
    for c in ['metrics.val_mae','metrics.val_rmse','metrics.val_r2','metrics.val_mape','metrics.val_mbe','params.n_estimators','params.max_depth','params.min_samples_leaf','params.min_samples_split','params.max_features','params.random_state']:
        if c in base.index:
            print(c, base[c])

r = mlflow.search_runs(experiment_names=['DemandCast_Tuning_Clean'])
comp = r.dropna(subset=['metrics.mean_cv_mae']) if len(r) > 0 else r
print('TUNED_COMPLETED', len(comp))
if len(comp) > 0:
    best = comp.sort_values('metrics.val_mae', ascending=True).iloc[0]
    print('TUNED_RUN_ID', best['run_id'])
    for c in ['metrics.mean_cv_mae','metrics.val_mae','metrics.val_rmse','metrics.val_r2','metrics.val_mape','metrics.val_mbe','metrics.val_mape_excluded_pct','metrics.val_mape_excluded_rows','params.n_estimators','params.max_depth','params.min_samples_leaf','params.min_samples_split','params.max_features','params.random_state']:
        if c in best.index:
            print(c, best[c])
