"""Run Part 3 time-series cross-validation for the best Part 2 model.

This script loads the feature matrix, keeps only the train+validation window,
prepares required feature columns, and runs TimeSeriesSplit CV with a
RandomForestRegressor.
"""

from sklearn.ensemble import RandomForestRegressor
import pandas as pd

from src.cv_skeleton import (
	DATA_PATH,
	FEATURE_COLS,
	SPLIT_METHOD,
	TARGET,
	TRAINVAL_CUTOFF,
	TRAINVAL_RATIO,
	RANDOM_SEED,
	time_series_cv,
)


def build_trainval_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
	"""Prepare train+validation feature matrix and target for CV."""
	if "hour" not in df.columns:
		raise KeyError("Missing required column 'hour' in features dataset.")
	if TARGET not in df.columns:
		raise KeyError(f"Missing required target column: {TARGET}")

	split_ts = pd.to_datetime(df["hour"], errors="coerce")
	if split_ts.isna().any():
		raise ValueError("Column 'hour' contains invalid timestamps after parsing.")

	split_method = SPLIT_METHOD.lower()
	if split_method == "random":
		# For random CV, shuffle the entire dataset then take train+val window
		shuffled_df = df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
		shuffled_ts = split_ts.loc[shuffled_df.index].reset_index(drop=True)
		trainval_end = int(len(shuffled_df) * TRAINVAL_RATIO)
		trainval_end = max(1, min(trainval_end, len(shuffled_df) - 1))
		trainval = shuffled_df.iloc[:trainval_end].copy()
		ts_trainval = shuffled_ts.iloc[:trainval_end].reset_index(drop=True)
	elif split_method == "percentage":
		sorted_df = df.copy()
		sorted_df["_split_ts"] = split_ts
		sort_cols = ["_split_ts"]
		if "PULocationID" in sorted_df.columns:
			sort_cols = ["_split_ts", "PULocationID"]
		sorted_df = sorted_df.sort_values(sort_cols).reset_index(drop=True)

		trainval_end = int(len(sorted_df) * TRAINVAL_RATIO)
		trainval_end = max(1, min(trainval_end, len(sorted_df) - 1))
		trainval = sorted_df.iloc[:trainval_end].copy()
		ts_trainval = trainval["_split_ts"]
	else:
		# Keep only the train+validation window; test remains sealed.
		trainval = df[split_ts < TRAINVAL_CUTOFF].copy()
		ts_trainval = split_ts.loc[trainval.index]
	if trainval.empty:
		raise ValueError("Train+validation window is empty. Check split settings.")

	if "day_of_week" not in trainval.columns:
		trainval["day_of_week"] = ts_trainval.dt.dayofweek
	if "is_weekend" not in trainval.columns:
		trainval["is_weekend"] = (trainval["day_of_week"] >= 5).astype(int)
	if "is_rush_hour" not in trainval.columns:
		is_weekday = trainval["day_of_week"] < 5
		is_rush = ts_trainval.dt.hour.isin([7, 8, 17, 18])
		trainval["is_rush_hour"] = (is_weekday & is_rush).astype(int)

	# Keep split timestamp for chronological sorting, then convert model hour.
	trainval["_split_ts"] = ts_trainval
	trainval["hour"] = ts_trainval.dt.hour

	missing_features = [col for col in FEATURE_COLS if col not in trainval.columns]
	if missing_features:
		raise KeyError(f"Missing required feature columns: {missing_features}")

	sort_cols = ["_split_ts"]
	if "PULocationID" in trainval.columns:
		sort_cols = ["_split_ts", "PULocationID"]
	trainval = trainval.sort_values(sort_cols).reset_index(drop=True)

	X = trainval[FEATURE_COLS]
	y = trainval[TARGET]
	return X, y


def main() -> None:
	df = pd.read_parquet(DATA_PATH)
	X_trainval, y_trainval = build_trainval_xy(df)

	rf_model = RandomForestRegressor(
		n_estimators=200,
		max_depth=18,
		min_samples_leaf=2,
		random_state=42,
		n_jobs=-1,
	)

	results = time_series_cv(
		model=rf_model,
		X=X_trainval,
		y=y_trainval,
		n_splits=5,
		run_name="cv_random_forest_regressor",
	)

	mae_mean = results["mae"].mean()
	mae_std = results["mae"].std()

	print("\nCross-validation fold results:")
	print(results.to_string(index=False))
	print(f"\nCV MAE: {mae_mean:.2f} +/- {mae_std:.2f}")


if __name__ == "__main__":
	main()


"""The model’s 5-fold TimeSeriesSplit result was MAE = 10.52 ± 3.81. 
This indicates good average accuracy, with moderate variability across folds, 
meaning performance is generally stable but still influenced by time-window 
differences in demand."""