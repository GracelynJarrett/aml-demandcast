"""Run the required Part 2 model training experiments.

This script calls src.train_skeleton.train_and_log() for three models and logs
all runs to the DemandCast MLflow experiment.
"""

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression

from src.train_skeleton import FEATURE_COLS, train_and_log


def base_params(model_name: str) -> dict:
	"""Return shared MLflow params to keep run logging consistent."""
	return {
		"model": model_name,
		"feature_list": ",".join(FEATURE_COLS),
	}


def main() -> None:
	run_ids: list[str] = []

	linear_model = LinearRegression()
	linear_params = {
		**base_params("LinearRegression"),
		"fit_intercept": linear_model.fit_intercept,
	}
	run_ids.append(
		train_and_log(
			model=linear_model,
			run_name="linear_regression_baseline",
			params=linear_params,
		)
	)

	rf_model = RandomForestRegressor(
		n_estimators=200,
		max_depth=18,
		min_samples_leaf=2,
		random_state=42,
		n_jobs=-1,
	)
	rf_params = {
		**base_params("RandomForestRegressor"),
		"n_estimators": rf_model.n_estimators,
		"max_depth": rf_model.max_depth,
		"min_samples_leaf": rf_model.min_samples_leaf,
		"random_state": rf_model.random_state,
	}
	run_ids.append(
		train_and_log(
			model=rf_model,
			run_name="random_forest_regressor",
			params=rf_params,
		)
	)

	# Gradient boosting is chosen as a third model to capture nonlinear
	# patterns and interactions while still being strong on tabular features.
	gbr_model = GradientBoostingRegressor(
		n_estimators=250,
		learning_rate=0.05,
		max_depth=3,
		random_state=42,
	)
	gbr_params = {
		**base_params("GradientBoostingRegressor"),
		"n_estimators": gbr_model.n_estimators,
		"learning_rate": gbr_model.learning_rate,
		"max_depth": gbr_model.max_depth,
		"random_state": gbr_model.random_state,
	}
	run_ids.append(
		train_and_log(
			model=gbr_model,
			run_name="gradient_boosting_regressor",
			params=gbr_params,
		)
	)

	print("Logged MLflow run IDs:")
	for run_id in run_ids:
		print(f" - {run_id}")


if __name__ == "__main__":
	main()
