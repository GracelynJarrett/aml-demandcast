"""Build model-ready features from raw NYC taxi trip data.

Usage
-----
python build_features.py
"""

from pathlib import Path

import pandas as pd

from src.features_skeleton import (
	add_lag_features,
	aggregate_to_hourly_demand,
	clean_data,
	create_temporal_features,
)


def build_features(
	input_path: Path,
	output_path: Path,
	drop_lag_nans: bool = True,
) -> pd.DataFrame:
	"""Run the full feature pipeline and save output parquet."""
	df = pd.read_parquet(input_path)

	cleaned = clean_data(df)
	temporal = create_temporal_features(cleaned)
	hourly = aggregate_to_hourly_demand(temporal)
	hourly = hourly.sort_values(["PULocationID", "hour"]).reset_index(drop=True)
	featured = add_lag_features(hourly, zone_col="PULocationID", target_col="demand")

	if drop_lag_nans:
		lag_cols = ["demand_lag_1h", "demand_lag_24h", "demand_lag_168h"]
		featured = featured.dropna(subset=lag_cols).reset_index(drop=True)

	output_path.parent.mkdir(parents=True, exist_ok=True)
	featured.to_parquet(output_path, index=False)

	return featured


def main() -> None:
	project_root = Path(__file__).resolve().parent
	input_path = project_root / "data" / "yellow_tripdata_2025-01.parquet"
	output_path = project_root / "data" / "features.parquet"

	if not input_path.exists():
		raise FileNotFoundError(f"Raw data not found: {input_path}")

	feature_df = build_features(input_path=input_path, output_path=output_path)
	print(f"Saved features: {output_path}")
	print(f"Output shape: {feature_df.shape}")
	print(f"Columns: {feature_df.columns.tolist()}")


if __name__ == "__main__":
	main()
