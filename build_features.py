"""Build model-ready features from raw NYC taxi trip data.

Usage
-----
python build_features.py
"""

from pathlib import Path
import os

import pandas as pd

from src.features_skeleton import (
	add_lag_features,
	aggregate_to_hourly_demand,
	clean_data,
	create_temporal_features,
)


def build_features(
	input_path: Path | list[Path],
	output_path: Path,
	drop_lag_nans: bool = True,
	clip_to_dominant_month: bool = True,
) -> pd.DataFrame:
	"""Run the full feature pipeline and save output parquet."""
	if isinstance(input_path, list):
		if not input_path:
			raise FileNotFoundError("No raw data files were provided.")
		raw_parts = [pd.read_parquet(path) for path in input_path]
		df = pd.concat(raw_parts, ignore_index=True)
	else:
		df = pd.read_parquet(input_path)

	cleaned = clean_data(df, clip_to_dominant_month=clip_to_dominant_month)
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
	data_dir = project_root / "data"
	input_path = data_dir / "yellow_tripdata_2025-01.parquet"
	output_path = project_root / "data" / "features.parquet"
	build_all_2025_months = os.getenv("BUILD_ALL_2025_MONTHS", "0") == "1"
	clip_to_dominant_month = os.getenv("CLIP_TO_DOMINANT_MONTH", "1") == "1"

	selected_input: Path | list[Path]
	if build_all_2025_months:
		selected_input = sorted(data_dir.glob("yellow_tripdata_2025-*.parquet"))
	else:
		selected_input = input_path

	if isinstance(selected_input, list):
		if not selected_input:
			raise FileNotFoundError(
				f"No files matching yellow_tripdata_2025-*.parquet found in {data_dir}"
			)
	else:
		if not selected_input.exists():
			raise FileNotFoundError(f"Raw data not found: {selected_input}")

	feature_df = build_features(
		input_path=selected_input,
		output_path=output_path,
		clip_to_dominant_month=clip_to_dominant_month,
	)

	if isinstance(selected_input, list):
		print("Input files:")
		for path in selected_input:
			print(f" - {path.name}")
	else:
		print(f"Input file: {selected_input.name}")
	print(f"build_all_2025_months={build_all_2025_months}")
	print(f"clip_to_dominant_month={clip_to_dominant_month}")
	print(f"Saved features: {output_path}")
	print(f"Output shape: {feature_df.shape}")
	print(f"Columns: {feature_df.columns.tolist()}")


if __name__ == "__main__":
	main()
