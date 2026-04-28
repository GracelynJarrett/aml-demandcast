from pathlib import Path
import os

import altair as alt
import mlflow
import pandas as pd
import streamlit as st

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_URI = "models:/DemandCast/Production"
PINNED_RUN_ID = os.getenv("DEMANDCAST_MODEL_RUN_ID", "d4bb5dd4b0eb42f6b0e84558efcd3699")
PINNED_MODEL_URI = f"runs:/{PINNED_RUN_ID}/model"
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "features.parquet"
LOCAL_MODEL_ROOT = Path(__file__).resolve().parents[1] / "mlartifacts"

# Keep this list exactly aligned with training feature engineering.
FEATURE_COLS = [
    "PULocationID",
    "hour",
    "day_of_week",
    "is_weekend",
    "is_rush_hour",
    "demand_lag_1h",
    "demand_lag_24h",
    "demand_lag_168h",
]

DAY_NAME_TO_NUM = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6,
}


@st.cache_resource
def load_production_model():
    """Load model, preferring a pinned run ID, then Production, then local cache."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    try:
        model = mlflow.pyfunc.load_model(PINNED_MODEL_URI)
        return model, f"Pinned MLflow run {PINNED_RUN_ID}"
    except Exception:
        pass

    try:
        model = mlflow.pyfunc.load_model(MODEL_URI)
        return model, f"MLflow registry at {MLFLOW_TRACKING_URI}"
    except Exception:
        local_model_candidates = sorted(
            LOCAL_MODEL_ROOT.rglob("MLmodel"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )

        if local_model_candidates:
            local_model_path = local_model_candidates[0].parent
            model = mlflow.pyfunc.load_model(str(local_model_path))
            return model, f"local artifact cache at {local_model_path}"

        raise RuntimeError(
            "Could not load the Production model from MLflow or from a local artifact cache."
        )


def get_production_model_details() -> str:
    """Return pinned run plus current Production version and run ID for display."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    details = [f"Pinned run target: {PINNED_RUN_ID}"]

    try:
        client = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions("DemandCast", stages=["Production"])
        if versions:
            version = versions[0]
            details.append(f"Production version {version.version} (run {version.run_id})")
            return " | ".join(details)
    except Exception:
        pass

    details.append("Production model metadata unavailable")
    return " | ".join(details)


@st.cache_data
def load_reference_data() -> pd.DataFrame:
    """Load engineered data for UI defaults and charting."""
    df = pd.read_parquet(DATA_PATH)
    return df


@st.cache_data
def median_defaults(df: pd.DataFrame) -> dict[str, float]:
    """Compute dataset-based medians for lag features."""
    defaults: dict[str, float] = {}
    for col in ["demand_lag_1h", "demand_lag_24h", "demand_lag_168h"]:
        defaults[col] = float(df[col].median()) if col in df.columns else 0.0
    return defaults


def contextual_lag_defaults(
    df: pd.DataFrame,
    pu_location_id: int,
    day_name: str,
    hour_of_day: int,
) -> dict[str, float]:
    """Estimate lag defaults from the most relevant historical slice."""
    defaults = median_defaults(df)

    if "PULocationID" not in df.columns or "hour" not in df.columns:
        return defaults

    context_df = df.copy()
    context_df["hour"] = pd.to_datetime(context_df["hour"], errors="coerce")
    if context_df["hour"].isna().any():
        return defaults

    if "day_of_week" not in context_df.columns:
        context_df["day_of_week"] = context_df["hour"].dt.dayofweek
    if "is_weekend" not in context_df.columns:
        context_df["is_weekend"] = (context_df["day_of_week"] >= 5).astype(int)
    if "is_rush_hour" not in context_df.columns:
        weekday_mask = context_df["day_of_week"] < 5
        rush_mask = context_df["hour"].dt.hour.isin([7, 8, 17, 18])
        context_df["is_rush_hour"] = (weekday_mask & rush_mask).astype(int)

    day_num = DAY_NAME_TO_NUM[day_name]
    is_rush_hour = int((day_num < 5) and (hour_of_day in [7, 8, 17, 18]))

    candidate_masks = [
        (context_df["PULocationID"] == pu_location_id)
        & (context_df["day_of_week"] == day_num)
        & (context_df["is_rush_hour"] == is_rush_hour),
        (context_df["PULocationID"] == pu_location_id) & (context_df["day_of_week"] == day_num),
        (context_df["PULocationID"] == pu_location_id) & (context_df["hour"].dt.hour == hour_of_day),
        context_df["PULocationID"] == pu_location_id,
    ]

    for mask in candidate_masks:
        if mask.any():
            candidate = context_df.loc[mask]
            for col in ["demand_lag_1h", "demand_lag_24h", "demand_lag_168h"]:
                if col in candidate.columns and candidate[col].notna().any():
                    defaults[col] = float(candidate[col].median())
            break

    return defaults


def build_feature_row(
    pu_location_id: int,
    hour_of_day: int,
    day_name: str,
    is_weekend: bool,
    lag_defaults: dict[str, float],
) -> pd.DataFrame:
    """Build a one-row feature DataFrame in the exact training column order."""
    day_num = DAY_NAME_TO_NUM[day_name]
    is_rush_hour = int((not is_weekend) and hour_of_day in [7, 8, 17, 18])

    row = {
        "PULocationID": int(pu_location_id),
        "hour": int(hour_of_day),  # model expects hour-of-day (0-23), not full datetime
        "day_of_week": int(day_num),
        "is_weekend": int(is_weekend),
        "is_rush_hour": int(is_rush_hour),
        "demand_lag_1h": float(lag_defaults.get("demand_lag_1h", 0.0)),
        "demand_lag_24h": float(lag_defaults.get("demand_lag_24h", 0.0)),
        "demand_lag_168h": float(lag_defaults.get("demand_lag_168h", 0.0)),
    }

    return pd.DataFrame([row], columns=FEATURE_COLS)


def main() -> None:
    st.set_page_config(page_title="DemandCast Dashboard", layout="wide")

    st.title("DemandCast - Production Prediction Dashboard")
    st.caption("Part 3 Streamlit app: predicts hourly taxi demand per pickup zone.")

    model = None
    model_source = None
    model_error = None

    try:
        model, model_source = load_production_model()
    except Exception as exc:
        model_error = exc

    # Always display model status clearly at the top
    if model is None:
        st.warning(
            "The dashboard loaded, but the model is unavailable. Start MLflow UI on port 5000 or restore a local model artifact to enable predictions."
        )
        st.error(f"Could not load Production model: {model_error}")
    else:
        st.info(f"✅ **Model loaded:** {model_source}")

    try:
        df = load_reference_data()
    except Exception as exc:
        st.error(f"Could not load reference data from {DATA_PATH}: {exc}")
        st.stop()

    if "PULocationID" in df.columns:
        pu_options = sorted(df["PULocationID"].dropna().astype(int).unique().tolist())
    else:
        pu_options = list(range(1, 266))

    if "selected_pu_location_id" not in st.session_state:
        st.session_state.selected_pu_location_id = pu_options[0]

    # Initialize session state for page selection
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Prediction"

    # Page navigation at the very top of sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Go to:", ["📊 Prediction", "ℹ️ About"], key="page_nav")
    
    # Update session state based on selection
    st.session_state.current_page = "Prediction" if page == "📊 Prediction" else "About"

    st.sidebar.header("Zone Filter")
    pu_location_id = st.sidebar.selectbox(
        "Pickup zone (PULocationID)",
        pu_options,
        index=pu_options.index(st.session_state.selected_pu_location_id)
        if st.session_state.selected_pu_location_id in pu_options
        else 0,
        key="selected_pu_location_id",
    )
    
    st.sidebar.divider()

    # Render the appropriate page
    if st.session_state.current_page == "Prediction":
        render_prediction_page(model, df, pu_location_id)
    else:
        render_about_page(df, model_source, pu_location_id, get_production_model_details())


def render_prediction_page(model, df, pu_location_id):
    """Render the main prediction page."""
    st.sidebar.header("Prediction Inputs")
    hour_of_day = st.sidebar.slider("Hour of day", min_value=0, max_value=23, value=8)
    day_name = st.sidebar.selectbox("Day of week", list(DAY_NAME_TO_NUM.keys()), index=0)

    # Auto-set weekend based on day of week
    is_weekend = day_name in ["Saturday", "Sunday"]
    lag_defaults = contextual_lag_defaults(df, pu_location_id, day_name, hour_of_day)

    with st.sidebar.expander("Estimated lag context", expanded=False):
        st.write("These lag values are estimated from similar historical rows for the selected zone and time.")
        st.dataframe(pd.DataFrame([lag_defaults]), use_container_width=True)

    if st.sidebar.button("Predict demand", type="primary"):
        if model is None:
            st.error("Prediction is unavailable until the model loads successfully.")
            st.stop()

        X_input = build_feature_row(
            pu_location_id=pu_location_id,
            hour_of_day=hour_of_day,
            day_name=day_name,
            is_weekend=is_weekend,
            lag_defaults=lag_defaults,
        )

        try:
            pred = float(model.predict(X_input)[0])
            # store last prediction for later comparison views
            st.session_state.last_prediction = float(pred)
            st.session_state.last_prediction_hour = int(hour_of_day)
            st.session_state.last_prediction_day = str(day_name)
            st.session_state.last_prediction_zone = int(pu_location_id)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            st.stop()

        # Display large prediction metric in center with bigger font
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(f"<h2 style='text-align: center; font-size: 2.5em; margin: 0;'>Predicted demand</h2>", unsafe_allow_html=True)
            st.markdown(f"<h1 style='text-align: center; font-size: 3.5em; margin: 0; color: #1f77b4;'>{pred:.2f} trips</h1>", unsafe_allow_html=True)
            st.caption(f"Zone {pu_location_id} | {day_name} | {hour_of_day:02d}:00")

        # training comparison moved to its own section at the bottom of the Prediction page

        with st.expander("Show feature vector sent to model"):
            st.dataframe(X_input, use_container_width=True)

        # Additional graph: Demand by day of week for the selected zone
        st.subheader("Demand patterns by day of week")
        try:
            zone_data = df[df["PULocationID"] == pu_location_id].copy() if "PULocationID" in df.columns else df.copy()
            
            if not zone_data.empty and "demand" in zone_data.columns:
                # Compute day_of_week from hour column if not present
                if "day_of_week" not in zone_data.columns and "hour" in zone_data.columns:
                    zone_data["hour"] = pd.to_datetime(zone_data["hour"], errors="coerce")
                    zone_data["day_of_week"] = zone_data["hour"].dt.dayofweek
                
                if "day_of_week" in zone_data.columns:
                    day_demand = zone_data.groupby("day_of_week", as_index=False)["demand"].mean().sort_values("day_of_week")
                    day_num_to_name = {v: k for k, v in DAY_NAME_TO_NUM.items()}
                    day_demand["day_name"] = day_demand["day_of_week"].map(day_num_to_name)
                    day_demand["is_selected"] = day_demand["day_name"] == day_name
                    
                    chart = alt.Chart(day_demand).mark_bar().encode(
                        x=alt.X("day_name:N", title="Day of Week", sort=list(DAY_NAME_TO_NUM.keys())),
                        y=alt.Y("demand:Q", title="Average Demand (trips)"),
                        color=alt.condition(
                            alt.datum.is_selected,
                            alt.value("#1f77b4"),  # Blue for selected day
                            alt.value("#cccccc")   # Gray for others
                        )
                    ).properties(width=700, height=300)
                    
                    st.altair_chart(chart, use_container_width=True)
                else:
                    st.info("Could not compute day-of-week from available data")
            else:
                st.info("Insufficient data for day-of-week analysis")
        except Exception as e:
            st.warning(f"Could not render day-of-week chart: {e}")

    # Moved from About page: hourly demand chart for the selected zone.
    pred_val = st.session_state.get("last_prediction", None)
    pred_zone = st.session_state.get("last_prediction_zone", None)
    if pred_val is not None and pred_zone == pu_location_id:
        st.subheader("Average Hourly Demand by Hour of Day")
        try:
            zone_data = df[df["PULocationID"] == pu_location_id].copy() if "PULocationID" in df.columns else df.copy()
            if "hour" in zone_data.columns and "demand" in zone_data.columns:
                hourly_data = zone_data.copy()
                hourly_data["hour"] = pd.to_datetime(hourly_data["hour"], errors="coerce")
                hourly_data = hourly_data[hourly_data["hour"].notna()]

                if not hourly_data.empty:
                    hourly_data["hour_of_day"] = hourly_data["hour"].dt.hour
                    hourly_agg = (
                        hourly_data.groupby("hour_of_day", as_index=False)["demand"]
                        .mean()
                        .sort_values("hour_of_day")
                    )
                    hourly_agg["hour_label"] = hourly_agg["hour_of_day"].astype(str) + ":00"

                    chart = alt.Chart(hourly_agg).mark_bar().encode(
                        x=alt.X("hour_label:N", title="Hour of Day", sort=None),
                        y=alt.Y("demand:Q", title="Average Demand (trips)"),
                    ).properties(width=700, height=300)

                    st.altair_chart(chart, use_container_width=True)
                else:
                    st.info("No valid hourly values available for this zone.")
            else:
                st.info("Insufficient data to render hourly demand chart for this zone.")
        except Exception as e:
            st.warning(f"Could not render hourly demand chart: {e}")
    else:
        st.info("Run a prediction to reveal the average hourly demand chart for this zone.")

def render_about_page(df, model_source, pu_location_id, model_details):
    """Render the About page with metrics explanation and training data info."""
    st.header("Model Status")
    st.markdown(
        f"""
        **Model Source**: {model_source}

        **Model Details**: {model_details}
        
        The DemandCast Production model is loaded from the MLflow Model Registry.
        If MLflow is not available, the dashboard will use a local artifact cache.
        """
    )

    # Metrics explanation section
    st.header("Model Metrics Explained")

    st.subheader("Mean Absolute Error (MAE)")
    st.markdown(
        """
        - **Value**: ~6.88 trips per zone-hour (tuned run d4bb5dd4...)
        - **Meaning**: On average, predictions miss by about 6 to 7 trips per pickup zone per hour.
        - **Interpretation**: Use this as a baseline expectation for prediction error in operational planning.
        """
    )

    st.subheader("Mean Absolute Percentage Error (MAPE)")
    st.markdown(
        """
        - **Value**: 57.794%
        - **Meaning**: Average percentage error across the evaluation split.
        - **Interpretation**: Percent error can still be unstable for low-demand hours, so MAE and RMSE remain the better planning metrics.
        """
    )

    st.subheader("Mean Bias Error (MBE)")
    st.markdown(
        """
        - **Value**: 0.013367 trips
        - **Meaning**: The model is essentially unbiased on average.
        - **Interpretation**: Positive and negative errors are nearly balanced on the evaluation split.
        """
    )

    st.subheader("R² (Coefficient of Determination)")
    st.markdown(
        """
        - **Value**: ~0.9562 (tuned run d4bb5dd4...)
        - **Meaning**: The model explains about 95.6% of the variance in demand.
        - **Interpretation**: Strong model performance; captures major hourly and zone-level patterns well.
        """
    )

    st.subheader("Root Mean Squared Error (RMSE)")
    st.markdown(
        """
        - **Value**: ~15.81 trips (tuned run d4bb5dd4...)
        - **Meaning**: Occasional larger misses still happen, especially around peak-demand periods.
        - **Interpretation**: More pessimistic than MAE; use when larger errors are costly.
        """
    )

    st.header("Plain-Language Summary")
    st.markdown(
        """
        **Quick takeaways for operational planning:**
        
        - **For typical planning**: Use MAE (~6.88 trips) as your margin of error. Most predictions will be within that range.
        - **For conservative estimates**: Use RMSE (~15.81 trips) if you want to account for occasional larger misses.
        - **For demand forecasting**: The tuned model is highly explanatory (R² ≈ 0.956) and captures major patterns well.
        - **Be aware**: MAPE is still relatively high at 57.794%, so rely on MAE/RMSE for most planning decisions.
        - **Bias check**: MBE is 0.013367 trips, which is effectively neutral.
        - **Avoid**: Don't rely solely on MAPE for low-demand hours; prefer MAE or RMSE for planning.
        """
    )

    # Training data charts were moved to the Prediction page.

    # MLflow link
    st.header("View Full Experiment Results")
    st.markdown(
        f"""
        The DemandCast model was trained and tuned using MLflow. View the full experiment results, 
        model registry, and run history:
        
        🔗 **[Open MLflow UI]({MLFLOW_TRACKING_URI})**
        
        In MLflow, you can:
        - View all training runs and their metrics
        - Compare different model versions
        - Check the registered Production model
        - Download model artifacts
        """
    )


main()