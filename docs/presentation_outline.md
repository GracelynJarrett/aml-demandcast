1.	Problem
DemandCast is an ML prediction system that forecasts hourly taxi demand per pickup location in NYC. A user selects a pickup zone and an hour, and the model predicts how many trips will occur in that zone-hour. This helps taxi drivers and dispatchers position vehicles before demand spikes.

2.	Data & Features
The dataset is NYC hourly taxi demand aggregated by pickup zone across January-February 2025. The engineered features are:
    1.	PULocationID — pickup zone identifier (1–265)
    2.	hour — hour of day (0–23)
    3.	day_of_week — day number (0=Monday, 6=Sunday)
    4.	is_weekend — binary flag for Saturday/Sunday
    5.	is_rush_hour — binary flag for weekday 7 am, 8 am, 5 pm, and 6 pm
    6.	demand_lag_1h — demand 1 hour ago
    7.	demand_lag_24h — demand 24 hours ago
    8.	demand_lag_168h — demand 7 days ago (same hour, same day of week)
The lag features are the strongest signal in the model, which means recent demand patterns matter more than static time labels alone.

3.	Best Model Summary
Best model: Random Forest Regressor.
Best split strategy: random split was slightly better than date-based splitting.
Best parameters:
    1.	n_estimators = 300
    2.	max_depth = 17
    3.	max_features = log2
    4.	min_samples_leaf = 3
    5.	min_samples_split = 8
    6.	random_state = 42

Best metrics:
    1.	MAE = 6.88 trips
    2.	RMSE = 15.81 trips
    3.	R² = 0.9562
    4.	MAPE = 57.794%
    5.	MBE = 0.013367 trips

4.	Why This Model Worked
I compared linear regression, random forest, and gradient boosting. Random forest performed best because it captured the non-linear relationships between zone, time, and lagged demand. The model also performed well on both split strategies, which suggests it is stable rather than overfitted to a single split.

5.	Demo
In the Streamlit app, I show how users choose a pickup zone, day of week, and time of day to generate a prediction. The dashboard also shows model metrics in plain English on the About page, which makes the results easier to interpret for non-technical users.

6.	Reflection
One surprise was how close the date split and random split performed. I expected the random split to improve the model much more, but the difference was small. If I had more time, I would test additional features such as month or other seasonal indicators to see whether they improve performance further.
