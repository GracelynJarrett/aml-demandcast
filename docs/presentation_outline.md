1.	Problem
Demadcast is an ML prediction system that forecasts hourly taxi demand per pickup location in NYC. Users give the ML a zone and an hour. The model will then predict how many trips will occur in the zone-hour. Being able to predict when and where high demand will be can allocate taxi drivers and vehicles to high-demand zones before the demand spikes.
2.	Data & Features
The dataset was NYC hourly taxi demand aggregated by pick zones spanning January-February 2025. Feature engineered included:
    1.	PULocationID — pickup zone identifier (1–265)
    2.	hour — hour of day (0–23)
    3.	day_of_week — day number (0=Monday, 6=Sunday)
    4.	is_weekend — binary flag for Saturday/Sunday
    5.	is_rush_hour — binary flag for weekday 7 am, 8 am, 5 pm, 6 pm
    6.	demand_lag_1h — demand 1 hour ago
    7.	demand_lag_24h — demand 24 hours ago
    8.	demand_lag_168h — demand 7 days ago (same hour, same day of week)
The trained random forest assigns 94% of the feature importance to the threshold variables, indicating that historical demand is the strongest signal for future demand. During rush hour in Zone 48, which generates over 100 ride demands, these high demands drive the need for zone-specific predictions, as other zones have low ride demands at the same time of day.
3.	Model
I tried the linear_regression, random_forest, and gradient_boosting models. Random forest was the best model. Additionally, I tried changing how I split my data. I started by splitting by dates, then by random precinct. In the end, both splitting methods were very close, with random pretesting slightly better than date-based splitting.
4.	Demo
In my Streamlit app, I will demonstrate how users can input the pickup, location ID, the day of the week, and the time of day. From there, the prediction will be displayed on the main board, along with two graphs showing average demand by day and location. My app also has an about page that explains the models’ metrics in plain English, so that those who don’t understand them can interpret them.
5.	Reflection
One thing that surprised me was how close the splitting by date and the random split were. I was expecting the random split to greatly improve my model, but it did improve by a bit, not as much as I expected. I would have liked to try adding different inputs to see if it improved the models’ metrics. I would like to add the number of passages and see whether, since we added February, the month has an impact on our model.
