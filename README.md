# DemandCast

DemandCast is an individual machine learning project focused on forecasting hourly NYC Yellow Taxi demand by pickup zone.

## What is the project?

This repository is the first step in a four-week ML workflow where the goal is to build a reliable demand forecasting model.

In this phase, the focus is on:
- Setting up a clean, reproducible project environment
- Organizing the repository using professional ML project structure
- Performing initial exploratory analysis of the NYC taxi dataset

The insights gathered here will guide future feature engineering, model selection, and evaluation decisions.

## What is the data?

The project uses NYC Yellow Taxi trip records, organized at the hourly level by pickup zone.

At a high level, the dataset provides trip activity information that can be grouped over time and location, allowing analysis of demand patterns such as:
- Rush-hour peaks and overnight lows
- Weekday vs. weekend behavior
- Zone-level differences in ride volume
- Seasonal and calendar-driven variation

This data foundation supports time-series style forecasting with both temporal and geographic context.

## What are we predicting

The prediction target is hourly taxi demand per pickup zone.

In other words, for each zone and each hour, the model estimates the number of taxi pickups expected. This forecast is intended to support better planning and operational decision-making based on anticipated ride demand.
