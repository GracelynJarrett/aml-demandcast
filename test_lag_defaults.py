import pandas as pd
from pathlib import Path

DATA_PATH = Path('data/features.parquet')
df = pd.read_parquet(DATA_PATH)

DAY_NAME_TO_NUM = {
    'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4,
    'Saturday': 5, 'Sunday': 6,
}

def contextual_lag_defaults(df, pu_location_id, day_name, hour_of_day):
    defaults = {}
    for col in ['demand_lag_1h', 'demand_lag_24h', 'demand_lag_168h']:
        defaults[col] = float(df[col].median()) if col in df.columns else 0.0
    
    if 'PULocationID' not in df.columns or 'hour' not in df.columns:
        return defaults
    
    context_df = df.copy()
    context_df['hour'] = pd.to_datetime(context_df['hour'], errors='coerce')
    if context_df['hour'].isna().any():
        return defaults
    
    if 'day_of_week' not in context_df.columns:
        context_df['day_of_week'] = context_df['hour'].dt.dayofweek
    if 'is_weekend' not in context_df.columns:
        context_df['is_weekend'] = (context_df['day_of_week'] >= 5).astype(int)
    if 'is_rush_hour' not in context_df.columns:
        weekday_mask = context_df['day_of_week'] < 5
        rush_mask = context_df['hour'].dt.hour.isin([7, 8, 17, 18])
        context_df['is_rush_hour'] = (weekday_mask & rush_mask).astype(int)
    
    day_num = DAY_NAME_TO_NUM[day_name]
    is_rush_hour = int((day_num < 5) and (hour_of_day in [7, 8, 17, 18]))
    
    candidate_masks = [
        (context_df['PULocationID'] == pu_location_id) & (context_df['day_of_week'] == day_num) & (context_df['is_rush_hour'] == is_rush_hour),
        (context_df['PULocationID'] == pu_location_id) & (context_df['day_of_week'] == day_num),
        (context_df['PULocationID'] == pu_location_id) & (context_df['hour'].dt.hour == hour_of_day),
        context_df['PULocationID'] == pu_location_id,
    ]
    
    for i, mask in enumerate(candidate_masks):
        if mask.any():
            candidate = context_df.loc[mask]
            print(f'Candidate level {i}: {mask.sum()} rows')
            for col in ['demand_lag_1h', 'demand_lag_24h', 'demand_lag_168h']:
                if col in candidate.columns and candidate[col].notna().any():
                    defaults[col] = float(candidate[col].median())
            break
    
    return defaults

# Test a few zones and times
test_cases = [
    (48, 'Friday', 17),
    (42, 'Friday', 8),
    (265, 'Monday', 12),
    (3, 'Wednesday', 13),
]

for pu_id, day, hour in test_cases:
    lags = contextual_lag_defaults(df, pu_id, day, hour)
    print(f'Zone {pu_id}, {day} {hour}:00')
    print(f'  lag_1h: {lags["demand_lag_1h"]:.1f}')
    print(f'  lag_24h: {lags["demand_lag_24h"]:.1f}')
    print(f'  lag_168h: {lags["demand_lag_168h"]:.1f}')
    print()
