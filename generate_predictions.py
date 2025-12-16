"""
M5 Forecasting - Generate Predictions for Submission
This script uses the trained XGBoost model to generate predictions for the submission file.
"""

import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("M5 FORECASTING - PREDICTION GENERATION")
print("="*60)

# Load the trained model
print("\nLoading trained model...")
with open('m5_xgboost_model.pkl', 'rb') as f:
    model = pickle.load(f)

print("Model loaded successfully!")

# Load data
print("\nLoading datasets...")
calendar = pd.read_csv('calendar.csv')
sell_prices = pd.read_csv('sell_prices.csv')
sales = pd.read_csv('sales_train_validation.csv')
sample_submission = pd.read_csv('sample_submission.csv')

print(f"Sales shape: {sales.shape}")
print(f"Sample submission shape: {sample_submission.shape}")

# ============================================================================
# PREPARE DATA FOR PREDICTION
# ============================================================================

print("\n" + "="*60)
print("PREPARING DATA FOR FORECASTING")
print("="*60)

id_cols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
day_cols = [f'd_{i}' for i in range(1, 1914)]

# Transform to long format
print("\nTransforming sales data...")
sales_long = sales.melt(
    id_vars=id_cols,
    value_vars=day_cols,
    var_name='d',
    value_name='sales'
)

# Merge with calendar and prices
sales_long = sales_long.merge(calendar, on='d', how='left')
sales_long = sales_long.merge(
    sell_prices, 
    on=['store_id', 'item_id', 'wm_yr_wk'], 
    how='left'
)

sales_long['date'] = pd.to_datetime(sales_long['date'])
sales_long = sales_long.sort_values(['id', 'date']).reset_index(drop=True)

print(f"Data prepared, shape: {sales_long.shape}")

# ============================================================================
# RECURSIVE FORECASTING (28 DAYS AHEAD)
# ============================================================================

print("\n" + "="*60)
print("GENERATING 28-DAY FORECASTS")
print("="*60)

def create_features(df):
    """Create all features for the model"""
    
    # Time-based features
    df['day_of_week'] = df['wday']
    df['day_of_month'] = df['date'].dt.day
    df['week_of_year'] = df['date'].dt.isocalendar().week
    df['month'] = df['month'].astype(np.int8)
    df['year'] = df['year'].astype(np.int16)
    
    # Event features
    df['has_event_1'] = (df['event_name_1'].notna()).astype(np.int8)
    df['has_event_2'] = (df['event_name_2'].notna()).astype(np.int8)
    
    # SNAP features
    df['snap_CA'] = df['snap_CA'].astype(np.int8)
    df['snap_TX'] = df['snap_TX'].astype(np.int8)
    df['snap_WI'] = df['snap_WI'].astype(np.int8)
    
    # Price features
    df['sell_price'] = df['sell_price'].fillna(0)
    df['price_change'] = df.groupby('id')['sell_price'].transform(lambda x: x.diff())
    df['price_change'] = df['price_change'].fillna(0)
    
    # Lag features
    for lag in [7, 14, 28]:
        df[f'lag_{lag}'] = df.groupby('id')['sales'].shift(lag)
    
    # Rolling features
    for window in [7, 14, 28]:
        df[f'rolling_mean_{window}'] = (
            df.groupby('id')['sales']
            .transform(lambda x: x.shift(1).rolling(window).mean())
        )
        df[f'rolling_std_{window}'] = (
            df.groupby('id')['sales']
            .transform(lambda x: x.shift(1).rolling(window).std())
        )
    
    # Label encode categorical variables
    categorical_cols = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    for col in categorical_cols:
        df[col] = df[col].astype('category').cat.codes
    
    return df

# Get calendar for future dates
future_calendar = calendar[calendar['d'].isin([f'd_{i}' for i in range(1914, 1942)])].copy()

# Store predictions
all_predictions = {}

# Get unique item-store combinations
unique_ids = sales['id'].unique()

print(f"\nForecasting for {len(unique_ids):,} unique time series...")

# Prepare base data for last known date
base_data = sales_long.copy()

# Recursive forecasting for 28 days
for day_ahead in range(1, 29):
    print(f"\nPredicting day {day_ahead}/28...")
    
    # Get calendar info for this forecast day
    d_col = f'd_{1913 + day_ahead}'
    day_calendar = future_calendar[future_calendar['d'] == d_col].copy()
    
    if len(day_calendar) == 0:
        print(f"  Warning: No calendar data for {d_col}")
        continue
    
    # Create prediction dataset
    pred_data_list = []
    
    for unique_id in unique_ids:
        # Get last row for this id
        item_data = base_data[base_data['id'] == unique_id].tail(1).copy()
        
        if len(item_data) == 0:
            continue
        
        # Update date information from calendar
        item_data['d'] = d_col
        item_data['date'] = pd.to_datetime(day_calendar.iloc[0]['date'])
        item_data['wm_yr_wk'] = day_calendar.iloc[0]['wm_yr_wk']
        item_data['weekday'] = day_calendar.iloc[0]['weekday']
        item_data['wday'] = day_calendar.iloc[0]['wday']
        item_data['month'] = day_calendar.iloc[0]['month']
        item_data['year'] = day_calendar.iloc[0]['year']
        item_data['event_name_1'] = day_calendar.iloc[0]['event_name_1']
        item_data['event_type_1'] = day_calendar.iloc[0]['event_type_1']
        item_data['event_name_2'] = day_calendar.iloc[0]['event_name_2']
        item_data['event_type_2'] = day_calendar.iloc[0]['event_type_2']
        item_data['snap_CA'] = day_calendar.iloc[0]['snap_CA']
        item_data['snap_TX'] = day_calendar.iloc[0]['snap_TX']
        item_data['snap_WI'] = day_calendar.iloc[0]['snap_WI']
        
        pred_data_list.append(item_data)
    
    pred_data = pd.concat(pred_data_list, ignore_index=True)
    
    # Prepare features
    pred_data = create_features(pred_data)
    
    # Define feature columns
    feature_cols = [
        'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id',
        'day_of_week', 'day_of_month', 'week_of_year', 'month', 'year',
        'has_event_1', 'has_event_2',
        'snap_CA', 'snap_TX', 'snap_WI',
        'sell_price', 'price_change',
        'lag_7', 'lag_14', 'lag_28',
        'rolling_mean_7', 'rolling_mean_14', 'rolling_mean_28',
        'rolling_std_7', 'rolling_std_14', 'rolling_std_28'
    ]
    
    # Fill any remaining NaN values
    X_pred = pred_data[feature_cols].fillna(0)
    
    # Make predictions
    predictions = model.predict(X_pred)
    predictions = np.maximum(predictions, 0)  # Ensure non-negative
    
    # Store predictions
    pred_data['sales'] = predictions
    all_predictions[f'F{day_ahead}'] = dict(zip(pred_data['id'], predictions))
    
    # Append to base_data for next iteration
    base_data = pd.concat([base_data, pred_data], ignore_index=True)
    base_data = base_data.sort_values(['id', 'date']).reset_index(drop=True)

print("\n" + "="*60)
print("CREATING SUBMISSION FILE")
print("="*60)

# Create submission dataframe
submission = sample_submission.copy()

# Fill predictions
for day in range(1, 29):
    f_col = f'F{day}'
    if f_col in all_predictions:
        submission[f_col] = submission['id'].map(all_predictions[f_col])

# Fill any missing values with 0
submission = submission.fillna(0)

# Save submission
submission.to_csv('submission_xgboost.csv', index=False)

print("\n✓ Submission file created: 'submission_xgboost.csv'")
print(f"  Shape: {submission.shape}")
print(f"\nPreview of predictions:")
print(submission.head(10))

print("\n" + "="*60)
print("PREDICTION GENERATION COMPLETE!")
print("="*60)
