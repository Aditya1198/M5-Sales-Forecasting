"""
M5 Forecasting Accuracy - XGBoost Model
This script builds a machine learning model to forecast Walmart sales using XGBoost.
"""

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import gc
import warnings
warnings.filterwarnings('ignore')

print("Loading datasets...")

# Load data
calendar = pd.read_csv('calendar.csv')
sell_prices = pd.read_csv('sell_prices.csv')
sales = pd.read_csv('sales_train_validation.csv')

print(f"Sales shape: {sales.shape}")
print(f"Calendar shape: {calendar.shape}")
print(f"Prices shape: {sell_prices.shape}")

# Sample products to reduce memory usage
print("\nSampling products to reduce memory usage...")
np.random.seed(42)
sample_size = 1000  # Train on 1000 products instead of 30k
sample_idx = np.random.choice(sales.shape[0], size=min(sample_size, sales.shape[0]), replace=False)
sales = sales.iloc[sample_idx].reset_index(drop=True)
print(f"Training on {len(sales)} products")

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

print("\nStarting feature engineering...")

# Melt sales data from wide to long format
id_cols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
day_cols = [f'd_{i}' for i in range(1, 1914)]

print("Transforming sales data to long format...")
sales_long = sales.melt(
    id_vars=id_cols,
    value_vars=day_cols,
    var_name='d',
    value_name='sales'
)

print(f"Long format shape: {sales_long.shape}")

# Merge with calendar
print("Merging with calendar data...")
sales_long = sales_long.merge(calendar, on='d', how='left')

# Merge with prices
print("Merging with price data...")
sales_long = sales_long.merge(
    sell_prices, 
    on=['store_id', 'item_id', 'wm_yr_wk'], 
    how='left'
)

# Convert date to datetime
sales_long['date'] = pd.to_datetime(sales_long['date'])

# Create time-based features
print("Creating time-based features...")
sales_long['day_of_week'] = sales_long['wday']
sales_long['day_of_month'] = sales_long['date'].dt.day
sales_long['week_of_year'] = sales_long['date'].dt.isocalendar().week
sales_long['month'] = sales_long['month'].astype(np.int8)
sales_long['year'] = sales_long['year'].astype(np.int16)

# Create event features
sales_long['has_event_1'] = (sales_long['event_name_1'].notna()).astype(np.int8)
sales_long['has_event_2'] = (sales_long['event_name_2'].notna()).astype(np.int8)

# SNAP benefits features
sales_long['snap_CA'] = sales_long['snap_CA'].astype(np.int8)
sales_long['snap_TX'] = sales_long['snap_TX'].astype(np.int8)
sales_long['snap_WI'] = sales_long['snap_WI'].astype(np.int8)

# Fill missing prices with 0 (items not yet available)
sales_long['sell_price'] = sales_long['sell_price'].fillna(0)

print("\nCreating lag features...")
# Sort by item and date for proper lag calculation
sales_long = sales_long.sort_values(['id', 'date']).reset_index(drop=True)

# Create lag features (previous days sales)
lag_days = [7, 14, 28]
for lag in lag_days:
    print(f"  Creating lag_{lag}...")
    sales_long[f'lag_{lag}'] = sales_long.groupby('id')['sales'].shift(lag)

print("\nCreating rolling window features...")
# Create rolling mean features
rolling_windows = [7, 14, 28]
for window in rolling_windows:
    print(f"  Creating rolling_mean_{window}...")
    sales_long[f'rolling_mean_{window}'] = (
        sales_long.groupby('id')['sales']
        .transform(lambda x: x.shift(1).rolling(window).mean())
    )
    
    print(f"  Creating rolling_std_{window}...")
    sales_long[f'rolling_std_{window}'] = (
        sales_long.groupby('id')['sales']
        .transform(lambda x: x.shift(1).rolling(window).std())
    )

# Price features
print("\nCreating price features...")
sales_long['price_change'] = (
    sales_long.groupby('id')['sell_price']
    .transform(lambda x: x.diff())
)
sales_long['price_change'] = sales_long['price_change'].fillna(0)

# Label encode categorical variables
print("\nEncoding categorical variables...")
categorical_cols = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
for col in categorical_cols:
    sales_long[col] = sales_long[col].astype('category').cat.codes

# ============================================================================
# TRAIN/VALIDATION SPLIT
# ============================================================================

print("\n" + "="*60)
print("PREPARING TRAIN/VALIDATION SPLIT")
print("="*60)

# Use last 28 days as validation (d_1886 onwards = April 24, 2016)
# Let's use a proper cutoff: train up to d_1850, validate on d_1851 to d_1913
train_end_date = pd.Timestamp('2016-03-27')  # Around d_1850
val_start_date = pd.Timestamp('2016-03-28')  # Around d_1851

train_data = sales_long[sales_long['date'] < train_end_date].copy()
val_data = sales_long[sales_long['date'] >= val_start_date].copy()

print(f"Training data: {train_data.shape}")
print(f"Validation data: {val_data.shape}")

# Drop rows with NaN in lag features (early dates)
print("\nDropping rows with missing lag features...")
feature_cols = [
    'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id',
    'day_of_week', 'day_of_month', 'week_of_year', 'month', 'year',
    'has_event_1', 'has_event_2',
    'snap_CA', 'snap_TX', 'snap_WI',
    'sell_price', 'price_change'
] + [f'lag_{lag}' for lag in lag_days] + \
    [f'rolling_mean_{w}' for w in rolling_windows] + \
    [f'rolling_std_{w}' for w in rolling_windows]

train_data_clean = train_data.dropna(subset=feature_cols)
val_data_clean = val_data.dropna(subset=feature_cols)

print(f"Training data after dropping NaN: {train_data_clean.shape}")
print(f"Validation data after dropping NaN: {val_data_clean.shape}")

# Prepare features and target
X_train = train_data_clean[feature_cols]
y_train = train_data_clean['sales']

X_val = val_data_clean[feature_cols]
y_val = val_data_clean['sales']

# Free up memory
del sales_long, train_data, val_data
gc.collect()

# ============================================================================
# MODEL TRAINING
# ============================================================================

print("\n" + "="*60)
print("TRAINING XGBOOST MODEL")
print("="*60)

# Initialize XGBoost model
model = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.1,
    max_depth=8,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0,
    reg_alpha=0.1,
    reg_lambda=1,
    random_state=42,
    n_jobs=-1,
    tree_method='hist'  # Faster training
)

print("\nFitting model...")
model.fit(
    X_train, 
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=50
)

# ============================================================================
# MODEL EVALUATION
# ============================================================================

print("\n" + "="*60)
print("MODEL EVALUATION")
print("="*60)

# Make predictions
print("\nMaking predictions on validation set...")
y_pred_train = model.predict(X_train)
y_pred_val = model.predict(X_val)

# Calculate metrics
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))

train_mae = mean_absolute_error(y_train, y_pred_train)
val_mae = mean_absolute_error(y_val, y_pred_val)

print(f"\nTraining RMSE: {train_rmse:.4f}")
print(f"Validation RMSE: {val_rmse:.4f}")
print(f"\nTraining MAE: {train_mae:.4f}")
print(f"Validation MAE: {val_mae:.4f}")

# Feature importance
print("\n" + "="*60)
print("TOP 20 FEATURE IMPORTANCES")
print("="*60)

feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.head(20).to_string(index=False))

# Save feature importance
feature_importance.to_csv('feature_importance.csv', index=False)
print("\nFeature importance saved to 'feature_importance.csv'")

# ============================================================================
# SAVE MODEL
# ============================================================================

print("\n" + "="*60)
print("SAVING MODEL")
print("="*60)

import pickle

with open('m5_xgboost_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved to 'm5_xgboost_model.pkl'")

print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)
print(f"""
Summary:
- Features: {len(feature_cols)}
- Training samples: {len(X_train):,}
- Validation samples: {len(X_val):,}
- Validation RMSE: {val_rmse:.4f}
- Validation MAE: {val_mae:.4f}
""")
