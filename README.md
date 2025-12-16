# M5 Forecasting Accuracy - XGBoost Model

## 📊 Project Overview

This project builds a **machine learning forecasting model** for the M5 Forecasting Accuracy competition using **XGBoost**. The model predicts daily sales for Walmart products across different stores using historical sales data, calendar information, and pricing data.

## 📁 Dataset Description

### Files Included

1. **calendar.csv** (103 KB, 1,969 rows)
   - Date information and calendar features
   - Columns: `date`, `wm_yr_wk`, `weekday`, `wday`, `month`, `year`, `d` (day ID)
   - Event information: `event_name_1`, `event_type_1`, `event_name_2`, `event_type_2`
   - SNAP benefits indicators: `snap_CA`, `snap_TX`, `snap_WI`

2. **sales_train_validation.csv** (115 MB, 30,490 products)
   - Historical daily sales data
   - 1,913 days of sales history (d_1 to d_1913)
   - Time period: January 29, 2011 to May 22, 2016
   - ID columns: `id`, `item_id`, `dept_id`, `cat_id`, `store_id`, `state_id`

3. **sales_train_evaluation.csv** (116 MB, 30,490 products)
   - Extended sales data with 28 additional days
   - 1,941 days total (d_1 to d_1941)
   - Includes validation period (d_1914 to d_1941)

4. **sell_prices.csv** (194 MB, 6.8M rows)
   - Weekly price information by store and item
   - Columns: `store_id`, `item_id`, `wm_yr_wk`, `sell_price`

5. **sample_submission.csv** (5 MB, 60,980 rows)
   - Submission template for 28-day forecasts
   - Columns: `id`, `F1` to `F28`

### Dataset Hierarchy

```
3 Categories (HOBBIES, FOODS, HOUSEHOLD)
  └─ 7 Departments
      └─ 3,049 Products
          └─ 10 Stores (3 states: CA, TX, WI)
              = 30,490 Time Series
```

## 🚀 Model Implementation

### Feature Engineering

The model uses **26 engineered features**:

#### 1. **Lag Features** (Historical sales)
- `lag_7`: Sales from 7 days ago
- `lag_14`: Sales from 14 days ago
- `lag_28`: Sales from 28 days ago (4 weeks)

#### 2. **Rolling Window Statistics**
- `rolling_mean_7/14/28`: Moving averages
- `rolling_std_7/14/28`: Rolling standard deviations

#### 3. **Time-Based Features**
- `day_of_week`, `day_of_month`, `week_of_year`
- `month`, `year`

#### 4. **Event Features**
- `has_event_1`, `has_event_2`: Binary indicators for special events

#### 5. **SNAP Benefits**
- `snap_CA`, `snap_TX`, `snap_WI`: SNAP eligibility by state

#### 6. **Price Features**
- `sell_price`: Current selling price
- `price_change`: Price change from previous week

#### 7. **Categorical Features** (Label encoded)
- `item_id`, `dept_id`, `cat_id`, `store_id`, `state_id`

### XGBoost Configuration

```python
XGBRegressor(
    n_estimators=1000,
    learning_rate=0.1,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,  # L1 regularization
    reg_lambda=1,   # L2 regularization
    tree_method='hist'
)
```

## 📈 Model Performance

### Training Results

- **Training Samples**: 1,856,000 (1000 products × ~1,856 days)
- **Validation Samples**: 28,000 (1000 products × 28 days)
- **Training RMSE**: 1.18
- **Validation RMSE**: 1.69
- **Training MAE**: 0.61
- **Validation MAE**: 0.91

### Feature Importance (Top 10)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `rolling_mean_7` | 34.2% |
| 2 | `rolling_mean_14` | 17.8% |
| 3 | `rolling_mean_28` | 8.0% |
| 4 | `price_change` | 3.7% |
| 5 | `lag_7` | 2.8% |
| 6 | `state_id` | 2.7% |
| 7 | `sell_price` | 2.3% |
| 8 | `snap_WI` | 2.3% |
| 9 | `lag_28` | 2.2% |
| 10 | `lag_14` | 2.2% |

**Key Insight**: Rolling mean features (short-term averages) are the most predictive, accounting for 60% of model importance.

## 🎯 Usage

### Train the Model

```bash
python m5_xgboost_model.py
```

This script will:
1. Load and preprocess the data
2. Engineer features
3. Train the XGBoost model
4. Evaluate performance
5. Save the model to `m5_xgboost_model.pkl`
6. Export feature importance to `feature_importance.csv`

### Generate Predictions

```bash
python generate_predictions.py
```

This script will:
1. Load the trained model
2. Generate 28-day forecasts recursively
3. Create submission file `submission_xgboost.csv`

## 📝 Files Created

- `m5_xgboost_model.py` - Training script
- `generate_predictions.py` - Prediction generation script
- `m5_xgboost_model.pkl` - Trained model (saved after training)
- `feature_importance.csv` - Feature importance analysis
- `submission_xgboost.csv` - Competition submission file

## 🔧 Requirements

```
pandas
numpy
xgboost
scikit-learn
```

Install with:
```bash
pip install xgboost scikit-learn pandas numpy
```

## 💡 Model Insights

1. **Recent trends matter most**: 7-day rolling mean is the most important feature (34%)
2. **Price sensitivity**: Price changes and current prices significantly impact sales
3. **Regional differences**: State-level features (SNAP benefits, state_id) are important
4. **Seasonality captured**: Week and month features help capture seasonal patterns

## 🎓 Future Improvements

To further improve the model:

1. **Hyperparameter tuning**: Use GridSearchCV or Bayesian optimization
2. **More features**: Add promotional flags, competitor pricing, weather data
3. **Hierarchical forecasting**: Train separate models for different product categories
4. **Ensemble methods**: Combine XGBoost with LightGBM, CatBoost
5. **Scale to full dataset**: Train on all 30,490 products (requires more memory/compute)

## 📊 Model Architecture

```
Input Features (26)
    ↓
[XGBoost Trees]
 - 1000 estimators
 - Max depth: 8
 - Histogram-based
    ↓
Sales Prediction
```

---

**Note**: This implementation uses 1,000 sampled products for efficient training. To scale to the full dataset (30,490 products), consider using distributed computing or incremental learning approaches.
