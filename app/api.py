"""
FastAPI Application for M5 Forecasting Model
Serves predictions via REST API endpoints
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import pandas as pd
import numpy as np
import pickle
import logging
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="M5 Forecasting API",
    description="XGBoost-based sales forecasting for Walmart products",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and data
model = None
calendar_df = None
prices_df = None
sales_df = None

# ============================================================================
# Pydantic Models
# ============================================================================

class PredictionRequest(BaseModel):
    """Request model for single prediction"""
    item_id: str = Field(..., example="HOBBIES_1_001")
    store_id: str = Field(..., example="CA_1")
    forecast_days: int = Field(default=28, ge=1, le=56, description="Number of days to forecast")

class PredictionResponse(BaseModel):
    """Response model for predictions"""
    item_id: str
    store_id: str
    predictions: List[float]
    dates: List[str]
    forecast_days: int

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    data_loaded: bool
    timestamp: str

# ============================================================================
# Startup and Utility Functions
# ============================================================================

@app.on_event("startup")
async def load_model_and_data():
    """Load model and data on startup"""
    global model, calendar_df, prices_df, sales_df
    
    try:
        logger.info("Loading model...")
        with open('../m5_xgboost_model.pkl', 'rb') as f:
            model = pickle.load(f)
        logger.info("✓ Model loaded successfully")
        
        logger.info("Loading data...")
        calendar_df = pd.read_csv('../calendar.csv')
        prices_df = pd.read_csv('../sell_prices.csv')
        sales_df = pd.read_csv('../sales_train_validation.csv')
        
        # Convert date column
        calendar_df['date'] = pd.to_datetime(calendar_df['date'])
        
        logger.info("✓ Data loaded successfully")
        logger.info(f"  Sales shape: {sales_df.shape}")
        logger.info(f"  Calendar shape: {calendar_df.shape}")
        logger.info(f"  Prices shape: {prices_df.shape}")
        
    except Exception as e:
        logger.error(f"Error loading model/data: {e}")
        raise

def create_features(df):
    """Create features for the model"""
    
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
    
    # Encode categorical variables
    categorical_cols = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    for col in categorical_cols:
        df[col] = df[col].astype('category').cat.codes
    
    return df

# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "message": "M5 Forecasting API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "products": "/products",
            "stores": "/stores"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        data_loaded=all([calendar_df is not None, prices_df is not None, sales_df is not None]),
        timestamp=datetime.now().isoformat()
    )

@app.get("/products", response_model=List[str])
async def get_products():
    """Get list of available products"""
    if sales_df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    # Get unique item_ids (first 100 for demo)
    items = sorted(sales_df['item_id'].unique()[:100].tolist())
    return items

@app.get("/stores", response_model=List[str])
async def get_stores():
    """Get list of available stores"""
    if sales_df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    stores = sorted(sales_df['store_id'].unique().tolist())
    return stores

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Generate sales forecast for a product-store combination"""
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Get product info
        product_key = f"{request.item_id}_{request.store_id}"
        product_data = sales_df[
            (sales_df['item_id'] == request.item_id) & 
            (sales_df['store_id'] == request.store_id)
        ]
        
        if product_data.empty:
            raise HTTPException(
                status_code=404, 
                detail=f"Product {request.item_id} not found in store {request.store_id}"
            )
        
        # Prepare historical data
        id_cols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
        day_cols = [f'd_{i}' for i in range(1, 1914)]
        
        # Melt to long format
        sales_long = product_data.melt(
            id_vars=id_cols,
            value_vars=day_cols,
            var_name='d',
            value_name='sales'
        )
        
        # Merge with calendar and prices
        sales_long = sales_long.merge(calendar_df, on='d', how='left')
        sales_long = sales_long.merge(
            prices_df, 
            on=['store_id', 'item_id', 'wm_yr_wk'], 
            how='left'
        )
        
        sales_long = sales_long.sort_values('date').reset_index(drop=True)
        
        # Recursive forecasting
        predictions = []
        forecast_dates = []
        
        # Get future calendar
        last_date = sales_long['date'].max()
        
        for day_ahead in range(1, request.forecast_days + 1):
            forecast_date = last_date + timedelta(days=day_ahead)
            forecast_dates.append(forecast_date.strftime('%Y-%m-%d'))
            
            # Get last row
            last_row = sales_long.tail(1).copy()
            
            # Update date info
            last_row['date'] = forecast_date
            last_row['d'] = f'd_{1913 + day_ahead}'
            
            # Simple date features (since we may not have future calendar)
            last_row['wday'] = forecast_date.dayofweek + 1
            last_row['weekday'] = forecast_date.strftime('%A')
            last_row['month'] = forecast_date.month
            last_row['year'] = forecast_date.year
            last_row['day_of_week'] = last_row['wday']
            last_row['day_of_month'] = forecast_date.day
            last_row['week_of_year'] = forecast_date.isocalendar()[1]
            
            # Create features
            temp_df = pd.concat([sales_long, last_row], ignore_index=True)
            temp_df = create_features(temp_df)
            
            # Get prediction row
            pred_row = temp_df.tail(1)
            
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
            
            X_pred = pred_row[feature_cols].fillna(0)
            
            # Predict
            pred = model.predict(X_pred)[0]
            pred = max(0, pred)  # Ensure non-negative
            predictions.append(float(pred))
            
            # Update sales_long with prediction
            last_row['sales'] = pred
            sales_long = pd.concat([sales_long, last_row], ignore_index=True)
        
        return PredictionResponse(
            item_id=request.item_id,
            store_id=request.store_id,
            predictions=predictions,
            dates=forecast_dates,
            forecast_days=request.forecast_days
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/historical/{item_id}/{store_id}")
async def get_historical_data(item_id: str, store_id: str, days: int = 90):
    """Get historical sales data for a product"""
    
    if sales_df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    try:
        product_data = sales_df[
            (sales_df['item_id'] == item_id) & 
            (sales_df['store_id'] == store_id)
        ]
        
        if product_data.empty:
            raise HTTPException(
                status_code=404, 
                detail=f"Product {item_id} not found in store {store_id}"
            )
        
        # Get last N days
        day_cols = [f'd_{i}' for i in range(max(1, 1913-days), 1914)]
        sales_values = product_data[day_cols].values[0].tolist()
        
        # Get corresponding dates
        dates = calendar_df[calendar_df['d'].isin(day_cols)]['date'].dt.strftime('%Y-%m-%d').tolist()
        
        return {
            "item_id": item_id,
            "store_id": store_id,
            "dates": dates,
            "sales": sales_values
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting historical data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
