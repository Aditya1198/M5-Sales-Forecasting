"""
Streamlit UI for M5 Forecasting Model
Interactive interface for sales predictions
"""

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import time

# ============================================================================
# Configuration
# ============================================================================

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="M5 Sales Forecasting",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# Helper Functions
# ============================================================================

def check_api_health():
    """Check if API is available"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def get_products():
    """Fetch available products from API"""
    try:
        response = requests.get(f"{API_URL}/products")
        if response.status_code == 200:
            return response.json()
        return []
    except:
        return []

def get_stores():
    """Fetch available stores from API"""
    try:
        response = requests.get(f"{API_URL}/stores")
        if response.status_code == 200:
            return response.json()
        return []
    except:
        return []

def get_historical_data(item_id, store_id, days=90):
    """Fetch historical sales data"""
    try:
        response = requests.get(f"{API_URL}/historical/{item_id}/{store_id}?days={days}")
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def get_prediction(item_id, store_id, forecast_days):
    """Get sales forecast"""
    try:
        payload = {
            "item_id": item_id,
            "store_id": store_id,
            "forecast_days": forecast_days
        }
        response = requests.post(f"{API_URL}/predict", json=payload)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"Error: {e}")
        return None

# ============================================================================
# UI Components
# ============================================================================

def render_header():
    """Render page header"""
    st.title("📊 M5 Sales Forecasting Dashboard")
    st.markdown("**XGBoost-powered sales predictions for Walmart products**")
    
    # API status indicator
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        if check_api_health():
            st.success("🟢 API Online")
        else:
            st.error("🔴 API Offline")
    
    st.markdown("---")

def render_sidebar():
    """Render sidebar with controls"""
    st.sidebar.header("🎯 Forecast Configuration")
    
    # Product selection
    st.sidebar.subheader("Product Selection")
    products = get_products()
    stores = get_stores()
    
    if not products or not stores:
        st.sidebar.error("Unable to load products/stores. Make sure the API is running.")
        return None, None, None
    
    selected_item = st.sidebar.selectbox(
        "Item ID",
        options=products,
        index=0,
        help="Select a product to forecast"
    )
    
    selected_store = st.sidebar.selectbox(
        "Store ID",
        options=stores,
        index=0,
        help="Select a store location"
    )
    
    # Forecast parameters
    st.sidebar.subheader("Forecast Settings")
    
    forecast_days = st.sidebar.slider(
        "Forecast Days",
        min_value=7,
        max_value=56,
        value=28,
        step=7,
        help="Number of days to forecast (multiples of 7)"
    )
    
    historical_days = st.sidebar.slider(
        "Historical Days",
        min_value=30,
        max_value=365,
        value=90,
        step=30,
        help="Number of historical days to display"
    )
    
    # Predict button
    st.sidebar.markdown("---")
    predict_button = st.sidebar.button("🚀 Generate Forecast", type="primary", use_container_width=True)
    
    return {
        "item_id": selected_item,
        "store_id": selected_store,
        "forecast_days": forecast_days,
        "historical_days": historical_days,
        "predict_button": predict_button
    }

def create_forecast_chart(historical_data, forecast_data):
    """Create interactive forecast visualization"""
    
    fig = make_subplots(
        rows=1, cols=1,
        subplot_titles=["Sales Forecast"]
    )
    
    # Historical data
    if historical_data:
        fig.add_trace(
            go.Scatter(
                x=historical_data['dates'],
                y=historical_data['sales'],
                mode='lines',
                name='Historical Sales',
                line=dict(color='#1f77b4', width=2),
                hovertemplate='<b>Date</b>: %{x}<br><b>Sales</b>: %{y:.1f}<extra></extra>'
            )
        )
    
    # Forecast data
    if forecast_data:
        fig.add_trace(
            go.Scatter(
                x=forecast_data['dates'],
                y=forecast_data['predictions'],
                mode='lines+markers',
                name='Forecast',
                line=dict(color='#ff7f0e', width=3, dash='dash'),
                marker=dict(size=8),
                hovertemplate='<b>Date</b>: %{x}<br><b>Forecast</b>: %{y:.1f}<extra></extra>'
            )
        )
    
    fig.update_layout(
        height=500,
        hovermode='x unified',
        xaxis_title="Date",
        yaxis_title="Sales (Units)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template="plotly_white"
    )
    
    return fig

def render_metrics(historical_data, forecast_data):
    """Render key metrics"""
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if historical_data:
            avg_sales = sum(historical_data['sales']) / len(historical_data['sales'])
            st.metric(
                "Avg Historical Sales",
                f"{avg_sales:.1f}",
                help="Average daily sales in historical period"
            )
    
    with col2:
        if forecast_data:
            avg_forecast = sum(forecast_data['predictions']) / len(forecast_data['predictions'])
            st.metric(
                "Avg Forecast Sales",
                f"{avg_forecast:.1f}",
                help="Average daily forecast for next period"
            )
    
    with col3:
        if forecast_data:
            total_forecast = sum(forecast_data['predictions'])
            st.metric(
                "Total Forecast",
                f"{total_forecast:.0f}",
                help="Total units forecasted"
            )
    
    with col4:
        if historical_data and forecast_data:
            change_pct = ((avg_forecast - avg_sales) / avg_sales) * 100
            st.metric(
                "Expected Change",
                f"{change_pct:+.1f}%",
                delta=f"{change_pct:+.1f}%",
                help="Change from historical average"
            )

def render_forecast_table(forecast_data):
    """Render forecast data table"""
    
    if not forecast_data:
        return
    
    st.subheader("📋 Detailed Forecast")
    
    df = pd.DataFrame({
        'Date': forecast_data['dates'],
        'Forecasted Sales': [f"{x:.2f}" for x in forecast_data['predictions']],
        'Day of Week': [pd.to_datetime(d).strftime('%A') for d in forecast_data['dates']]
    })
    
    st.dataframe(df, use_container_width=True, hide_index=True)

def render_info_section():
    """Render information section"""
    
    with st.expander("ℹ️ About This Model", expanded=False):
        st.markdown("""
        ### XGBoost Sales Forecasting Model
        
        This application uses an **XGBoost gradient boosting model** trained on historical Walmart sales data.
        
        **Key Features:**
        - 📈 **26 engineered features** including lag features, rolling statistics, and price data
        - 🎯 **Validation RMSE**: 1.69 units
        - ⚡ **Fast predictions**: ~100ms per forecast
        
        **What the model uses:**
        - Historical sales patterns (7, 14, 28-day lags)
        - Rolling averages and trends
        - Price information and changes
        - Calendar features (day, week, month)
        - Special events and SNAP benefits
        
        **Best for:**
        - Short to medium-term forecasts (7-28 days)
        - Products with consistent sales patterns
        - Regular inventory planning
        """)

# ============================================================================
# Main Application
# ============================================================================

def main():
    """Main application logic"""
    
    # Header
    render_header()
    
    # Sidebar
    config = render_sidebar()
    
    if config is None:
        st.warning("⚠️ Please start the FastAPI server first: `python app/api.py`")
        return
    
    # Info section
    render_info_section()
    
    # Main content
    if config['predict_button']:
        with st.spinner("🔮 Generating forecast..."):
            # Get historical data
            historical_data = get_historical_data(
                config['item_id'], 
                config['store_id'], 
                config['historical_days']
            )
            
            # Get forecast
            forecast_data = get_prediction(
                config['item_id'],
                config['store_id'],
                config['forecast_days']
            )
            
            if forecast_data:
                st.success(f"✅ Forecast generated successfully!")
                
                # Display metrics
                st.subheader("📊 Key Metrics")
                render_metrics(historical_data, forecast_data)
                
                st.markdown("---")
                
                # Display chart
                st.subheader("📈 Sales Forecast Visualization")
                chart = create_forecast_chart(historical_data, forecast_data)
                st.plotly_chart(chart, use_container_width=True)
                
                st.markdown("---")
                
                # Display table
                render_forecast_table(forecast_data)
                
            else:
                st.error("❌ Failed to generate forecast. Please try again.")
    
    else:
        # Show placeholder
        st.info("👈 Configure your forecast settings in the sidebar and click **Generate Forecast** to begin.")
        
        # Show example
        st.subheader("🎨 Example Preview")
        st.image("https://via.placeholder.com/1200x400/e8f4f8/1f77b4?text=Sales+Forecast+Chart+Will+Appear+Here")

if __name__ == "__main__":
    main()
