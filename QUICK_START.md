# 🚀 Quick Start Guide - M5 Forecasting App

## Your Application is Running! ✅

### Status Check

- **FastAPI Backend**: ✅ Running on http://localhost:8000
- **Streamlit UI**: ✅ Running on http://localhost:8501

---

## How to Use the Application

### Step 1: Access the UI

Open your browser and go to:
```
http://localhost:8501
```

Or click: [Open Streamlit UI](http://localhost:8501)

### Step 2: Configure Your Forecast

In the **left sidebar**, you'll see:

1. **Item ID** dropdown
   - Select a product (e.g., `HOBBIES_1_001`)
   
2. **Store ID** dropdown
   - Select a store (e.g., `CA_1`)
   
3. **Forecast Days** slider
   - Choose how many days ahead to predict (7-56 days)
   - Default: 28 days
   
4. **Historical Days** slider
   - Choose how much history to display (30-365 days)
   - Default: 90 days

### Step 3: Generate Forecast

Click the **🚀 Generate Forecast** button

### Step 4: View Results

You'll see:

#### 📊 Key Metrics (Top Row)
- **Avg Historical Sales**: Average daily sales in the past
- **Avg Forecast Sales**: Average predicted daily sales
- **Total Forecast**: Sum of all forecasted units
- **Expected Change**: Percentage change from historical average

#### 📈 Interactive Chart
- **Blue line**: Historical sales data
- **Orange dashed line**: Forecasted sales
- **Hover** over any point to see details
- **Zoom** and **pan** using mouse controls

#### 📋 Detailed Forecast Table
- Day-by-day predictions
- Dates and day of week
- Exact forecasted values

---

## Example Usage

### Example 1: Quick 7-Day Forecast

1. Select: `HOBBIES_1_001` (Item)
2. Select: `CA_1` (Store)
3. Set Forecast Days: `7`
4. Click **Generate Forecast**

Result: See 1-week sales prediction

### Example 2: Monthly Forecast

1. Select: `FOODS_1_001` (Item)
2. Select: `TX_1` (Store)
3. Set Forecast Days: `28`
4. Set Historical Days: `180`
5. Click **Generate Forecast**

Result: See 4-week forecast with 6 months of history

---

## API Documentation

Want to use the API directly? Visit:
```
http://localhost:8000/docs
```

This opens the **interactive API documentation** where you can:
- Test all endpoints
- See request/response examples
- Try making predictions directly

---

## Troubleshooting

### UI shows "API Offline"

**Solution:**
1. Check if API is running: Open http://localhost:8000/health
2. If not running, open a new terminal and run:
   ```bash
   start_api.bat
   ```

### Prediction fails

**Solution:**
1. Make sure you selected both Item ID and Store ID
2. Check that the API status shows "🟢 API Online"
3. Try a different product/store combination

### UI won't load

**Solution:**
1. Refresh the browser (F5)
2. Check terminal for errors
3. Restart Streamlit:
   ```bash
   start_ui.bat
   ```

---

## Testing the API Directly

### Using Browser

Visit http://localhost:8000/docs and try the `/predict` endpoint:

```json
{
  "item_id": "HOBBIES_1_001",
  "store_id": "CA_1",
  "forecast_days": 28
}
```

### Using cURL

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d "{\"item_id\":\"HOBBIES_1_001\",\"store_id\":\"CA_1\",\"forecast_days\":28}"
```

### Using Python

```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={
        "item_id": "HOBBIES_1_001",
        "store_id": "CA_1",
        "forecast_days": 28
    }
)

forecast = response.json()
print(f"Predictions: {forecast['predictions'][:5]}")  # First 5 days
```

---

## Stopping the Application

### Stop Streamlit UI
- Press `Ctrl+C` in the terminal running Streamlit

### Stop FastAPI Server
- Press `Ctrl+C` in the terminal running the API

---

## Next Steps

1. **Try different products**: Explore various items and stores
2. **Compare forecasts**: Run multiple predictions to see patterns
3. **Export data**: Copy forecast table for reports
4. **Check feature importance**: Review `feature_importance.csv`
5. **Read documentation**: See `DEPLOYMENT.md` for advanced features

---

## Quick Reference

| What | Where |
|------|-------|
| Streamlit UI | http://localhost:8501 |
| API Docs | http://localhost:8000/docs |
| API Health | http://localhost:8000/health |
| Start API | `start_api.bat` |
| Start UI | `start_ui.bat` |
| Deployment Guide | `DEPLOYMENT.md` |
| Model Details | `README.md` |

---

**Happy Forecasting! 📈**
