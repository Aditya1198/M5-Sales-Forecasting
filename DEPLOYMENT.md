# M5 Forecasting Model Deployment

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8+** installed
2. **Model trained** (run `m5_xgboost_model.py` first)

### Installation

```bash
# Install dependencies
pip install -r app/requirements.txt
```

### Running the Application

#### Option 1: Using Batch Scripts (Windows)

**Terminal 1 - Start API Server:**
```bash
start_api.bat
```

**Terminal 2 - Start Streamlit UI:**
```bash
start_ui.bat
```

#### Option 2: Manual Start

**Terminal 1 - Start API Server:**
```bash
cd app
python -m uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 - Start Streamlit UI:**
```bash
cd app
streamlit run streamlit_app.py
```

### Access the Application

- **Streamlit UI**: http://localhost:8501
- **FastAPI Docs**: http://localhost:8000/docs
- **API Health**: http://localhost:8000/health

---

## 📁 Project Structure

```
m5-forecasting-accuracy/
│
├── app/
│   ├── api.py                 # FastAPI backend
│   ├── streamlit_app.py       # Streamlit UI
│   └── requirements.txt       # Python dependencies
│
├── m5_xgboost_model.py        # Training script
├── generate_predictions.py    # Batch prediction script
├── m5_xgboost_model.pkl       # Trained model (generated)
├── feature_importance.csv     # Feature analysis
│
├── calendar.csv               # Dataset
├── sales_train_validation.csv # Dataset
├── sell_prices.csv            # Dataset
├── sample_submission.csv      # Dataset
│
├── start_api.bat              # API launcher (Windows)
├── start_ui.bat               # UI launcher (Windows)
└── README.md                  # Documentation
```

---

## 🛠️ API Endpoints

### Health Check
```
GET /health
```
Returns API status and model/data loading state.

### List Products
```
GET /products
```
Returns available product IDs.

### List Stores
```
GET /stores
```
Returns available store IDs.

### Get Prediction
```
POST /predict
Content-Type: application/json

{
  "item_id": "HOBBIES_1_001",
  "store_id": "CA_1",
  "forecast_days": 28
}
```
Returns sales forecast for specified product and store.

### Get Historical Data
```
GET /historical/{item_id}/{store_id}?days=90
```
Returns historical sales data.

---

## 🎨 Streamlit UI Features

### 1. Product Selection
- Choose from available products and stores
- Dynamic dropdown menus

### 2. Forecast Configuration
- Adjust forecast horizon (7-56 days)
- Configure historical data range (30-365 days)

### 3. Visualization
- Interactive Plotly charts
- Historical vs. forecast comparison
- Hover tooltips with detailed information

### 4. Metrics Dashboard
- Average historical sales
- Average forecast sales
- Total forecast
- Expected change percentage

### 5. Data Table
- Detailed forecast breakdown
- Day-by-day predictions
- Exportable data

---

## 🧪 Testing the API

### Using cURL

```bash
# Health check
curl http://localhost:8000/health

# Get products
curl http://localhost:8000/products

# Get stores
curl http://localhost:8000/stores

# Make prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"item_id":"HOBBIES_1_001","store_id":"CA_1","forecast_days":28}'

# Get historical data
curl http://localhost:8000/historical/HOBBIES_1_001/CA_1?days=90
```

### Using Python

```python
import requests

# Make prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={
        "item_id": "HOBBIES_1_001",
        "store_id": "CA_1",
        "forecast_days": 28
    }
)
forecast = response.json()
print(forecast)
```

### Using FastAPI Docs

Navigate to http://localhost:8000/docs for interactive API documentation.

---

## 🔧 Configuration

### API Configuration

Edit `app/api.py`:

```python
# Change port
uvicorn.run(app, host="0.0.0.0", port=8080)

# Change model path
with open('../path/to/model.pkl', 'rb') as f:
    model = pickle.load(f)
```

### Streamlit Configuration

Edit `app/streamlit_app.py`:

```python
# Change API URL
API_URL = "http://localhost:8000"

# Change page config
st.set_page_config(
    page_title="Custom Title",
    page_icon="🎯",
    layout="wide"
)
```

---

## 📊 Model Information

- **Algorithm**: XGBoost Regression
- **Features**: 26 engineered features
- **Validation RMSE**: 1.69
- **Validation MAE**: 0.91
- **Training Samples**: 1.8M+

### Top Features
1. Rolling mean (7 days) - 34.2%
2. Rolling mean (14 days) - 17.8%
3. Rolling mean (28 days) - 8.0%
4. Price change - 3.7%
5. Lag 7 days - 2.8%

---

## 🐛 Troubleshooting

### API Won't Start

**Problem**: `ModuleNotFoundError: No module named 'fastapi'`

**Solution**:
```bash
pip install -r app/requirements.txt
```

### Model Not Found

**Problem**: `FileNotFoundError: m5_xgboost_model.pkl`

**Solution**:
```bash
python m5_xgboost_model.py
```

### Streamlit Can't Connect to API

**Problem**: Connection refused

**Solution**:
1. Check if API is running: `curl http://localhost:8000/health`
2. Start API in separate terminal
3. Wait for "Application startup complete" message

### Port Already in Use

**Problem**: `Address already in use`

**Solution**:
```bash
# Kill process on port 8000 (Windows)
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

---

##Performance Considerations

### Memory Usage
- API: ~500MB with model loaded
- Streamlit: ~200MB
- Total: ~700MB minimum

### Response Times
- Prediction endpoint: 100-200ms
- Historical data: 50-100ms
- First request (cold start): 1-2s

### Scalability
- Single instance: 10-50 requests/second
- For production: Use gunicorn/nginx
- Consider caching frequently requested forecasts

---

## 🚢 Production Deployment

### Docker Deployment (Optional)

Create `Dockerfile`:
```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY app/requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t m5-forecasting .
docker run -p 8000:8000 m5-forecasting
```

### Cloud Deployment

**AWS EC2** / **Google Cloud** / **Azure**:
1. Upload files to instance
2. Install dependencies
3. Run with systemd service or PM2
4. Configure reverse proxy (nginx)
5. Set up SSL certificate

---

## 📝 License & Credits

- **Dataset**: M5 Forecasting Accuracy Competition
- **Model**: XGBoost
- **Framework**: FastAPI + Streamlit
- **Visualization**: Plotly

---

## 🆘 Support

For issues or questions:
1. Check the troubleshooting section
2. Review API docs at `/docs`
3. Check server logs for errors
