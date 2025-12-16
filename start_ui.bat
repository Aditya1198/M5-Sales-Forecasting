@echo off
echo ============================================
echo Starting M5 Forecasting Streamlit UI
echo ============================================
echo.
echo Make sure the API is running on http://localhost:8000
echo.

cd app
streamlit run streamlit_app.py
