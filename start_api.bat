@echo off
echo ============================================
echo Starting M5 Forecasting API Server
echo ============================================
echo.

cd app
python -m uvicorn api:app --reload --host 0.0.0.0 --port 8000
