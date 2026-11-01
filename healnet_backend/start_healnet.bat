@echo off
REM ==============================================================================
REM HEALNET — Self-Healing IoT Sensor Network Backend Startup Script (Windows CMD)
REM ==============================================================================

echo [HEALNET] Initializing Edge Monitoring System...

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH. Please install Python 3.10+.
    pause
    exit /b 1
)

REM Train / Verify ML Anomaly Detection Model
if not exist "models\anomaly_model.joblib" (
    echo [HEALNET] Anomaly detection model not found. Training baseline model...
    python ml_pipeline.py --train
)

echo [HEALNET] Starting Processor in a new console...
start "HEALNET Processor" cmd /k "python processor.py"

echo [HEALNET] Starting REST API on http://127.0.0.1:8000 in a new console...
start "HEALNET REST API" cmd /k "python api.py"

echo.
echo [HEALNET] Backend services started.
echo   - Local REST API:    http://127.0.0.1:8000
echo   - Interactive Docs:  http://127.0.0.1:8000/docs
echo   - Health Check:      http://127.0.0.1:8000/api/health
echo.
echo If using real hardware with Coordinator on COM14:
echo   Run: python serial_bridge.py --port COM14
echo.
echo If running in simulation mode:
echo   Run: python simulator.py --demo
echo ==============================================================================

