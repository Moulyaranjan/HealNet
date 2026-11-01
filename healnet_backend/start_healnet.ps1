# ==============================================================================
# HEALNET — Self-Healing IoT Sensor Network Startup Script (PowerShell)
# ==============================================================================

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  HEALNET Edge Backend Startup" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# Check Python environment
try {
    $pyVer = python --version
    Write-Host "[OK] Detected Python: $pyVer" -ForegroundColor Green
} catch {
    Write-Error "Python is not installed or not in PATH."
    exit 1
}

# Verify / Train ML Model
if (-not (Test-Path "models\anomaly_model.joblib")) {
    Write-Host "[HEALNET] Anomaly detection model not found. Training baseline..." -ForegroundColor Yellow
    python ml_pipeline.py --train
}

# Start Processor process
Write-Host "[HEALNET] Launching Processor..." -ForegroundColor Cyan
$processorProc = Start-Process python -ArgumentList "processor.py" -PassThru

# Start REST API process
Write-Host "[HEALNET] Launching REST API on http://127.0.0.1:8000..." -ForegroundColor Cyan
$apiProc = Start-Process python -ArgumentList "api.py" -PassThru

Write-Host ""
Write-Host "[HEALNET] Core services running:" -ForegroundColor Green
Write-Host "  - Processor PID: $($processorProc.Id)"
Write-Host "  - API PID:       $($apiProc.Id)"
Write-Host "  - API URL:       http://127.0.0.1:8000"
Write-Host "  - API Docs:      http://127.0.0.1:8000/docs"
Write-Host "  - Health Check:  http://127.0.0.1:8000/api/health"
Write-Host ""
Write-Host "Hardware bridge (Coordinator on COM14):" -ForegroundColor Yellow
Write-Host "  python serial_bridge.py --port COM14"
Write-Host ""
Write-Host "Or simulated demo run:" -ForegroundColor Yellow
Write-Host "  python simulator.py --demo"
Write-Host "============================================================" -ForegroundColor Cyan

