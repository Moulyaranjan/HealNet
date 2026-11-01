#!/usr/bin/env bash
# ==============================================================================
# HEALNET — Self-Healing IoT Sensor Network Startup Script (Linux / Raspberry Pi)
# ==============================================================================

set -e

echo "============================================================"
echo "  HEALNET Edge Backend Startup (Linux / Raspberry Pi)"
echo "============================================================"

# Check python3
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] python3 could not be found. Please install Python 3.10+."
    exit 1
fi

# Check if Mosquitto is running on port 1883
if ! nc -z 127.0.0.1 1883 2>/dev/null && ! ss -lnt | grep -q ":1883 "; then
    echo "[WARNING] Mosquitto does not appear to be running on port 1883."
    echo "To start Mosquitto: mosquitto -c mosquitto.conf -d"
fi

# Ensure model exists
if [ ! -f "models/anomaly_model.joblib" ]; then
    echo "[HEALNET] Anomaly detection model not found. Training baseline model..."
    python3 ml_pipeline.py --train
fi

# Start Processor in background
echo "[HEALNET] Launching Processor..."
python3 processor.py &
PROCESSOR_PID=$!

# Start REST API in background
echo "[HEALNET] Launching REST API on http://127.0.0.1:8000..."
python3 api.py &
API_PID=$!

echo ""
echo "[HEALNET] Services running in background:"
echo "  - Processor PID: ${PROCESSOR_PID}"
echo "  - API PID:       ${API_PID}"
echo "  - REST API:      http://127.0.0.1:8000"
echo "  - API Docs:      http://127.0.0.1:8000/docs"
echo "  - Health Check:  http://127.0.0.1:8000/api/health"
echo "  - Summary:       http://127.0.0.1:8000/api/system/summary"
echo ""
echo "To connect serial bridge (e.g., Coordinator on /dev/ttyUSB0):"
echo "  python3 serial_bridge.py --port /dev/ttyUSB0"
echo ""
echo "Or run simulator demo:"
echo "  python3 simulator.py --demo"
echo ""
echo "Press Ctrl+C to terminate background services."

trap "kill ${PROCESSOR_PID} ${API_PID} 2>/dev/null || true; exit 0" SIGINT SIGTERM
wait
