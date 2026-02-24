#!/usr/bin/env bash
# ============================================================
#  ExplainMyXray — Start API + Frontend (Linux/macOS)
# ============================================================

set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

echo "===================================="
echo "  ExplainMyXray Local Dev Server"
echo "===================================="
echo

# Activate venv if it exists
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "[OK] Virtual environment activated."
else
    echo "[INFO] No venv found — using system Python."
fi

# Check dependencies
python -c "import streamlit, fastapi, uvicorn" 2>/dev/null || {
    echo "[INSTALL] Installing dependencies..."
    pip install -r requirements.txt --quiet
}

echo
echo "Starting API server on http://localhost:8000 ..."
python app/api.py &
API_PID=$!

# Wait for API to be ready
sleep 3

echo "Starting Streamlit frontend on http://localhost:8501 ..."
streamlit run app/frontend.py --server.port 8501 &
UI_PID=$!

echo
echo "===================================="
echo "  Both servers running!"
echo "  API:      http://localhost:8000"
echo "  Frontend: http://localhost:8501"
echo "===================================="
echo
echo "Open http://localhost:8501 in your browser."
echo "Press Ctrl+C to stop both servers."

trap "kill $API_PID $UI_PID 2>/dev/null; exit" INT TERM
wait
