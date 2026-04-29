#!/usr/bin/env bash
# =============================================================================
# OncoIDT — Quick demo starter
# Works on: local machine, GitHub Codespaces, any Linux/Mac environment
# Does NOT require Docker.
#
# Usage:
#   bash start_demo.sh
# =============================================================================

set -e

echo "============================================"
echo "  OncoIDT Demo Starter"
echo "============================================"

# Check Python
if ! command -v python3 &>/dev/null && ! command -v python &>/dev/null; then
  echo "ERROR: Python 3 is required. Install from https://python.org"
  exit 1
fi

PYTHON=$(command -v python3 || command -v python)
echo "Using Python: $($PYTHON --version)"

# Check Node
if ! command -v node &>/dev/null; then
  echo "ERROR: Node.js is required. Install from https://nodejs.org"
  exit 1
fi
echo "Using Node: $(node --version)"

# Install Python deps for demo API
echo ""
echo "[1/3] Installing Python dependencies for demo API..."
$PYTHON -m pip install --quiet fastapi "uvicorn[standard]"

# Install frontend deps
echo ""
echo "[2/3] Installing frontend dependencies..."
cd frontend
npm install --silent
cd ..

echo ""
echo "[3/3] Starting services..."
echo ""

# Start demo API in background
echo "Starting demo API on port 8000..."
$PYTHON demo_api.py &
DEMO_API_PID=$!

# Give it a moment to start
sleep 2

# Check it started
if ! kill -0 $DEMO_API_PID 2>/dev/null; then
  echo "ERROR: Demo API failed to start. Check for port 8000 conflicts."
  exit 1
fi

echo "Demo API running (PID $DEMO_API_PID)"
echo ""

# Start frontend
echo "Starting frontend on port 3000..."
echo ""
echo "============================================"
echo "  Access the app:"
echo "  Frontend:  http://localhost:3000"
echo "  API docs:  http://localhost:8000/docs"
echo ""
echo "  In GitHub Codespaces:"
echo "  Go to the PORTS tab and open port 3000"
echo "============================================"
echo ""
echo "Press Ctrl+C to stop everything."
echo ""

# Trap Ctrl+C to kill both processes
trap "echo ''; echo 'Stopping...'; kill $DEMO_API_PID 2>/dev/null; exit 0" INT TERM

cd frontend
npm run dev -- --host 0.0.0.0 --port 3000
