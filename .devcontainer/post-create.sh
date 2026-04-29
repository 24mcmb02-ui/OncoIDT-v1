#!/usr/bin/env bash
# =============================================================================
# OncoIDT — Codespaces post-create setup
# Runs once after the devcontainer is built.
# =============================================================================
set -e

echo "================================================"
echo "  OncoIDT — Codespaces post-create setup"
echo "================================================"

# Install Python dependencies for running scripts/tests outside Docker
echo "[1/3] Installing Python dependencies..."
pip install --quiet \
    fastapi uvicorn pydantic pydantic-settings \
    sqlalchemy asyncpg redis httpx \
    pyjwt pyyaml numpy scikit-learn \
    pytest hypothesis

echo "[2/3] Installing frontend dependencies..."
cd frontend && npm install --silent && cd ..

echo "[3/3] Setup complete."
echo ""
echo "To start the full stack, run:"
echo "  docker compose -f docker-compose.codespaces.yml up --build"
echo ""
echo "To check service health after startup:"
echo "  bash check_services.sh"
