# Running OncoIDT on GitHub Codespaces — Complete Guide

## What you're running and why

OncoIDT is an 18-container microservices system. Your laptop (Ryzen 3, 8GB RAM, no GPU)
cannot run it because:
- Building all Docker images needs ~4GB RAM just for the build process
- Running all containers simultaneously needs 8–12GB RAM
- Neo4j alone uses 512MB heap + 256MB pagecache

GitHub Codespaces gives you a proper Linux VM with 16GB RAM and fast internet,
which is exactly what this project needs.

---

## Step 1 — Create the Codespace with the right machine size

This is the most important step. The default 2-core/8GB machine will OOM-kill containers.

1. Go to your GitHub repository
2. Click the green **Code** button → **Codespaces** tab
3. Click **"..."** (three dots) → **New with options**
4. Set **Machine type** to **4-core / 16GB RAM**
5. Click **Create codespace**

> If you already have a Codespace running on 2-core, delete it and create a new one.
> You cannot change machine type after creation.

The Codespace will open VS Code in the browser. The `.devcontainer/post-create.sh`
script runs automatically and installs Python + Node dependencies.

---

## Step 2 — Open a terminal and start the stack

In the VS Code terminal (Ctrl+` to open):

```bash
# Use the Codespaces-specific compose file (fixes all known issues)
docker compose -f docker-compose.codespaces.yml up --build
```

**What this does:**
- Builds all 11 Python service images (takes 8–12 minutes on first run)
- Starts PostgreSQL + TimescaleDB (stores patient data, risk scores, audit log)
- Starts Redis (event streaming between services)
- Starts Neo4j (patient relationship graph)
- Starts MLflow (model registry)
- Starts all 11 microservices
- Starts the React frontend
- Starts Prometheus + Grafana (monitoring)

**First run is slow** because Docker downloads base images and installs Python packages.
Subsequent runs are fast because Docker caches the layers.

To skip Prometheus and Grafana (saves ~300MB RAM):
```bash
docker compose -f docker-compose.codespaces.yml up --build \
  --scale prometheus=0 --scale grafana=0
```

---

## Step 3 — Wait for services to become healthy

Services start in dependency order. The full stack takes about 2–3 minutes after
the build completes. You'll see logs streaming in the terminal.

Open a **second terminal** (click the + icon in the terminal panel) and run:

```bash
# Check all services at once
bash check_services.sh

# Or wait automatically (retries every 10s for up to 3 minutes)
bash check_services.sh --wait
```

Expected output when everything is healthy:
```
Infrastructure:
  ✓ PostgreSQL  (port 5432)
  ✓ Redis  (port 6379)
  ✓ Neo4j Browser  (http://localhost:7474)
  ✓ MLflow  (http://localhost:5000/)

Application Services:
  ✓ API Gateway  (http://localhost:8000/health)
  ✓ Ingestion Service  (http://localhost:8001/health)
  ✓ Graph Service  (http://localhost:8002/health)
  ✓ Inference Service  (http://localhost:8003/health)
  ✓ Reasoner Service  (http://localhost:8004/health)
  ✓ Alert Service  (http://localhost:8005/health)
  ✓ Explainability Service  (http://localhost:8006/health)
  ✓ Simulation Service  (http://localhost:8007/health)
  ✓ Training Service  (http://localhost:8008/health)
  ✓ Feature Store Service  (http://localhost:8009/health)
  ✓ WebSocket Hub  (http://localhost:8010/health)

Frontend & Monitoring:
  ✓ Frontend  (http://localhost:3000)
  ✓ Prometheus  (http://localhost:9090/-/healthy)
  ✓ Grafana  (http://localhost:3001/api/health)

  Result: 18 passed  0 failed
```

---

## Step 4 — Access the services in your browser

In Codespaces, go to the **Ports** tab (bottom panel, next to Terminal).
You'll see all forwarded ports. For each port you want to open:
- Right-click → **Port Visibility** → **Public** (required for browser access)
- Click the globe icon to open in browser

| Service | Port | What to open |
|---------|------|-------------|
| **Frontend** | 3000 | Main application UI |
| **API Gateway docs** | 8000 | `/api/v1/docs` — Swagger UI for all endpoints |
| **MLflow** | 5000 | Model registry and experiment tracking |
| **Neo4j Browser** | 7474 | Graph database browser |
| **Grafana** | 3001 | Monitoring dashboards (login: admin / oncoidt_grafana) |
| **Prometheus** | 9090 | Raw metrics |

---

## Step 5 — Ingest synthetic patient data

The services are running but the database is empty. Ingest synthetic data:

```bash
# Generate 50 synthetic oncology patients and push them through the pipeline
curl -X POST http://localhost:8001/ingest/synthetic/cohort \
  -H "Content-Type: application/json" \
  -d '{"num_patients": 50, "seed": 42}'
```

This triggers the full pipeline:
1. Ingestion service parses and stores records in PostgreSQL
2. Publishes events to Redis Stream `onco:ingestion:patient_event`
3. Inference service picks up events, runs ML models, publishes scores
4. Reasoner service fuses ML scores with clinical rules
5. Alert service evaluates thresholds, generates alerts
6. WebSocket hub fans out updates to the frontend

---

## Troubleshooting

### "Container keeps restarting" / OOM killed
```bash
# Check which container is failing
docker compose -f docker-compose.codespaces.yml ps

# Check its logs
docker compose -f docker-compose.codespaces.yml logs <service-name>

# Check memory usage
docker stats --no-stream
```
If you see containers using >14GB total, upgrade to 8-core/32GB Codespace.

### Neo4j fails to start
Neo4j takes 30–60 seconds to initialise. Wait and retry:
```bash
docker compose -f docker-compose.codespaces.yml logs neo4j
```
If you see `cypher-shell: command not found` in the healthcheck, it means
the Neo4j image is still starting. It will pass once Neo4j is fully up.

### MLflow healthcheck fails
MLflow v2.13 doesn't have a `/health` endpoint — it serves `/` (root).
The `docker-compose.codespaces.yml` already fixes this. If you're using
the original `docker-compose.yml`, this is the known bug.

### "Port already in use"
Another process is using a port. Find and kill it:
```bash
# Find what's using port 8000
lsof -i :8000
# or
ss -tlnp | grep 8000
```

### Frontend shows blank page / "Failed to fetch"
The API Gateway isn't healthy yet. Check:
```bash
curl http://localhost:8000/health
```
If it returns `{"status": "ok"}`, the frontend should work.
If not, check the gateway logs:
```bash
docker compose -f docker-compose.codespaces.yml logs api-gateway
```

### Build fails with "no space left on device"
```bash
# Clean up unused Docker images and containers
docker system prune -f
docker volume prune -f
```

### Services can't connect to each other
All services are on the `oncoidt` Docker network and use container names
as hostnames (e.g. `postgres`, `redis`, `neo4j`). This is already configured
in the compose file. If you see connection errors, the dependency service
probably hasn't finished starting yet — wait and check its health.

---

## What each service does (quick reference)

| Service | Port | Role |
|---------|------|------|
| PostgreSQL | 5432 | Stores patient records, risk scores, audit log, feature store |
| Redis | 6379 | Event streaming between services (Redis Streams) |
| Neo4j | 7474/7687 | Patient relationship graph (co-location, staff contacts) |
| MLflow | 5000 | ML model registry and experiment tracking |
| Ingestion | 8001 | Accepts FHIR R4, HL7v2, CSV, JSON — parses and stores records |
| Graph | 8002 | Manages Neo4j patient graph, computes graph features |
| Inference | 8003 | Runs Neural CDE + Graph Transformer → risk scores |
| Reasoner | 8004 | Fuses ML scores with clinical rules (SIRS, NEWS2, ANC) |
| Alerts | 8005 | Evaluates thresholds, generates prioritised alerts |
| Explainability | 8006 | Computes SHAP attributions, generates NL explanations |
| Simulation | 8007 | Runs what-if counterfactual simulations |
| Training | 8008 | Trains models on synthetic cohort, logs to MLflow |
| Feature Store | 8009 | Point-in-time correct feature retrieval |
| API Gateway | 8000 | Single entry point — JWT auth, RBAC, rate limiting |
| WebSocket Hub | 8010 | Real-time push to frontend (score updates, alerts) |
| Frontend | 3000 | React dashboard |
| Prometheus | 9090 | Metrics collection |
| Grafana | 3001 | Monitoring dashboards |
