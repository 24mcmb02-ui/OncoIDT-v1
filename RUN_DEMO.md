# Running OncoIDT for Demo / Guide Presentation

## What this project does

OncoIDT is a clinical AI platform for oncology wards. It:
- Predicts **infection risk** and **clinical deterioration** for chemotherapy patients
- Shows a real-time **ward heatmap** with per-bed risk scores
- Generates **SHAP-based explanations** ("why is this patient high risk?")
- Lets clinicians run **what-if simulations** (e.g., "what if we give antibiotics?")
- Fires **prioritised alerts** (Critical / High / Medium) to clinical staff

The system uses a **synthetic patient dataset** — no real patient data is needed.

---

## Option 1 — Fully local, no Docker (fastest, works anywhere)

This is the recommended approach for demos. You need Python 3.10+ and Node.js 18+.

### Step 1 — Start the backend (demo API)

```bash
# From the project root
pip install fastapi uvicorn
python demo_api.py
```

The API will be available at: http://localhost:8000  
Swagger docs: http://localhost:8000/docs

### Step 2 — Start the frontend

```bash
cd frontend
npm install
npm run dev
```

The frontend will be available at: http://localhost:5173

> The frontend is already configured to talk to `http://localhost:8000` by default.

---

## Option 2 — GitHub Codespaces (recommended if your laptop is low on RAM)

### Machine type
When creating the Codespace, choose **4-core / 16GB RAM** (not the default 2-core).  
Go to: Codespaces → New Codespace → Configure → Machine type → 4-core

### Step 1 — Start the backend

```bash
pip install fastapi uvicorn
python demo_api.py
```

Codespaces will show a popup "Port 8000 is available" — click **Open in Browser** or go to the **Ports** tab and make port 8000 **Public**.

### Step 2 — Start the frontend

```bash
cd frontend
npm install
npm run dev -- --host 0.0.0.0
```

Go to the **Ports** tab in Codespaces, find port 5173, and make it **Public**.  
Copy the forwarded URL for port 5173 — that's your frontend.

### Step 3 — Point the frontend at the correct API URL

In Codespaces, `localhost` in the browser doesn't work across ports. You need to set the API URL:

```bash
# In the frontend directory, create a .env.local file:
echo "VITE_API_BASE_URL=https://<your-codespace-name>-8000.app.github.dev" > frontend/.env.local
echo "VITE_WS_BASE_URL=wss://<your-codespace-name>-8000.app.github.dev" >> frontend/.env.local
```

Replace `<your-codespace-name>` with your actual Codespace name (visible in the URL bar).  
Then restart the frontend: `npm run dev -- --host 0.0.0.0`

---

## Option 3 — Lightweight Docker (2 containers only)

Use this if you want Docker but can't run the full 18-container stack.

```bash
docker compose -f docker-compose.demo.yml up
```

This starts:
- PostgreSQL (for future full-stack use)
- Redis (for future full-stack use)  
- Demo API on port 8000

Then start the frontend separately:
```bash
cd frontend && npm install && npm run dev
```

---

## Option 4 — Full Docker stack (needs 16GB+ RAM)

Only use this on a machine with 16GB+ RAM (e.g., a large Codespace or cloud VM).

```bash
docker compose up --build
```

This builds and starts all 18 services. First build takes 15–20 minutes.

---

## What you'll see in the demo

| Page | What it shows |
|------|--------------|
| **Ward Overview** | Heatmap of all beds, colour-coded by infection/deterioration risk |
| **Patient Detail** | Individual risk scores (6h/12h/24h/48h), SHAP explanations, clinical timeline |
| **Alert Center** | Prioritised alerts (Critical/High), acknowledge/snooze/escalate actions |
| **Simulation Panel** | What-if: apply intervention → see predicted risk reduction |
| **Admin** | Clinical rules management, model version control |

---

## Troubleshooting

### "CORS error" in browser console
Make sure the API URL in `.env.local` matches exactly (no trailing slash).

### Frontend shows "Failed to fetch"
The demo API isn't running. Check that `python demo_api.py` is still running in another terminal.

### Codespaces port not accessible
Go to the **Ports** tab → right-click the port → **Port Visibility** → **Public**.

### npm install fails
```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
```

### Docker "out of memory" / containers keep restarting
Use `docker-compose.demo.yml` instead of `docker-compose.yml`.  
Or increase Codespace machine size to 8-core / 16GB.
