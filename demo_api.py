"""
OncoIDT Demo API — serves data matching the exact TypeScript types in frontend/src/types/index.ts
Run with: python demo_api.py
"""
from __future__ import annotations
import random
import uuid
from datetime import datetime, timezone, timedelta
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json

app = FastAPI(title="OncoIDT Demo API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

rng = random.Random(42)
WARD_ID = "default"
NOW = datetime.now(timezone.utc)

def _ts(minutes_ago: int = 0) -> str:
    return (NOW - timedelta(minutes=minutes_ago)).isoformat()

def _risk_score(score: float, horizon: int) -> dict:
    return {
        "score": round(score, 3),
        "uncertainty_lower": round(max(0, score - 0.08), 3),
        "uncertainty_upper": round(min(1, score + 0.08), 3),
        "forecast_horizon_hours": horizon,
        "model_version": "v1.0",
        "feature_snapshot_id": str(uuid.uuid4()),
        "rule_overrides": [],
        "timestamp": _ts(5),
    }

def _vitals(temp: float, hr: int) -> dict:
    return {
        "temperature_c": temp, "heart_rate_bpm": hr,
        "respiratory_rate_rpm": 20, "sbp_mmhg": 118,
        "dbp_mmhg": 76, "spo2_pct": 97, "gcs": 15,
        "timestamp": _ts(10),
    }

def _labs(anc: float) -> dict:
    return {
        "anc": anc, "wbc": round(anc * 6, 1), "lymphocytes": 0.8,
        "crp_mg_l": 42.0, "procalcitonin_ug_l": 0.8,
        "lactate_mmol_l": 1.2, "creatinine_umol_l": 88.0,
        "timestamp": _ts(30),
    }

PATIENTS = [
    {
        "patient_id": "p-001", "mrn": "MRN-10001", "ward_id": WARD_ID,
        "bed_id": "bed-1", "admission_timestamp": _ts(2880),
        "discharge_timestamp": None, "status": "active",
        "age_years": 58, "sex": "M", "primary_diagnosis_icd10": "C83.3",
        "comorbidities": ["E11", "I10"],
        "chemo_regimen": "R-CHOP", "chemo_cycle_number": 3,
        "chemo_cycle_phase": "nadir", "days_since_last_chemo_dose": 7.0,
        "cumulative_dose_mg_m2": 1200.0, "immunosuppression_score": 0.82,
        "vitals": _vitals(38.6, 98), "labs": _labs(0.3),
        "infection_risk_scores": {
            "6h": _risk_score(0.71, 6), "12h": _risk_score(0.79, 12),
            "24h": _risk_score(0.87, 24), "48h": _risk_score(0.91, 48),
        },
        "deterioration_risk_scores": {
            "6h": _risk_score(0.48, 6), "12h": _risk_score(0.55, 12),
            "24h": _risk_score(0.62, 24),
        },
        "survival_estimate": {
            "median_hours": 18.5, "ci_80_lower_hours": 10.0,
            "ci_80_upper_hours": 32.0, "event_type": "infection",
            "model_version": "v1.0", "timestamp": _ts(5),
        },
        "last_updated": _ts(5), "data_quality_flags": [], "feature_version": "v1",
    },
    {
        "patient_id": "p-002", "mrn": "MRN-10002", "ward_id": WARD_ID,
        "bed_id": "bed-2", "admission_timestamp": _ts(1440),
        "discharge_timestamp": None, "status": "active",
        "age_years": 64, "sex": "F", "primary_diagnosis_icd10": "C50.9",
        "comorbidities": [],
        "chemo_regimen": "BEP", "chemo_cycle_number": 2,
        "chemo_cycle_phase": "recovery", "days_since_last_chemo_dose": 14.0,
        "cumulative_dose_mg_m2": 800.0, "immunosuppression_score": 0.31,
        "vitals": _vitals(37.1, 76), "labs": _labs(1.2),
        "infection_risk_scores": {
            "6h": _risk_score(0.22, 6), "12h": _risk_score(0.31, 12),
            "24h": _risk_score(0.41, 24), "48h": _risk_score(0.45, 48),
        },
        "deterioration_risk_scores": {
            "6h": _risk_score(0.14, 6), "12h": _risk_score(0.21, 12),
            "24h": _risk_score(0.28, 24),
        },
        "survival_estimate": None,
        "last_updated": _ts(8), "data_quality_flags": [], "feature_version": "v1",
    },
    {
        "patient_id": "p-003", "mrn": "MRN-10003", "ward_id": WARD_ID,
        "bed_id": "bed-3", "admission_timestamp": _ts(4320),
        "discharge_timestamp": None, "status": "active",
        "age_years": 71, "sex": "M", "primary_diagnosis_icd10": "C34.1",
        "comorbidities": ["J44", "I10"],
        "chemo_regimen": "FOLFOX", "chemo_cycle_number": 1,
        "chemo_cycle_phase": "nadir", "days_since_last_chemo_dose": 9.0,
        "cumulative_dose_mg_m2": 600.0, "immunosuppression_score": 0.71,
        "vitals": _vitals(38.1, 105), "labs": _labs(0.5),
        "infection_risk_scores": {
            "6h": _risk_score(0.58, 6), "12h": _risk_score(0.65, 12),
            "24h": _risk_score(0.73, 24), "48h": _risk_score(0.78, 48),
        },
        "deterioration_risk_scores": {
            "6h": _risk_score(0.68, 6), "12h": _risk_score(0.75, 12),
            "24h": _risk_score(0.81, 24),
        },
        "survival_estimate": {
            "median_hours": 28.0, "ci_80_lower_hours": 16.0,
            "ci_80_upper_hours": 48.0, "event_type": "deterioration",
            "model_version": "v1.0", "timestamp": _ts(5),
        },
        "last_updated": _ts(3), "data_quality_flags": [], "feature_version": "v1",
    },
    {
        "patient_id": "p-004", "mrn": "MRN-10004", "ward_id": WARD_ID,
        "bed_id": "bed-4", "admission_timestamp": _ts(720),
        "discharge_timestamp": None, "status": "active",
        "age_years": 45, "sex": "F", "primary_diagnosis_icd10": "C91.0",
        "comorbidities": [],
        "chemo_regimen": "R-CHOP", "chemo_cycle_number": 4,
        "chemo_cycle_phase": "pre", "days_since_last_chemo_dose": 21.0,
        "cumulative_dose_mg_m2": 2400.0, "immunosuppression_score": 0.18,
        "vitals": _vitals(36.8, 72), "labs": _labs(2.8),
        "infection_risk_scores": {
            "6h": _risk_score(0.09, 6), "12h": _risk_score(0.12, 12),
            "24h": _risk_score(0.18, 24), "48h": _risk_score(0.21, 48),
        },
        "deterioration_risk_scores": {
            "6h": _risk_score(0.06, 6), "12h": _risk_score(0.09, 12),
            "24h": _risk_score(0.12, 24),
        },
        "survival_estimate": None,
        "last_updated": _ts(12), "data_quality_flags": [], "feature_version": "v1",
    },
]

ALERTS = [
    {
        "alert_id": str(uuid.uuid4()),
        "patient_id": "p-001", "alert_type": "infection_risk",
        "priority": "Critical", "status": "active",
        "message": "Infection risk 87% at 24h. ANC critically low (0.3 × 10⁹/L) + fever (38.6°C). Hard rule override applied.",
        "score_value": 0.87, "score_delta": 0.12, "escalation_count": 1,
        "created_at": _ts(8), "updated_at": _ts(3),
        "recipient_roles": ["physician", "charge_nurse"],
        "top_features": ["ANC (0.3)", "Temperature (38.6°C)", "Days since chemo (7)"],
    },
    {
        "alert_id": str(uuid.uuid4()),
        "patient_id": "p-003", "alert_type": "deterioration_risk",
        "priority": "High", "status": "active",
        "message": "Deterioration risk 81% at 24h. NEWS2 = 7 — hard rule override applied.",
        "score_value": 0.81, "score_delta": 0.08, "escalation_count": 0,
        "created_at": _ts(22), "updated_at": _ts(22),
        "recipient_roles": ["physician", "charge_nurse"],
        "top_features": ["NEWS2 (7)", "Heart Rate (105 bpm)", "ANC (0.5)"],
    },
]

def _feature_attr(name: str, shap: float, value, rank: int, sentence: str) -> dict:
    return {
        "feature_name": name, "shap_value": shap,
        "feature_value": value,
        "direction": "positive" if shap > 0 else "negative",
        "rank": rank, "nl_sentence": sentence,
    }

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "service": "onco-idt-demo-api"}

@app.get("/api/v1/ward/{ward_id}")
def get_ward(ward_id: str):
    return {
        "ward_id": ward_id, "ward_name": "Oncology Ward A",
        "total_beds": 10, "occupied_beds": len(PATIENTS),
        "ward_infection_risk": 0.55, "ward_deterioration_risk": 0.46,
        "high_risk_patient_count": 2,
        "last_updated": _ts(1),
        "environmental": None,
        "active_exposure_events": [],
        "recent_confirmed_infections": ["p-001"],
        "beds": {
            p["bed_id"]: {
                "bed_id": p["bed_id"], "room_id": f"room-{p['bed_id'][-1]}",
                "zone": "A",
                "patient_id": p["patient_id"],
                "infection_risk_score": p["infection_risk_scores"]["24h"]["score"],
                "deterioration_risk_score": p["deterioration_risk_scores"]["24h"]["score"],
                "exposure_flag": p["labs"]["anc"] < 0.5,
                "last_score_timestamp": _ts(5),
            }
            for p in PATIENTS
        },
    }

@app.get("/api/v1/patients")
def list_patients(ward_id: str = WARD_ID):
    return PATIENTS  # frontend expects array directly

@app.get("/api/v1/patients/{patient_id}")
def get_patient(patient_id: str):
    p = next((p for p in PATIENTS if p["patient_id"] == patient_id), PATIENTS[0])
    return p

@app.get("/api/v1/patients/{patient_id}/scores")
def get_scores(patient_id: str):
    p = next((p for p in PATIENTS if p["patient_id"] == patient_id), PATIENTS[0])
    scores = []
    for h, rs in p["infection_risk_scores"].items():
        scores.append({**rs, "score_type": "infection"})
    for h, rs in p["deterioration_risk_scores"].items():
        scores.append({**rs, "score_type": "deterioration"})
    return scores

@app.get("/api/v1/patients/{patient_id}/timeline")
def get_timeline(patient_id: str):
    p = next((p for p in PATIENTS if p["patient_id"] == patient_id), PATIENTS[0])
    base_anc = p["labs"]["anc"]
    base_temp = p["vitals"]["temperature_c"]
    base_inf = p["infection_risk_scores"]["24h"]["score"]
    points = []
    for i in range(48):
        t = NOW - timedelta(hours=48 - i)
        noise = random.gauss(0, 0.01)
        points.append({
            "timestamp": t.isoformat(),
            "temperature_c": round(base_temp - (48 - i) * 0.01 + noise, 1),
            "heart_rate_bpm": round(p["vitals"]["heart_rate_bpm"] + noise * 5),
            "anc": round(max(0.1, base_anc + (i - 48) * 0.005 + noise), 2),
            "infection_risk_24h": round(min(1.0, max(0.0, base_inf - (48 - i) * 0.005 + noise)), 3),
            "deterioration_risk_24h": round(min(1.0, max(0.0, p["deterioration_risk_scores"]["24h"]["score"] - (48 - i) * 0.003 + noise)), 3),
        })
    return points  # flat array — ClinicalTimeline receives as records prop

@app.get("/api/v1/explanations/ward/{ward_id}/global")
def global_explanation(ward_id: str):
    return {
        "ward_id": ward_id,
        "computed_at": _ts(15),
        "top_features": [
            _feature_attr("anc", 0.42, None, 1, "ANC is the top ward-level infection risk driver."),
            _feature_attr("temperature_c", 0.31, None, 2, "Elevated temperature across multiple patients."),
            _feature_attr("days_since_chemo", 0.24, None, 3, "Several patients in nadir phase."),
            _feature_attr("heart_rate_bpm", 0.18, None, 4, "Tachycardia present in high-risk patients."),
            _feature_attr("immunosuppression_score", 0.15, None, 5, "High immunosuppression burden ward-wide."),
        ],
    }

@app.get("/api/v1/explanations/{patient_id}")
def get_explanations_v2(patient_id: str, score_type: str = "infection", horizon: int = 24):
    return _explanation_data(patient_id, score_type, horizon)

@app.get("/api/v1/patients/{patient_id}/explanations")
def get_explanations(patient_id: str, score_type: str = "infection", horizon: int = 24):
    return _explanation_data(patient_id, score_type, horizon)

def _explanation_data(patient_id: str, score_type: str = "infection", horizon: int = 24):
    p = next((p for p in PATIENTS if p["patient_id"] == patient_id), PATIENTS[0])
    anc = p["labs"]["anc"]
    temp = p["vitals"]["temperature_c"]
    # Return single object — RiskScorePanel expects Explanation not Explanation[]
    return {
        "explanation_id": str(uuid.uuid4()),
        "patient_id": patient_id,
        "score_type": score_type,
        "forecast_horizon_hours": horizon,
        "model_version": "v1.0",
        "timestamp": _ts(5),
        "is_rule_triggered": anc < 0.5 and temp > 38.3,
        "rule_ids": ["hard_anc_fever_infection_risk"] if anc < 0.5 and temp > 38.3 else [],
        "top_features": [
            _feature_attr("anc", -0.45, anc, 1, f"ANC of {anc} × 10⁹/L is critically low and is the primary driver of this infection risk score."),
            _feature_attr("temperature_c", 0.32, temp, 2, f"Temperature of {temp}°C indicates fever and is the 2nd driver."),
            _feature_attr("days_since_chemo", 0.21, p["days_since_last_chemo_dose"], 3, f"{p['days_since_last_chemo_dose']} days since last chemo — patient is in nadir phase."),
            _feature_attr("heart_rate_bpm", 0.14, p["vitals"]["heart_rate_bpm"], 4, f"Heart rate of {p['vitals']['heart_rate_bpm']} bpm is elevated."),
            _feature_attr("immunosuppression_score", 0.11, p["immunosuppression_score"], 5, f"Immunosuppression score of {p['immunosuppression_score']:.2f} reflects chemo burden."),
        ],
    }

@app.get("/api/v1/alerts")
def get_alerts(ward_id: str = WARD_ID, patient_id: str = None):
    if patient_id:
        return [a for a in ALERTS if a["patient_id"] == patient_id]
    return ALERTS

@app.post("/api/v1/alerts/{alert_id}/acknowledge")
def acknowledge_alert(alert_id: str):
    for a in ALERTS:
        if a["alert_id"] == alert_id:
            a["status"] = "acknowledged"
    return {"status": "acknowledged", "alert_id": alert_id}

@app.post("/api/v1/alerts/{alert_id}/snooze")
def snooze_alert(alert_id: str, body: dict = {}):
    for a in ALERTS:
        if a["alert_id"] == alert_id:
            a["status"] = "snoozed"
    return {"status": "snoozed", "alert_id": alert_id}

@app.post("/api/v1/alerts/{alert_id}/escalate")
def escalate_alert(alert_id: str):
    for a in ALERTS:
        if a["alert_id"] == alert_id:
            a["escalation_count"] += 1
    return {"status": "escalated", "alert_id": alert_id}

@app.get("/api/v1/explanations/ward/{ward_id}/global")
def global_explanation(ward_id: str):
    return {
        "ward_id": ward_id,
        "computed_at": _ts(15),
        "top_features": [
            _feature_attr("anc", 0.42, None, 1, "ANC is the top ward-level infection risk driver."),
            _feature_attr("temperature_c", 0.31, None, 2, "Elevated temperature across multiple patients."),
            _feature_attr("days_since_chemo", 0.24, None, 3, "Several patients in nadir phase."),
            _feature_attr("heart_rate_bpm", 0.18, None, 4, "Tachycardia present in high-risk patients."),
            _feature_attr("immunosuppression_score", 0.15, None, 5, "High immunosuppression burden ward-wide."),
        ],
    }

# Store simulation results in memory so GET can return them
_sim_store: dict = {}

@app.post("/api/v1/simulations")
async def run_simulation(body: dict):
    patient_id = body.get("patient_id", "p-001")
    p = next((p for p in PATIENTS if p["patient_id"] == patient_id), PATIENTS[0])
    session_id = str(uuid.uuid4())
    result = {
        "session_id": session_id,
        "patient_id": patient_id,
        "status": "complete",
        "baseline_scores": p["infection_risk_scores"],
        "counterfactual_scores": {
            h: {**rs, "score": round(rs["score"] * 0.45, 3),
                "uncertainty_lower": round(rs["uncertainty_lower"] * 0.45, 3),
                "uncertainty_upper": round(rs["uncertainty_upper"] * 0.45, 3)}
            for h, rs in p["infection_risk_scores"].items()
        },
        "delta_explanation": {
            "explanation_id": str(uuid.uuid4()),
            "patient_id": patient_id,
            "score_type": "infection",
            "forecast_horizon_hours": 24,
            "model_version": "v1.0",
            "timestamp": _ts(0),
            "is_rule_triggered": False,
            "rule_ids": [],
            "top_features": [
                _feature_attr("antibiotic_active", -0.38, True, 1, "Antibiotic administration is predicted to reduce infection risk by 55%."),
                _feature_attr("anc", -0.12, p["labs"]["anc"], 2, "ANC contribution reduced after antibiotic treatment."),
                _feature_attr("temperature_c", -0.09, p["vitals"]["temperature_c"], 3, "Fever trajectory projected to improve with treatment."),
            ],
        },
        "created_at": _ts(0),
        "completed_at": _ts(0),
    }
    _sim_store[session_id] = result
    return result

@app.get("/api/v1/simulations/{session_id}")
def get_simulation(session_id: str):
    return _sim_store.get(session_id, {"session_id": session_id, "status": "complete"})

@app.get("/api/v1/admin/rules")
def get_rules():
    return [
        {"rule_id": "hard_anc_fever_infection_risk", "description": "ANC < 0.5 + temp > 38.3°C → infection_risk ≥ 0.85", "enabled": True, "type": "hard"},
        {"rule_id": "hard_sirs_infection_risk", "description": "SIRS ≥ 2 → infection_risk ≥ 0.7", "enabled": True, "type": "hard"},
        {"rule_id": "hard_news2_deterioration", "description": "NEWS2 ≥ 7 → deterioration_risk ≥ 0.8", "enabled": True, "type": "hard"},
    ]

# ---------------------------------------------------------------------------
# WebSocket — live score updates every 5 seconds
# ---------------------------------------------------------------------------

@app.websocket("/ws/v1/ward/{ward_id}")
async def websocket_endpoint(websocket: WebSocket, ward_id: str):
    await websocket.accept()
    try:
        while True:
            p = random.choice(PATIENTS)
            jitter = random.gauss(0, 0.008)
            event = {
                "type": "score_update",
                "patient_id": p["patient_id"],
                "infection_risk_scores": {
                    h: {**rs, "score": round(min(1, max(0, rs["score"] + jitter)), 3)}
                    for h, rs in p["infection_risk_scores"].items()
                },
                "deterioration_risk_scores": {
                    h: {**rs, "score": round(min(1, max(0, rs["score"] + jitter)), 3)}
                    for h, rs in p["deterioration_risk_scores"].items()
                },
                "survival_estimate": p["survival_estimate"],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            await websocket.send_json(event)
            await asyncio.sleep(5)
    except WebSocketDisconnect:
        pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
