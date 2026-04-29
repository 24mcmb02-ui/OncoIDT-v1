import { useState } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { api } from '@/api/client'
import { RiskScorePanel } from '@/components/RiskScorePanel'
import { ClinicalTimeline } from '@/components/ClinicalTimeline'
import { SimulationPanel } from '@/components/SimulationPanel'
import type { PatientTwin, Alert, CanonicalRecord, SimulationResult } from '@/types'

// ---------------------------------------------------------------------------
// Patient header
// ---------------------------------------------------------------------------

function PatientHeader({ patient }: { patient: PatientTwin }) {
  const navigate = useNavigate()

  const phaseColors: Record<string, string> = {
    pre: 'bg-blue-100 text-blue-700',
    nadir: 'bg-red-100 text-red-700',
    recovery: 'bg-green-100 text-green-700',
    off: 'bg-gray-100 text-gray-600',
  }

  return (
    <div className="flex flex-wrap items-start justify-between gap-3 rounded-lg border border-gray-200 bg-white p-4 shadow-sm">
      <div className="flex items-center gap-3">
        <button
          onClick={() => navigate(-1)}
          className="rounded p-1 text-gray-400 hover:text-gray-700"
          aria-label="Go back"
        >
          ←
        </button>
        <div>
          <h1 className="text-lg font-bold text-gray-900">
            MRN: {patient.mrn}
          </h1>
          <p className="text-sm text-gray-500">
            {patient.age_years}y {patient.sex} · {patient.primary_diagnosis_icd10}
            {patient.bed_id && ` · Bed ${patient.bed_id}`}
          </p>
        </div>
      </div>

      <div className="flex flex-wrap gap-2 text-xs">
        <span className={`rounded-full px-2 py-0.5 font-medium capitalize ${phaseColors[patient.chemo_cycle_phase] ?? 'bg-gray-100 text-gray-600'}`}>
          {patient.chemo_regimen} · Cycle {patient.chemo_cycle_number} · {patient.chemo_cycle_phase}
        </span>
        <span className="rounded-full bg-gray-100 px-2 py-0.5 text-gray-600">
          Day {Math.round(patient.days_since_last_chemo_dose)} post-dose
        </span>
        <span className="rounded-full bg-purple-50 px-2 py-0.5 text-purple-700">
          Immunosuppression: {(patient.immunosuppression_score * 100).toFixed(0)}%
        </span>
        <span className={`rounded-full px-2 py-0.5 font-medium ${patient.status === 'active' ? 'bg-green-100 text-green-700' : 'bg-gray-100 text-gray-500'}`}>
          {patient.status}
        </span>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Alert history sub-component
// ---------------------------------------------------------------------------

function AlertHistory({ alerts }: { alerts: Alert[] }) {
  const priorityColors: Record<string, string> = {
    Critical: 'bg-red-100 text-red-800',
    High: 'bg-orange-100 text-orange-800',
    Medium: 'bg-yellow-100 text-yellow-800',
    Low: 'bg-green-100 text-green-800',
  }

  const statusColors: Record<string, string> = {
    active: 'text-red-600',
    acknowledged: 'text-gray-500',
    snoozed: 'text-yellow-600',
    resolved: 'text-green-600',
  }

  return (
    <div className="rounded-lg border border-gray-200 bg-white p-4 shadow-sm">
      <h2 className="mb-3 text-sm font-semibold text-gray-800">Alert History</h2>
      {alerts.length === 0 ? (
        <p className="text-xs text-gray-400">No alerts for this patient.</p>
      ) : (
        <div className="space-y-2 max-h-64 overflow-y-auto">
          {alerts.map((alert) => (
            <div key={alert.alert_id} className="flex items-start gap-2 rounded-lg bg-gray-50 p-2">
              <span className={`mt-0.5 rounded-full px-1.5 py-0.5 text-xs font-semibold ${priorityColors[alert.priority] ?? ''}`}>
                {alert.priority}
              </span>
              <div className="flex-1 min-w-0">
                <p className="text-xs text-gray-800">{alert.message}</p>
                {alert.top_features.length > 0 && (
                  <p className="text-xs text-gray-500 mt-0.5">
                    Drivers: {alert.top_features.join(', ')}
                  </p>
                )}
              </div>
              <div className="text-right shrink-0">
                <p className={`text-xs font-medium ${statusColors[alert.status] ?? ''}`}>{alert.status}</p>
                <p className="text-xs text-gray-400">{new Date(alert.created_at).toLocaleString()}</p>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Vitals summary strip
// ---------------------------------------------------------------------------

function VitalsSummary({ patient }: { patient: PatientTwin }) {
  const v = patient.vitals
  const l = patient.labs

  const items = [
    { label: 'Temp', value: v.temperature_c != null ? `${v.temperature_c.toFixed(1)}°C` : '—', warn: v.temperature_c != null && v.temperature_c > 38.3 },
    { label: 'HR', value: v.heart_rate_bpm != null ? `${Math.round(v.heart_rate_bpm)} bpm` : '—', warn: false },
    { label: 'RR', value: v.respiratory_rate_rpm != null ? `${Math.round(v.respiratory_rate_rpm)} rpm` : '—', warn: false },
    { label: 'SpO₂', value: v.spo2_pct != null ? `${v.spo2_pct.toFixed(1)}%` : '—', warn: v.spo2_pct != null && v.spo2_pct < 94 },
    { label: 'BP', value: v.sbp_mmhg != null && v.dbp_mmhg != null ? `${Math.round(v.sbp_mmhg)}/${Math.round(v.dbp_mmhg)}` : '—', warn: false },
    { label: 'GCS', value: v.gcs != null ? String(v.gcs) : '—', warn: v.gcs != null && v.gcs < 14 },
    { label: 'ANC', value: l.anc != null ? `${l.anc.toFixed(2)} ×10⁹/L` : '—', warn: l.anc != null && l.anc < 0.5 },
    { label: 'CRP', value: l.crp_mg_l != null ? `${l.crp_mg_l.toFixed(1)} mg/L` : '—', warn: l.crp_mg_l != null && l.crp_mg_l > 50 },
    { label: 'PCT', value: l.procalcitonin_ug_l != null ? `${l.procalcitonin_ug_l.toFixed(2)} μg/L` : '—', warn: l.procalcitonin_ug_l != null && l.procalcitonin_ug_l > 0.5 },
  ]

  return (
    <div className="flex flex-wrap gap-2 rounded-lg border border-gray-200 bg-white p-3 shadow-sm">
      {items.map((item) => (
        <div key={item.label} className={`flex flex-col items-center rounded px-2 py-1 ${item.warn ? 'bg-red-50' : 'bg-gray-50'}`}>
          <span className="text-xs text-gray-500">{item.label}</span>
          <span className={`text-sm font-semibold ${item.warn ? 'text-red-600' : 'text-gray-800'}`}>
            {item.value}
          </span>
        </div>
      ))}
      <div className="ml-auto self-center text-xs text-gray-400">
        Updated {new Date(v.timestamp).toLocaleTimeString()}
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Main page
// ---------------------------------------------------------------------------

export function PatientDetail() {
  const { patientId } = useParams<{ patientId: string }>()
  const id = patientId ?? ''
  const [activeSimulation, setActiveSimulation] = useState<SimulationResult | null>(null)

  const { data: patient, isLoading: patientLoading, error: patientError } = useQuery<PatientTwin>({
    queryKey: ['patient', id],
    queryFn: () => api.get<PatientTwin>(`/patients/${id}`),
    enabled: !!id,
    refetchInterval: 30_000,
    staleTime: 0,
  })

  const { data: alerts = [] } = useQuery<Alert[]>({
    queryKey: ['patient-alerts', id],
    queryFn: () => api.get<Alert[]>(`/alerts?patient_id=${id}`),
    enabled: !!id,
    refetchInterval: 30_000,
    staleTime: 0,
  })

  const { data: timeline = [] } = useQuery<CanonicalRecord[]>({
    queryKey: ['patient-timeline', id],
    queryFn: () => api.get<CanonicalRecord[]>(`/patients/${id}/timeline`),
    enabled: !!id,
    staleTime: 60_000,
  })

  const { data: riskHistory = [] } = useQuery<Array<{
    timestamp: string
    score: number
    uncertainty_lower: number
    uncertainty_upper: number
    score_type: string
  }>>({
    queryKey: ['patient-scores', id],
    queryFn: () => api.get(`/patients/${id}/scores`),
    enabled: !!id,
    staleTime: 30_000,
  })

  if (patientLoading) {
    return (
      <div className="min-h-screen bg-gray-50 p-4">
        <div className="mx-auto max-w-screen-xl space-y-4">
          {[...Array(4)].map((_, i) => (
            <div key={i} className="h-24 animate-pulse rounded-lg bg-gray-200" />
          ))}
        </div>
      </div>
    )
  }

  if (patientError || !patient) {
    return (
      <div className="flex min-h-screen items-center justify-center text-gray-400">
        Patient not found or failed to load.
      </div>
    )
  }

  const activeAlerts = alerts.filter((a) => a.status === 'active')
  const criticalActive = activeAlerts.some((a) => a.priority === 'Critical')

  return (
    <div className="min-h-screen bg-gray-50 p-4">
      <div className="mx-auto max-w-screen-xl space-y-4">

        {/* Critical alert banner for this patient */}
        {criticalActive && (
          <div
            role="alert"
            className="flex items-center gap-3 rounded-lg bg-red-600 px-4 py-3 text-white shadow"
          >
            <span className="animate-pulse text-xl" aria-hidden="true">🚨</span>
            <span className="font-semibold">Critical alert active for this patient</span>
          </div>
        )}

        {/* Header */}
        <PatientHeader patient={patient} />

        {/* Vitals strip */}
        <VitalsSummary patient={patient} />

        {/* Main content: risk panel + timeline */}
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
          {/* Risk scores + SHAP */}
          <div className="lg:col-span-1">
            <RiskScorePanel patient={patient} />
          </div>

          {/* Timeline */}
          <div className="lg:col-span-2">
            <ClinicalTimeline
              records={timeline}
              riskHistory={riskHistory}
              alerts={alerts}
              simulation={activeSimulation}
            />
          </div>
        </div>

        {/* Alert history + simulation side by side */}
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          <AlertHistory alerts={alerts} />
          <SimulationPanel
            patientId={id}
            onSimulationComplete={(result) => setActiveSimulation(result)}
          />
        </div>

      </div>
    </div>
  )
}
