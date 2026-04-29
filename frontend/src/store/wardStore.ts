import { create } from 'zustand'
import { immer } from 'zustand/middleware/immer'
import type {
  PatientTwin,
  WardTwin,
  Alert,
  RiskScore,
  SurvivalEstimate,
  WsScoreUpdate,
  WsAlertGenerated,
  WsWardStateChange,
} from '@/types'

// ---------------------------------------------------------------------------
// State shape
// ---------------------------------------------------------------------------

interface WardState {
  // Patients slice — keyed by patient_id
  patients: Record<string, PatientTwin>
  patientsLoading: boolean

  // Alerts slice — ordered newest-first
  alerts: Alert[]
  criticalAlertBanner: Alert | null

  // Ward twin slice
  wardTwin: WardTwin | null
  wardTwinLoading: boolean

  // Connection status
  wsConnected: boolean
}

// ---------------------------------------------------------------------------
// Actions
// ---------------------------------------------------------------------------

interface WardActions {
  // Bulk setters (from REST initial load)
  setPatients: (patients: PatientTwin[]) => void
  setPatientsLoading: (loading: boolean) => void
  setAlerts: (alerts: Alert[]) => void
  setWardTwin: (ward: WardTwin) => void
  setWardTwinLoading: (loading: boolean) => void
  setWsConnected: (connected: boolean) => void

  // WebSocket mutation actions
  applyScoreUpdate: (msg: WsScoreUpdate) => void
  applyAlertGenerated: (msg: WsAlertGenerated) => void
  applyWardStateChange: (msg: WsWardStateChange) => void

  // Alert UI actions
  dismissCriticalBanner: () => void
  acknowledgeAlert: (alertId: string) => void
}

// ---------------------------------------------------------------------------
// Store
// ---------------------------------------------------------------------------

export const useWardStore = create<WardState & WardActions>()(
  immer((set) => ({
    // Initial state
    patients: {},
    patientsLoading: false,
    alerts: [],
    criticalAlertBanner: null,
    wardTwin: null,
    wardTwinLoading: false,
    wsConnected: false,

    // Bulk setters
    setPatients: (patients) =>
      set((state) => {
        state.patients = {}
        for (const p of patients) {
          state.patients[p.patient_id] = p
        }
        state.patientsLoading = false
      }),

    setPatientsLoading: (loading) =>
      set((state) => {
        state.patientsLoading = loading
      }),

    setAlerts: (alerts) =>
      set((state) => {
        state.alerts = alerts
      }),

    setWardTwin: (ward) =>
      set((state) => {
        state.wardTwin = ward
        state.wardTwinLoading = false
      }),

    setWardTwinLoading: (loading) =>
      set((state) => {
        state.wardTwinLoading = loading
      }),

    setWsConnected: (connected) =>
      set((state) => {
        state.wsConnected = connected
      }),

    // WebSocket: score update — mutate only the affected patient row
    applyScoreUpdate: (msg: WsScoreUpdate) =>
      set((state) => {
        const patient = state.patients[msg.patient_id]
        if (!patient) return
        patient.infection_risk_scores = msg.infection_risk_scores as Record<string, RiskScore>
        patient.deterioration_risk_scores = msg.deterioration_risk_scores as Record<string, RiskScore>
        patient.survival_estimate = msg.survival_estimate as SurvivalEstimate | null
        patient.last_updated = msg.timestamp

        // Also update bed state in ward twin if present
        if (state.wardTwin && patient.bed_id) {
          const bed = state.wardTwin.beds[patient.bed_id]
          if (bed) {
            const infScore = Object.values(msg.infection_risk_scores)[0]
            const detScore = Object.values(msg.deterioration_risk_scores)[0]
            bed.infection_risk_score = infScore?.score ?? null
            bed.deterioration_risk_score = detScore?.score ?? null
            bed.last_score_timestamp = msg.timestamp
          }
        }
      }),

    // WebSocket: new alert — prepend to feed, show banner for Critical
    applyAlertGenerated: (msg: WsAlertGenerated) =>
      set((state) => {
        // Deduplicate by alert_id
        const exists = state.alerts.some((a) => a.alert_id === msg.alert.alert_id)
        if (!exists) {
          state.alerts.unshift(msg.alert)
        }
        if (msg.alert.priority === 'Critical') {
          state.criticalAlertBanner = msg.alert
        }
      }),

    // WebSocket: full ward state replacement
    applyWardStateChange: (msg: WsWardStateChange) =>
      set((state) => {
        state.wardTwin = msg.ward
      }),

    dismissCriticalBanner: () =>
      set((state) => {
        state.criticalAlertBanner = null
      }),

    acknowledgeAlert: (alertId: string) =>
      set((state) => {
        const alert = state.alerts.find((a) => a.alert_id === alertId)
        if (alert) {
          alert.status = 'acknowledged'
        }
        if (state.criticalAlertBanner?.alert_id === alertId) {
          state.criticalAlertBanner = null
        }
      }),
  })),
)
