import { useEffect } from 'react'
import { useParams } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { api } from '@/api/client'
import { useWardStore } from '@/store/wardStore'
import { useWardWebSocket } from '@/hooks/useWardWebSocket'
import { WardHeatmap } from '@/components/WardHeatmap'
import { PatientListTable } from '@/components/PatientListTable'
import { WardRiskSummary } from '@/components/WardRiskSummary'
import { GlobalExplanationPanel } from '@/components/GlobalExplanationPanel'
import { CriticalAlertBanner } from '@/components/CriticalAlertBanner'
import type { WardTwin, PatientTwin, Alert } from '@/types'

export function WardOverview() {
  const { wardId } = useParams<{ wardId: string }>()
  const id = wardId ?? ''

  // Connect WebSocket for live updates
  useWardWebSocket(id)

  const wsConnected = useWardStore((s) => s.wsConnected)
  const storePatients = useWardStore((s) => s.patients)
  const storeAlerts = useWardStore((s) => s.alerts)
  const storeWardTwin = useWardStore((s) => s.wardTwin)

  const setPatients = useWardStore((s) => s.setPatients)
  const setPatientsLoading = useWardStore((s) => s.setPatientsLoading)
  const setAlerts = useWardStore((s) => s.setAlerts)
  const setWardTwin = useWardStore((s) => s.setWardTwin)
  const setWardTwinLoading = useWardStore((s) => s.setWardTwinLoading)

  // Initial REST load — ward twin
  const { data: wardData, isLoading: wardLoading } = useQuery<WardTwin>({
    queryKey: ['ward', id],
    queryFn: () => api.get<WardTwin>(`/ward/${id}`),
    enabled: !!id,
  })

  // Initial REST load — patients
  const { data: patientsData, isLoading: patientsLoading } = useQuery<PatientTwin[]>({
    queryKey: ['patients', id],
    queryFn: () => api.get<PatientTwin[]>(`/patients?ward_id=${id}`),
    enabled: !!id,
  })

  // Initial REST load — alerts
  const { data: alertsData } = useQuery<Alert[]>({
    queryKey: ['alerts', id],
    queryFn: () => api.get<Alert[]>(`/alerts?ward_id=${id}`),
    enabled: !!id,
  })

  // Sync REST data into Zustand store
  useEffect(() => {
    if (wardData) setWardTwin(wardData)
  }, [wardData, setWardTwin])

  useEffect(() => {
    setPatientsLoading(patientsLoading)
    if (patientsData) setPatients(patientsData)
  }, [patientsData, patientsLoading, setPatients, setPatientsLoading])

  useEffect(() => {
    setWardTwinLoading(wardLoading)
  }, [wardLoading, setWardTwinLoading])

  useEffect(() => {
    if (alertsData) setAlerts(alertsData)
  }, [alertsData, setAlerts])

  const patients = Object.values(storePatients)
  const alerts = storeAlerts

  // Minimal patient map for heatmap tooltip
  const patientMap = Object.fromEntries(
    patients.map((p) => [p.patient_id, { patient_id: p.patient_id, mrn: p.mrn }]),
  )

  if (!id) {
    return (
      <div className="flex h-full items-center justify-center text-gray-400">
        No ward selected.
      </div>
    )
  }

  return (
    <>
      <CriticalAlertBanner />

      <div className="min-h-screen bg-gray-50 p-4 pt-6">
        <div className="mx-auto max-w-screen-2xl space-y-6">

          {/* Ward aggregate stats bar */}
          {storeWardTwin ? (
            <WardRiskSummary ward={storeWardTwin} wsConnected={wsConnected} />
          ) : wardLoading ? (
            <div className="h-16 animate-pulse rounded-lg bg-gray-200" />
          ) : (
            <p className="text-sm text-gray-400">Ward data unavailable.</p>
          )}

          {/* Main content: heatmap + global explanation side-by-side on wide screens */}
          <div className="flex flex-col gap-6 lg:flex-row">
            {/* Heatmap */}
            <div className="flex-1 rounded-lg border border-gray-200 bg-white p-4 shadow-sm">
              <h2 className="mb-3 text-sm font-semibold text-gray-700">Bed Heatmap</h2>
              {storeWardTwin ? (
                <WardHeatmap
                  beds={storeWardTwin.beds}
                  patients={patientMap}
                  alerts={alerts}
                />
              ) : (
                <div className="flex h-40 items-center justify-center text-gray-300">
                  {wardLoading ? 'Loading…' : 'No bed data'}
                </div>
              )}
            </div>

            {/* Global explanation panel */}
            <div className="w-full lg:w-72">
              <GlobalExplanationPanel wardId={id} />
            </div>
          </div>

          {/* Patient list table */}
          <div className="rounded-lg border border-gray-200 bg-white p-4 shadow-sm">
            <h2 className="mb-3 text-sm font-semibold text-gray-700">
              Patients
              {patientsLoading && (
                <span className="ml-2 text-xs font-normal text-gray-400">Loading…</span>
              )}
            </h2>
            <PatientListTable patients={patients} alerts={alerts} />
          </div>

        </div>
      </div>
    </>
  )
}
