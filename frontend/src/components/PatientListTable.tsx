import { useState, useMemo } from 'react'
import { useNavigate } from 'react-router-dom'
import type { PatientTwin, Alert, AlertPriority, ChemoCyclePhase } from '@/types'

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function getMaxRiskScore(scores: Record<string, { score: number }>): number {
  const values = Object.values(scores)
  return values.length > 0 ? Math.max(...values.map((s) => s.score)) : 0
}

function riskBadgeClass(score: number): string {
  if (score >= 0.7) return 'bg-red-100 text-red-800'
  if (score >= 0.5) return 'bg-orange-100 text-orange-800'
  if (score >= 0.3) return 'bg-yellow-100 text-yellow-800'
  return 'bg-green-100 text-green-800'
}

function priorityBadgeClass(priority: AlertPriority): string {
  const map: Record<AlertPriority, string> = {
    Critical: 'bg-red-600 text-white',
    High: 'bg-orange-500 text-white',
    Medium: 'bg-yellow-400 text-gray-900',
    Low: 'bg-green-500 text-white',
  }
  return map[priority]
}

// ---------------------------------------------------------------------------
// Filter state
// ---------------------------------------------------------------------------

interface Filters {
  riskMin: number
  riskMax: number
  alertPriority: AlertPriority | ''
  bedLocation: string
  chemoPhase: ChemoCyclePhase | ''
}

const DEFAULT_FILTERS: Filters = {
  riskMin: 0,
  riskMax: 1,
  alertPriority: '',
  bedLocation: '',
  chemoPhase: '',
}

// ---------------------------------------------------------------------------
// Sort state
// ---------------------------------------------------------------------------

type SortKey =
  | 'bed'
  | 'patient'
  | 'infection_risk'
  | 'deterioration_risk'
  | 'news2'
  | 'chemo_phase'
  | 'top_alert'
  | 'last_updated'

interface SortState {
  key: SortKey
  dir: 'asc' | 'desc'
}

// ---------------------------------------------------------------------------
// Row type
// ---------------------------------------------------------------------------

interface PatientRow {
  patient: PatientTwin
  infectionRisk: number
  deteriorationRisk: number
  topAlert: Alert | null
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

interface PatientListTableProps {
  patients: PatientTwin[]
  alerts: Alert[]
}

export function PatientListTable({ patients, alerts }: PatientListTableProps) {
  const navigate = useNavigate()
  const [sort, setSort] = useState<SortState>({ key: 'infection_risk', dir: 'desc' })
  const [filters, setFilters] = useState<Filters>(DEFAULT_FILTERS)

  // Build rows with derived fields
  const rows: PatientRow[] = useMemo(() => {
    return patients.map((p) => {
      const infectionRisk = getMaxRiskScore(p.infection_risk_scores)
      const deteriorationRisk = getMaxRiskScore(p.deterioration_risk_scores)
      const patientAlerts = alerts
        .filter((a) => a.patient_id === p.patient_id && a.status === 'active')
        .sort((a, b) => {
          const order: Record<string, number> = { Critical: 0, High: 1, Medium: 2, Low: 3 }
          return (order[a.priority] ?? 4) - (order[b.priority] ?? 4)
        })
      return { patient: p, infectionRisk, deteriorationRisk, topAlert: patientAlerts[0] ?? null }
    })
  }, [patients, alerts])

  // Apply filters
  const filtered = useMemo(() => {
    return rows.filter((r) => {
      const maxRisk = Math.max(r.infectionRisk, r.deteriorationRisk)
      if (maxRisk < filters.riskMin || maxRisk > filters.riskMax) return false
      if (filters.alertPriority && r.topAlert?.priority !== filters.alertPriority) return false
      if (filters.bedLocation && r.patient.bed_id !== filters.bedLocation) return false
      if (filters.chemoPhase && r.patient.chemo_cycle_phase !== filters.chemoPhase) return false
      return true
    })
  }, [rows, filters])

  // Apply sort
  const sorted = useMemo(() => {
    return [...filtered].sort((a, b) => {
      let cmp = 0
      switch (sort.key) {
        case 'bed':
          cmp = (a.patient.bed_id ?? '').localeCompare(b.patient.bed_id ?? '')
          break
        case 'patient':
          cmp = a.patient.mrn.localeCompare(b.patient.mrn)
          break
        case 'infection_risk':
          cmp = a.infectionRisk - b.infectionRisk
          break
        case 'deterioration_risk':
          cmp = a.deteriorationRisk - b.deteriorationRisk
          break
        case 'chemo_phase':
          cmp = a.patient.chemo_cycle_phase.localeCompare(b.patient.chemo_cycle_phase)
          break
        case 'top_alert': {
          const order: Record<string, number> = { Critical: 0, High: 1, Medium: 2, Low: 3 }
          cmp = (order[a.topAlert?.priority ?? ''] ?? 4) - (order[b.topAlert?.priority ?? ''] ?? 4)
          break
        }
        case 'last_updated':
          cmp = new Date(a.patient.last_updated).getTime() - new Date(b.patient.last_updated).getTime()
          break
      }
      return sort.dir === 'asc' ? cmp : -cmp
    })
  }, [filtered, sort])

  function toggleSort(key: SortKey) {
    setSort((prev) =>
      prev.key === key ? { key, dir: prev.dir === 'asc' ? 'desc' : 'asc' } : { key, dir: 'desc' },
    )
  }

  function SortIcon({ col }: { col: SortKey }) {
    if (sort.key !== col) return <span className="ml-1 text-gray-300">↕</span>
    return <span className="ml-1">{sort.dir === 'asc' ? '↑' : '↓'}</span>
  }

  const thClass = 'px-3 py-2 text-left text-xs font-semibold text-gray-600 uppercase tracking-wide cursor-pointer select-none hover:text-gray-900'

  return (
    <div className="flex flex-col gap-3">
      {/* Filter bar */}
      <div className="flex flex-wrap gap-2 rounded-lg bg-gray-50 p-3">
        <div className="flex items-center gap-1">
          <label className="text-xs text-gray-600">Risk</label>
          <input
            type="number"
            min={0}
            max={1}
            step={0.05}
            value={filters.riskMin}
            onChange={(e) => setFilters((f) => ({ ...f, riskMin: parseFloat(e.target.value) }))}
            className="w-16 rounded border border-gray-300 px-1 py-0.5 text-xs"
            aria-label="Minimum risk score"
          />
          <span className="text-xs text-gray-400">–</span>
          <input
            type="number"
            min={0}
            max={1}
            step={0.05}
            value={filters.riskMax}
            onChange={(e) => setFilters((f) => ({ ...f, riskMax: parseFloat(e.target.value) }))}
            className="w-16 rounded border border-gray-300 px-1 py-0.5 text-xs"
            aria-label="Maximum risk score"
          />
        </div>

        <select
          value={filters.alertPriority}
          onChange={(e) => setFilters((f) => ({ ...f, alertPriority: e.target.value as AlertPriority | '' }))}
          className="rounded border border-gray-300 px-2 py-0.5 text-xs"
          aria-label="Filter by alert priority"
        >
          <option value="">All priorities</option>
          <option value="Critical">Critical</option>
          <option value="High">High</option>
          <option value="Medium">Medium</option>
          <option value="Low">Low</option>
        </select>

        <select
          value={filters.chemoPhase}
          onChange={(e) => setFilters((f) => ({ ...f, chemoPhase: e.target.value as ChemoCyclePhase | '' }))}
          className="rounded border border-gray-300 px-2 py-0.5 text-xs"
          aria-label="Filter by chemo cycle phase"
        >
          <option value="">All phases</option>
          <option value="pre">Pre</option>
          <option value="nadir">Nadir</option>
          <option value="recovery">Recovery</option>
          <option value="off">Off</option>
        </select>

        <input
          type="text"
          placeholder="Bed ID"
          value={filters.bedLocation}
          onChange={(e) => setFilters((f) => ({ ...f, bedLocation: e.target.value }))}
          className="w-24 rounded border border-gray-300 px-2 py-0.5 text-xs"
          aria-label="Filter by bed location"
        />

        <button
          onClick={() => setFilters(DEFAULT_FILTERS)}
          className="rounded px-2 py-0.5 text-xs text-gray-500 hover:text-gray-800"
        >
          Reset
        </button>

        <span className="ml-auto text-xs text-gray-500">{sorted.length} patients</span>
      </div>

      {/* Table */}
      <div className="overflow-x-auto rounded-lg border border-gray-200">
        <table className="min-w-full divide-y divide-gray-200 text-sm">
          <thead className="bg-gray-50">
            <tr>
              <th className={thClass} onClick={() => toggleSort('bed')}>
                Bed <SortIcon col="bed" />
              </th>
              <th className={thClass} onClick={() => toggleSort('patient')}>
                Patient <SortIcon col="patient" />
              </th>
              <th className={thClass} onClick={() => toggleSort('infection_risk')}>
                Inf. Risk <SortIcon col="infection_risk" />
              </th>
              <th className={thClass} onClick={() => toggleSort('deterioration_risk')}>
                Det. Risk <SortIcon col="deterioration_risk" />
              </th>
              <th className={thClass}>NEWS2</th>
              <th className={thClass} onClick={() => toggleSort('chemo_phase')}>
                Chemo Phase <SortIcon col="chemo_phase" />
              </th>
              <th className={thClass} onClick={() => toggleSort('top_alert')}>
                Top Alert <SortIcon col="top_alert" />
              </th>
              <th className={thClass} onClick={() => toggleSort('last_updated')}>
                Updated <SortIcon col="last_updated" />
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100 bg-white">
            {sorted.length === 0 && (
              <tr>
                <td colSpan={8} className="py-8 text-center text-gray-400">
                  No patients match the current filters.
                </td>
              </tr>
            )}
            {sorted.map(({ patient, infectionRisk, deteriorationRisk, topAlert }) => (
              <tr
                key={patient.patient_id}
                className="cursor-pointer hover:bg-gray-50"
                onClick={() => navigate(`/patient/${patient.patient_id}`)}
                role="row"
                tabIndex={0}
                onKeyDown={(e) => e.key === 'Enter' && navigate(`/patient/${patient.patient_id}`)}
                aria-label={`Patient ${patient.mrn}`}
              >
                <td className="px-3 py-2 font-mono text-xs text-gray-700">{patient.bed_id ?? '—'}</td>
                <td className="px-3 py-2">
                  <div className="font-medium text-gray-900">{patient.mrn}</div>
                  <div className="text-xs text-gray-500">{patient.age_years}y {patient.sex}</div>
                </td>
                <td className="px-3 py-2">
                  <span className={`rounded-full px-2 py-0.5 text-xs font-semibold ${riskBadgeClass(infectionRisk)}`}>
                    {(infectionRisk * 100).toFixed(1)}%
                  </span>
                </td>
                <td className="px-3 py-2">
                  <span className={`rounded-full px-2 py-0.5 text-xs font-semibold ${riskBadgeClass(deteriorationRisk)}`}>
                    {(deteriorationRisk * 100).toFixed(1)}%
                  </span>
                </td>
                <td className="px-3 py-2 text-gray-700">
                  {/* NEWS2 derived from vitals — placeholder; real value comes from reasoner */}
                  —
                </td>
                <td className="px-3 py-2">
                  <span className="rounded bg-blue-50 px-1.5 py-0.5 text-xs font-medium text-blue-700 capitalize">
                    {patient.chemo_cycle_phase}
                  </span>
                </td>
                <td className="px-3 py-2">
                  {topAlert ? (
                    <span className={`rounded-full px-2 py-0.5 text-xs font-semibold ${priorityBadgeClass(topAlert.priority)}`}>
                      {topAlert.priority}
                    </span>
                  ) : (
                    <span className="text-gray-300">—</span>
                  )}
                </td>
                <td className="px-3 py-2 text-xs text-gray-500">
                  {new Date(patient.last_updated).toLocaleTimeString()}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
