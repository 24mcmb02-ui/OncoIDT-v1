import { useState, useMemo } from 'react'
import {
  ComposedChart,
  Line,
  Area,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceArea,
  ReferenceLine,
} from 'recharts'
import type { CanonicalRecord, Alert, RiskScore, SimulationResult } from '@/types'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface TimelinePoint {
  ts: number // epoch ms
  // Vitals
  temperature?: number
  heart_rate?: number
  respiratory_rate?: number
  spo2?: number
  sbp?: number
  // Labs
  anc?: number
  crp?: number
  // Risk scores
  infection_risk?: number
  infection_risk_lo?: number
  infection_risk_hi?: number
  deterioration_risk?: number
  deterioration_risk_lo?: number
  deterioration_risk_hi?: number
  // Counterfactual overlay
  cf_infection_risk?: number
  cf_deterioration_risk?: number
}

interface AlertMarker {
  ts: number
  priority: string
  message: string
}

interface InterventionMarker {
  ts: number
  label: string
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const RANGE_OPTIONS = [
  { label: '6h', hours: 6 },
  { label: '12h', hours: 12 },
  { label: '24h', hours: 24 },
  { label: '48h', hours: 48 },
  { label: '7d', hours: 168 },
  { label: 'All', hours: Infinity },
]

function alertDotColor(priority: string): string {
  const map: Record<string, string> = {
    Critical: '#dc2626',
    High: '#f97316',
    Medium: '#eab308',
    Low: '#22c55e',
  }
  return map[priority] ?? '#6b7280'
}

function buildTimelinePoints(records: CanonicalRecord[]): TimelinePoint[] {
  const byTs: Record<number, TimelinePoint> = {}

  for (const rec of records) {
    const ts = new Date(rec.timestamp_utc).getTime()
    if (!byTs[ts]) byTs[ts] = { ts }
    const pt = byTs[ts]

    if (rec.record_type === 'vital') {
      const p = rec.payload as { loinc_code?: string; value_numeric?: number }
      const code = p.loinc_code ?? ''
      const val = p.value_numeric
      if (val == null) continue
      if (code === '8310-5') pt.temperature = val
      else if (code === '8867-4') pt.heart_rate = val
      else if (code === '9279-1') pt.respiratory_rate = val
      else if (code === '59408-5') pt.spo2 = val
      else if (code === '8480-6') pt.sbp = val
    } else if (rec.record_type === 'lab') {
      const p = rec.payload as { loinc_code?: string; value_numeric?: number }
      const code = p.loinc_code ?? ''
      const val = p.value_numeric
      if (val == null) continue
      if (code === '26499-4') pt.anc = val
      else if (code === '1988-5') pt.crp = val
    }
  }

  return Object.values(byTs).sort((a, b) => a.ts - b.ts)
}

function mergeRiskScores(
  points: TimelinePoint[],
  riskHistory: Array<{ timestamp: string; score: number; uncertainty_lower: number; uncertainty_upper: number; score_type: string }>,
): TimelinePoint[] {
  const merged = [...points]
  for (const rs of riskHistory) {
    const ts = new Date(rs.timestamp).getTime()
    let pt = merged.find((p) => Math.abs(p.ts - ts) < 30_000)
    if (!pt) {
      pt = { ts }
      merged.push(pt)
    }
    if (rs.score_type === 'infection') {
      pt.infection_risk = rs.score
      pt.infection_risk_lo = rs.uncertainty_lower
      pt.infection_risk_hi = rs.uncertainty_upper
    } else {
      pt.deterioration_risk = rs.score
      pt.deterioration_risk_lo = rs.uncertainty_lower
      pt.deterioration_risk_hi = rs.uncertainty_upper
    }
  }
  return merged.sort((a, b) => a.ts - b.ts)
}

function mergeCounterfactual(
  points: TimelinePoint[],
  sim: SimulationResult,
): TimelinePoint[] {
  const merged = [...points]
  for (const [horizonKey, rs] of Object.entries(sim.counterfactual_scores)) {
    const score = (rs as RiskScore).score
    const ts = new Date((rs as RiskScore).timestamp).getTime()
    let pt = merged.find((p) => Math.abs(p.ts - ts) < 30_000)
    if (!pt) {
      pt = { ts }
      merged.push(pt)
    }
    if (horizonKey.includes('infection') || (rs as RiskScore).forecast_horizon_hours) {
      pt.cf_infection_risk = score
    } else {
      pt.cf_deterioration_risk = score
    }
  }
  return merged.sort((a, b) => a.ts - b.ts)
}

// ---------------------------------------------------------------------------
// Custom tooltip
// ---------------------------------------------------------------------------

function CustomTooltip({ active, payload, label }: {
  active?: boolean
  payload?: Array<{ name: string; value: number; color: string }>
  label?: number
}) {
  if (!active || !payload?.length) return null
  return (
    <div className="rounded-lg border border-gray-200 bg-white p-2 shadow-lg text-xs max-w-xs">
      <p className="mb-1 font-semibold text-gray-700">
        {label ? new Date(label).toLocaleString() : ''}
      </p>
      {payload.map((entry) => (
        <div key={entry.name} className="flex items-center gap-1">
          <span className="inline-block h-2 w-2 rounded-full" style={{ background: entry.color }} />
          <span className="text-gray-600">{entry.name}:</span>
          <span className="font-medium text-gray-900">
            {typeof entry.value === 'number' ? entry.value.toFixed(3) : entry.value}
          </span>
        </div>
      ))}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export interface ClinicalTimelineProps {
  records: CanonicalRecord[]
  riskHistory?: Array<{
    timestamp: string
    score: number
    uncertainty_lower: number
    uncertainty_upper: number
    score_type: string
  }>
  alerts?: Alert[]
  interventionMarkers?: InterventionMarker[]
  simulation?: SimulationResult | null
}

export function ClinicalTimeline({
  records,
  riskHistory = [],
  alerts = [],
  interventionMarkers = [],
  simulation = null,
}: ClinicalTimelineProps) {
  const [rangeHours, setRangeHours] = useState(24)
  const [visibleSeries, setVisibleSeries] = useState<Set<string>>(
    new Set(['temperature', 'anc', 'infection_risk', 'deterioration_risk']),
  )

  const allPoints = useMemo(() => {
    let pts = buildTimelinePoints(records)
    pts = mergeRiskScores(pts, riskHistory)
    if (simulation?.status === 'complete') {
      pts = mergeCounterfactual(pts, simulation)
    }
    return pts
  }, [records, riskHistory, simulation])

  const now = Date.now()
  const cutoff = rangeHours === Infinity ? 0 : now - rangeHours * 3_600_000

  const visiblePoints = useMemo(
    () => allPoints.filter((p) => p.ts >= cutoff),
    [allPoints, cutoff],
  )

  const alertMarkers: AlertMarker[] = useMemo(
    () =>
      alerts
        .filter((a) => new Date(a.created_at).getTime() >= cutoff)
        .map((a) => ({
          ts: new Date(a.created_at).getTime(),
          priority: a.priority,
          message: a.message,
        })),
    [alerts, cutoff],
  )

  function toggleSeries(name: string) {
    setVisibleSeries((prev) => {
      const next = new Set(prev)
      next.has(name) ? next.delete(name) : next.add(name)
      return next
    })
  }

  const show = (name: string) => visibleSeries.has(name)

  const seriesToggleItems = [
    { key: 'temperature', label: 'Temp (°C)', color: '#f97316' },
    { key: 'heart_rate', label: 'HR', color: '#ec4899' },
    { key: 'respiratory_rate', label: 'RR', color: '#8b5cf6' },
    { key: 'spo2', label: 'SpO₂', color: '#06b6d4' },
    { key: 'anc', label: 'ANC', color: '#10b981' },
    { key: 'crp', label: 'CRP', color: '#f59e0b' },
    { key: 'infection_risk', label: 'Inf. Risk', color: '#dc2626' },
    { key: 'deterioration_risk', label: 'Det. Risk', color: '#7c3aed' },
  ]

  return (
    <div className="rounded-lg border border-gray-200 bg-white p-4 shadow-sm">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <h2 className="text-sm font-semibold text-gray-800">Clinical Timeline</h2>

        {/* Time range selector */}
        <div className="flex gap-1" role="group" aria-label="Time range">
          {RANGE_OPTIONS.map((opt) => (
            <button
              key={opt.label}
              onClick={() => setRangeHours(opt.hours)}
              className={`rounded px-2 py-0.5 text-xs font-medium transition-colors ${
                rangeHours === opt.hours
                  ? 'bg-indigo-600 text-white'
                  : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
              }`}
              aria-pressed={rangeHours === opt.hours}
            >
              {opt.label}
            </button>
          ))}
        </div>
      </div>

      {/* Series toggles */}
      <div className="mb-3 flex flex-wrap gap-1.5">
        {seriesToggleItems.map((s) => (
          <button
            key={s.key}
            onClick={() => toggleSeries(s.key)}
            className={`flex items-center gap-1 rounded-full px-2 py-0.5 text-xs transition-opacity ${
              show(s.key) ? 'opacity-100' : 'opacity-40'
            }`}
            style={{ background: `${s.color}22`, color: s.color, border: `1px solid ${s.color}` }}
            aria-pressed={show(s.key)}
          >
            {s.label}
          </button>
        ))}
      </div>

      {visiblePoints.length === 0 ? (
        <div className="flex h-48 items-center justify-center text-sm text-gray-400">
          No data in selected time range.
        </div>
      ) : (
        <ResponsiveContainer width="100%" height={320}>
          <ComposedChart data={visiblePoints} margin={{ top: 4, right: 16, bottom: 4, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
            <XAxis
              dataKey="ts"
              type="number"
              domain={['dataMin', 'dataMax']}
              scale="time"
              tickFormatter={(v) => new Date(v).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
              tick={{ fontSize: 10 }}
            />
            {/* Left axis: vitals */}
            <YAxis yAxisId="vitals" orientation="left" tick={{ fontSize: 10 }} width={32} />
            {/* Right axis: risk [0,1] */}
            <YAxis yAxisId="risk" orientation="right" domain={[0, 1]} tick={{ fontSize: 10 }} width={32} />

            <Tooltip content={<CustomTooltip />} />
            <Legend wrapperStyle={{ fontSize: 11 }} />

            {/* Alert reference lines */}
            {alertMarkers.map((am, i) => (
              <ReferenceLine
                key={i}
                x={am.ts}
                yAxisId="risk"
                stroke={alertDotColor(am.priority)}
                strokeDasharray="4 2"
                label={{ value: am.priority[0], position: 'top', fontSize: 9, fill: alertDotColor(am.priority) }}
              />
            ))}

            {/* Intervention markers */}
            {interventionMarkers.map((im, i) => (
              <ReferenceLine
                key={`iv-${i}`}
                x={im.ts}
                yAxisId="risk"
                stroke="#6366f1"
                strokeDasharray="6 3"
                label={{ value: '↓', position: 'top', fontSize: 11, fill: '#6366f1' }}
              />
            ))}

            {/* Vitals lines */}
            {show('temperature') && (
              <Line yAxisId="vitals" type="monotone" dataKey="temperature" name="Temp (°C)" stroke="#f97316" dot={false} strokeWidth={1.5} connectNulls />
            )}
            {show('heart_rate') && (
              <Line yAxisId="vitals" type="monotone" dataKey="heart_rate" name="HR" stroke="#ec4899" dot={false} strokeWidth={1.5} connectNulls />
            )}
            {show('respiratory_rate') && (
              <Line yAxisId="vitals" type="monotone" dataKey="respiratory_rate" name="RR" stroke="#8b5cf6" dot={false} strokeWidth={1.5} connectNulls />
            )}
            {show('spo2') && (
              <Line yAxisId="vitals" type="monotone" dataKey="spo2" name="SpO₂" stroke="#06b6d4" dot={false} strokeWidth={1.5} connectNulls />
            )}
            {show('anc') && (
              <Line yAxisId="vitals" type="monotone" dataKey="anc" name="ANC" stroke="#10b981" dot={false} strokeWidth={2} connectNulls />
            )}
            {show('crp') && (
              <Line yAxisId="vitals" type="monotone" dataKey="crp" name="CRP" stroke="#f59e0b" dot={false} strokeWidth={1.5} connectNulls />
            )}

            {/* Infection risk area with CI band */}
            {show('infection_risk') && (
              <>
                <Area
                  yAxisId="risk"
                  type="monotone"
                  dataKey="infection_risk_hi"
                  name="_inf_hi"
                  stroke="none"
                  fill="#fca5a5"
                  fillOpacity={0.3}
                  legendType="none"
                  connectNulls
                />
                <Area
                  yAxisId="risk"
                  type="monotone"
                  dataKey="infection_risk_lo"
                  name="_inf_lo"
                  stroke="none"
                  fill="#ffffff"
                  fillOpacity={1}
                  legendType="none"
                  connectNulls
                />
                <Line
                  yAxisId="risk"
                  type="monotone"
                  dataKey="infection_risk"
                  name="Inf. Risk"
                  stroke="#dc2626"
                  strokeWidth={2}
                  dot={false}
                  connectNulls
                />
              </>
            )}

            {/* Deterioration risk */}
            {show('deterioration_risk') && (
              <Line
                yAxisId="risk"
                type="monotone"
                dataKey="deterioration_risk"
                name="Det. Risk"
                stroke="#7c3aed"
                strokeWidth={2}
                dot={false}
                connectNulls
              />
            )}

            {/* Counterfactual overlay */}
            {simulation?.status === 'complete' && (
              <>
                <Line
                  yAxisId="risk"
                  type="monotone"
                  dataKey="cf_infection_risk"
                  name="CF Inf. Risk"
                  stroke="#dc2626"
                  strokeWidth={2}
                  strokeDasharray="6 3"
                  dot={false}
                  connectNulls
                />
                <Line
                  yAxisId="risk"
                  type="monotone"
                  dataKey="cf_deterioration_risk"
                  name="CF Det. Risk"
                  stroke="#7c3aed"
                  strokeWidth={2}
                  strokeDasharray="6 3"
                  dot={false}
                  connectNulls
                />
              </>
            )}
          </ComposedChart>
        </ResponsiveContainer>
      )}

      {simulation?.status === 'complete' && (
        <p className="mt-1 text-xs text-indigo-600">
          Dashed lines show counterfactual trajectory under simulated intervention.
        </p>
      )}
    </div>
  )
}
