import { useState, useEffect } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api } from '@/api/client'
import type { Intervention, InterventionType, SimulationRequest, SimulationResult, Explanation } from '@/types'

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const INTERVENTION_TYPES: { value: InterventionType; label: string }[] = [
  { value: 'antibiotic_administration', label: 'Antibiotic Administration' },
  { value: 'dose_modification', label: 'Dose Modification' },
  { value: 'isolation_measure', label: 'Isolation Measure' },
]

const HORIZONS = [6, 12, 24, 48]

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function emptyIntervention(): Intervention {
  return { type: 'antibiotic_administration', parameter: '', value: '', apply_at_hours: 0 }
}

function deltaSign(delta: number): string {
  return delta > 0 ? `+${(delta * 100).toFixed(1)}%` : `${(delta * 100).toFixed(1)}%`
}

function deltaColor(delta: number): string {
  return delta > 0 ? 'text-red-600' : 'text-green-600'
}

// ---------------------------------------------------------------------------
// Delta explanation sub-component
// ---------------------------------------------------------------------------

function DeltaExplanation({ explanation }: { explanation: Explanation }) {
  const maxAbs = Math.max(...explanation.top_features.map((f) => Math.abs(f.shap_value)), 0.001)
  return (
    <div className="mt-3 rounded-lg border border-indigo-100 bg-indigo-50 p-3">
      <p className="mb-2 text-xs font-semibold text-indigo-800">Delta Explanation</p>
      <div className="space-y-1.5">
        {explanation.top_features.map((feat) => {
          const barWidth = (Math.abs(feat.shap_value) / maxAbs) * 100
          return (
            <div key={feat.feature_name}>
              <div className="flex items-center justify-between text-xs">
                <span className="text-gray-700 capitalize">{feat.feature_name.replace(/_/g, ' ')}</span>
                <span className={feat.direction === 'positive' ? 'text-red-500 font-medium' : 'text-green-600 font-medium'}>
                  {feat.direction === 'positive' ? '+' : '−'}{Math.abs(feat.shap_value).toFixed(3)}
                </span>
              </div>
              <div className="h-1 w-full rounded-full bg-gray-200">
                <div
                  className={`h-1 rounded-full ${feat.direction === 'positive' ? 'bg-red-400' : 'bg-green-400'}`}
                  style={{ width: `${barWidth}%` }}
                />
              </div>
              <p className="text-xs text-gray-500 italic">{feat.nl_sentence}</p>
            </div>
          )
        })}
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Score comparison table
// ---------------------------------------------------------------------------

function ScoreComparison({ result }: { result: SimulationResult }) {
  const horizons = HORIZONS.filter((h) => {
    const key = `${h}h`
    return result.baseline_scores[key] || result.counterfactual_scores[key]
  })

  return (
    <div className="mt-3 overflow-x-auto rounded-lg border border-gray-200">
      <table className="min-w-full text-xs">
        <thead className="bg-gray-50">
          <tr>
            <th className="px-3 py-2 text-left font-semibold text-gray-600">Horizon</th>
            <th className="px-3 py-2 text-right font-semibold text-gray-600">Baseline</th>
            <th className="px-3 py-2 text-right font-semibold text-gray-600">Counterfactual</th>
            <th className="px-3 py-2 text-right font-semibold text-gray-600">Δ</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-100 bg-white">
          {horizons.map((h) => {
            const key = `${h}h`
            const base = result.baseline_scores[key]?.score ?? null
            const cf = result.counterfactual_scores[key]?.score ?? null
            const delta = base != null && cf != null ? cf - base : null
            return (
              <tr key={h}>
                <td className="px-3 py-1.5 text-gray-700">{h}h</td>
                <td className="px-3 py-1.5 text-right text-gray-700">
                  {base != null ? `${(base * 100).toFixed(1)}%` : '—'}
                </td>
                <td className="px-3 py-1.5 text-right text-gray-700">
                  {cf != null ? `${(cf * 100).toFixed(1)}%` : '—'}
                </td>
                <td className={`px-3 py-1.5 text-right font-semibold ${delta != null ? deltaColor(delta) : 'text-gray-400'}`}>
                  {delta != null ? deltaSign(delta) : '—'}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

interface SimulationPanelProps {
  patientId: string
  onSimulationComplete?: (result: SimulationResult) => void
}

export function SimulationPanel({ patientId, onSimulationComplete }: SimulationPanelProps) {
  const qc = useQueryClient()
  const [interventions, setInterventions] = useState<Intervention[]>([emptyIntervention()])
  const [jobId, setJobId] = useState<string | null>(null)
  const [pollEnabled, setPollEnabled] = useState(false)

  // Submit simulation
  const submitMutation = useMutation({
    mutationFn: (req: SimulationRequest) =>
      api.post<{ session_id: string }>('/simulations', req),
    onSuccess: (data) => {
      setJobId(data.session_id)
      setPollEnabled(true)
    },
  })

  // Poll for result
  const { data: simResult } = useQuery<SimulationResult>({
    queryKey: ['simulation', jobId],
    queryFn: () => api.get<SimulationResult>(`/simulations/${jobId}`),
    enabled: pollEnabled && !!jobId,
    refetchInterval: (query) => {
      const status = query.state.data?.status
      if (status === 'complete' || status === 'failed') return false
      return 2000
    },
  })

  // Notify parent when complete
  useEffect(() => {
    if (simResult?.status === 'complete') {
      setPollEnabled(false)
      onSimulationComplete?.(simResult)
    }
  }, [simResult, onSimulationComplete])

  function addRow() {
    setInterventions((prev) => [...prev, emptyIntervention()])
  }

  function removeRow(idx: number) {
    setInterventions((prev) => prev.filter((_, i) => i !== idx))
  }

  function updateRow(idx: number, patch: Partial<Intervention>) {
    setInterventions((prev) => prev.map((iv, i) => (i === idx ? { ...iv, ...patch } : iv)))
  }

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    const valid = interventions.filter((iv) => iv.parameter.trim() !== '')
    if (valid.length === 0) return
    setJobId(null)
    setPollEnabled(false)
    submitMutation.mutate({
      patient_id: patientId,
      interventions: valid,
      horizons: HORIZONS,
    })
  }

  const isRunning = simResult?.status === 'pending' || simResult?.status === 'running' || submitMutation.isPending

  return (
    <div className="rounded-lg border border-gray-200 bg-white p-4 shadow-sm">
      <h2 className="mb-3 text-sm font-semibold text-gray-800">What-If Simulation</h2>

      <form onSubmit={handleSubmit} className="space-y-3">
        {/* Intervention rows */}
        <div className="space-y-2">
          {interventions.map((iv, idx) => (
            <div key={idx} className="flex flex-wrap items-end gap-2 rounded-lg bg-gray-50 p-2">
              {/* Type */}
              <div className="flex flex-col gap-0.5">
                <label className="text-xs text-gray-500">Type</label>
                <select
                  value={iv.type}
                  onChange={(e) => updateRow(idx, { type: e.target.value as InterventionType })}
                  className="rounded border border-gray-300 px-2 py-1 text-xs"
                  aria-label={`Intervention ${idx + 1} type`}
                >
                  {INTERVENTION_TYPES.map((t) => (
                    <option key={t.value} value={t.value}>{t.label}</option>
                  ))}
                </select>
              </div>

              {/* Parameter */}
              <div className="flex flex-col gap-0.5">
                <label className="text-xs text-gray-500">Parameter</label>
                <input
                  type="text"
                  value={iv.parameter}
                  onChange={(e) => updateRow(idx, { parameter: e.target.value })}
                  placeholder="e.g. piperacillin"
                  className="w-32 rounded border border-gray-300 px-2 py-1 text-xs"
                  aria-label={`Intervention ${idx + 1} parameter`}
                />
              </div>

              {/* Value */}
              <div className="flex flex-col gap-0.5">
                <label className="text-xs text-gray-500">Value</label>
                <input
                  type="text"
                  value={String(iv.value)}
                  onChange={(e) => updateRow(idx, { value: e.target.value })}
                  placeholder="e.g. true"
                  className="w-20 rounded border border-gray-300 px-2 py-1 text-xs"
                  aria-label={`Intervention ${idx + 1} value`}
                />
              </div>

              {/* Apply at hours */}
              <div className="flex flex-col gap-0.5">
                <label className="text-xs text-gray-500">At (h)</label>
                <input
                  type="number"
                  min={0}
                  max={48}
                  value={iv.apply_at_hours}
                  onChange={(e) => updateRow(idx, { apply_at_hours: parseFloat(e.target.value) })}
                  className="w-16 rounded border border-gray-300 px-2 py-1 text-xs"
                  aria-label={`Intervention ${idx + 1} apply at hours`}
                />
              </div>

              {interventions.length > 1 && (
                <button
                  type="button"
                  onClick={() => removeRow(idx)}
                  className="mb-0.5 rounded px-1.5 py-1 text-xs text-red-400 hover:text-red-600"
                  aria-label={`Remove intervention ${idx + 1}`}
                >
                  ✕
                </button>
              )}
            </div>
          ))}
        </div>

        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={addRow}
            className="rounded border border-dashed border-gray-300 px-3 py-1 text-xs text-gray-500 hover:border-gray-400 hover:text-gray-700"
          >
            + Add intervention
          </button>

          <button
            type="submit"
            disabled={isRunning}
            className="ml-auto rounded bg-indigo-600 px-4 py-1.5 text-xs font-semibold text-white hover:bg-indigo-700 disabled:opacity-50"
          >
            {isRunning ? 'Running…' : 'Run Simulation'}
          </button>
        </div>

        {submitMutation.isError && (
          <p className="text-xs text-red-500">
            {(submitMutation.error as Error).message}
          </p>
        )}
      </form>

      {/* Results */}
      {simResult && (
        <div className="mt-4">
          {simResult.status === 'pending' || simResult.status === 'running' ? (
            <div className="flex items-center gap-2 text-xs text-gray-500">
              <span className="inline-block h-3 w-3 animate-spin rounded-full border-2 border-indigo-400 border-t-transparent" />
              Simulation running…
            </div>
          ) : simResult.status === 'failed' ? (
            <p className="text-xs text-red-500">Simulation failed. Please try again.</p>
          ) : (
            <>
              <p className="mb-1 text-xs font-semibold text-gray-700">
                Results{' '}
                <span className="font-normal text-gray-400">
                  (completed {simResult.completed_at ? new Date(simResult.completed_at).toLocaleTimeString() : ''})
                </span>
              </p>
              <ScoreComparison result={simResult} />
              {simResult.delta_explanation && (
                <DeltaExplanation explanation={simResult.delta_explanation} />
              )}
            </>
          )}
        </div>
      )}
    </div>
  )
}
