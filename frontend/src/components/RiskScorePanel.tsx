import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { api } from '@/api/client'
import type { PatientTwin, Explanation, RiskScore } from '@/types'

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const HORIZONS = [6, 12, 24, 48] as const
type Horizon = (typeof HORIZONS)[number]

function riskColor(score: number): string {
  if (score >= 0.7) return 'text-red-600'
  if (score >= 0.5) return 'text-orange-500'
  if (score >= 0.3) return 'text-yellow-500'
  return 'text-green-600'
}

function riskBarColor(score: number): string {
  if (score >= 0.7) return 'bg-red-500'
  if (score >= 0.5) return 'bg-orange-400'
  if (score >= 0.3) return 'bg-yellow-400'
  return 'bg-green-500'
}

function shapBarColor(direction: 'positive' | 'negative'): string {
  return direction === 'positive' ? 'bg-red-400' : 'bg-blue-400'
}

function getRiskScore(scores: Record<string, RiskScore>, horizon: number): RiskScore | null {
  return scores[`${horizon}h`] ?? scores[String(horizon)] ?? null
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

interface ScoreGaugeProps {
  label: string
  score: RiskScore | null
}

function ScoreGauge({ label, score }: ScoreGaugeProps) {
  if (!score) {
    return (
      <div className="flex flex-col gap-1">
        <span className="text-xs font-medium text-gray-500">{label}</span>
        <span className="text-2xl font-bold text-gray-300">—</span>
      </div>
    )
  }

  const pct = Math.round(score.score * 100)
  const loPct = Math.round(score.uncertainty_lower * 100)
  const hiPct = Math.round(score.uncertainty_upper * 100)

  return (
    <div className="flex flex-col gap-1">
      <span className="text-xs font-medium text-gray-500">{label}</span>
      <span className={`text-2xl font-bold ${riskColor(score.score)}`}>{pct}%</span>
      <div className="relative h-2 w-full rounded-full bg-gray-200" aria-label={`${label} risk bar`}>
        {/* Uncertainty band */}
        <div
          className="absolute h-2 rounded-full bg-gray-300 opacity-50"
          style={{ left: `${loPct}%`, width: `${hiPct - loPct}%` }}
        />
        {/* Point estimate */}
        <div
          className={`absolute h-2 rounded-full ${riskBarColor(score.score)}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="text-xs text-gray-400">
        CI: {loPct}%–{hiPct}%
      </span>
      {score.rule_overrides.length > 0 && (
        <span className="mt-0.5 inline-flex items-center gap-1 rounded bg-amber-50 px-1.5 py-0.5 text-xs text-amber-700">
          ⚠ Rule override: {score.rule_overrides.map((r) => r.rule_id).join(', ')}
        </span>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// SHAP explanation panel
// ---------------------------------------------------------------------------

interface ShapPanelProps {
  explanation: Explanation | null
  isLoading: boolean
}

function ShapPanel({ explanation, isLoading }: ShapPanelProps) {
  if (isLoading) {
    return (
      <div className="space-y-2">
        {[...Array(5)].map((_, i) => (
          <div key={i} className="h-8 animate-pulse rounded bg-gray-100" />
        ))}
      </div>
    )
  }

  if (!explanation || explanation.top_features.length === 0) {
    return <p className="text-xs text-gray-400">No explanation available.</p>
  }

  const maxAbs = Math.max(...explanation.top_features.map((f) => Math.abs(f.shap_value)))

  return (
    <div className="space-y-2">
      {explanation.is_rule_triggered && (
        <div className="rounded bg-amber-50 px-2 py-1 text-xs text-amber-700">
          Rule-triggered: {explanation.rule_ids.join(', ')}
        </div>
      )}
      {explanation.top_features.map((feat) => {
        const barWidth = maxAbs > 0 ? (Math.abs(feat.shap_value) / maxAbs) * 100 : 0
        return (
          <div key={feat.feature_name} className="flex flex-col gap-0.5">
            <div className="flex items-center justify-between">
              <span className="text-xs font-medium text-gray-700 capitalize">
                {feat.feature_name.replace(/_/g, ' ')}
              </span>
              <span className={`text-xs font-semibold ${feat.direction === 'positive' ? 'text-red-500' : 'text-blue-500'}`}>
                {feat.direction === 'positive' ? '+' : '−'}{Math.abs(feat.shap_value).toFixed(3)}
              </span>
            </div>
            <div className="h-1.5 w-full rounded-full bg-gray-100">
              <div
                className={`h-1.5 rounded-full ${shapBarColor(feat.direction)}`}
                style={{ width: `${barWidth}%` }}
              />
            </div>
            <p className="text-xs text-gray-500 italic">{feat.nl_sentence}</p>
          </div>
        )
      })}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

interface RiskScorePanelProps {
  patient: PatientTwin
}

export function RiskScorePanel({ patient }: RiskScorePanelProps) {
  const [horizon, setHorizon] = useState<Horizon>(24)
  const [scoreType, setScoreType] = useState<'infection' | 'deterioration'>('infection')

  const infScore = getRiskScore(patient.infection_risk_scores, horizon)
  const detScore = getRiskScore(patient.deterioration_risk_scores, horizon)

  const { data: explanation, isLoading: explLoading } = useQuery<Explanation>({
    queryKey: ['explanation', patient.patient_id, scoreType, horizon],
    queryFn: () =>
      api.get<Explanation>(
        `/explanations/${patient.patient_id}?score_type=${scoreType}&horizon=${horizon}`,
      ),
    staleTime: 60_000,
  })

  return (
    <div className="rounded-lg border border-gray-200 bg-white p-4 shadow-sm">
      <div className="mb-3 flex items-center justify-between">
        <h2 className="text-sm font-semibold text-gray-800">Risk Scores</h2>
        {/* Horizon selector */}
        <div className="flex gap-1" role="group" aria-label="Forecast horizon">
          {HORIZONS.map((h) => (
            <button
              key={h}
              onClick={() => setHorizon(h)}
              className={`rounded px-2 py-0.5 text-xs font-medium transition-colors ${
                horizon === h
                  ? 'bg-indigo-600 text-white'
                  : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
              }`}
              aria-pressed={horizon === h}
            >
              {h}h
            </button>
          ))}
        </div>
      </div>

      {/* Score gauges */}
      <div className="mb-4 grid grid-cols-2 gap-4">
        <ScoreGauge label="Infection Risk" score={infScore} />
        <ScoreGauge label="Deterioration Risk" score={detScore} />
      </div>

      {/* Survival estimate */}
      {patient.survival_estimate && (
        <div className="mb-4 rounded bg-gray-50 px-3 py-2 text-xs text-gray-600">
          <span className="font-medium">Survival estimate ({patient.survival_estimate.event_type}):</span>{' '}
          median {patient.survival_estimate.median_hours.toFixed(1)}h
          <span className="text-gray-400">
            {' '}(80% CI: {patient.survival_estimate.ci_80_lower_hours.toFixed(1)}–
            {patient.survival_estimate.ci_80_upper_hours.toFixed(1)}h)
          </span>
        </div>
      )}

      {/* SHAP explanation */}
      <div>
        <div className="mb-2 flex items-center gap-2">
          <span className="text-xs font-semibold text-gray-700">Top features</span>
          <div className="flex gap-1" role="group" aria-label="Score type for explanation">
            {(['infection', 'deterioration'] as const).map((t) => (
              <button
                key={t}
                onClick={() => setScoreType(t)}
                className={`rounded px-2 py-0.5 text-xs capitalize transition-colors ${
                  scoreType === t
                    ? 'bg-gray-800 text-white'
                    : 'bg-gray-100 text-gray-500 hover:bg-gray-200'
                }`}
                aria-pressed={scoreType === t}
              >
                {t}
              </button>
            ))}
          </div>
        </div>
        <ShapPanel explanation={explanation ?? null} isLoading={explLoading} />
      </div>
    </div>
  )
}
