import { useQuery } from '@tanstack/react-query'
import { api } from '@/api/client'
import type { WardExplanation } from '@/types'

interface GlobalExplanationPanelProps {
  wardId: string
}

export function GlobalExplanationPanel({ wardId }: GlobalExplanationPanelProps) {
  const { data, isLoading, isError } = useQuery<WardExplanation>({
    queryKey: ['ward-explanation', wardId],
    queryFn: () => api.get<WardExplanation>(`/explanations/ward/${wardId}/global`),
    refetchInterval: 30_000, // refresh every 30s per Requirement 11.6
  })

  if (isLoading) {
    return (
      <div className="rounded-lg border border-gray-200 bg-white p-4">
        <h3 className="mb-3 text-sm font-semibold text-gray-700">Top Ward Risk Drivers</h3>
        <div className="space-y-2">
          {[...Array(5)].map((_, i) => (
            <div key={i} className="h-4 animate-pulse rounded bg-gray-100" />
          ))}
        </div>
      </div>
    )
  }

  if (isError || !data) {
    return (
      <div className="rounded-lg border border-gray-200 bg-white p-4">
        <h3 className="mb-2 text-sm font-semibold text-gray-700">Top Ward Risk Drivers</h3>
        <p className="text-xs text-gray-400">Explanation data unavailable.</p>
      </div>
    )
  }

  const maxAbs = Math.max(...data.top_features.map((f) => Math.abs(f.shap_value)), 0.001)

  return (
    <div className="rounded-lg border border-gray-200 bg-white p-4">
      <div className="mb-3 flex items-center justify-between">
        <h3 className="text-sm font-semibold text-gray-700">Top Ward Risk Drivers</h3>
        <span className="text-xs text-gray-400">
          Updated {new Date(data.computed_at).toLocaleTimeString()}
        </span>
      </div>

      <ol className="space-y-2" aria-label="Top ward risk drivers">
        {data.top_features.slice(0, 5).map((feat, i) => {
          const barWidth = (Math.abs(feat.shap_value) / maxAbs) * 100
          const isPositive = feat.direction === 'positive'
          return (
            <li key={feat.feature_name} className="flex items-center gap-2">
              <span className="w-4 text-right text-xs font-medium text-gray-400">{i + 1}</span>
              <div className="flex-1">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-medium text-gray-800">{feat.feature_name}</span>
                  <span className={`text-xs font-semibold ${isPositive ? 'text-red-600' : 'text-green-600'}`}>
                    {isPositive ? '+' : '−'}{Math.abs(feat.shap_value).toFixed(3)}
                  </span>
                </div>
                <div className="mt-0.5 h-1.5 w-full rounded-full bg-gray-100">
                  <div
                    className={`h-1.5 rounded-full ${isPositive ? 'bg-red-400' : 'bg-green-400'}`}
                    style={{ width: `${barWidth}%` }}
                    role="progressbar"
                    aria-valuenow={barWidth}
                    aria-valuemin={0}
                    aria-valuemax={100}
                  />
                </div>
              </div>
            </li>
          )
        })}
      </ol>
    </div>
  )
}
