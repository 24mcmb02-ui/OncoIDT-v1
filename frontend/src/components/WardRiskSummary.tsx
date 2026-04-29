import type { WardTwin } from '@/types'

interface WardRiskSummaryProps {
  ward: WardTwin
  wsConnected: boolean
}

function StatCard({
  label,
  value,
  sub,
  color,
}: {
  label: string
  value: string
  sub?: string
  color?: string
}) {
  return (
    <div className="flex flex-col rounded-lg border border-gray-200 bg-white px-4 py-3 shadow-sm">
      <span className="text-xs font-medium uppercase tracking-wide text-gray-500">{label}</span>
      <span className={`mt-1 text-2xl font-bold ${color ?? 'text-gray-900'}`}>{value}</span>
      {sub && <span className="mt-0.5 text-xs text-gray-400">{sub}</span>}
    </div>
  )
}

export function WardRiskSummary({ ward, wsConnected }: WardRiskSummaryProps) {
  const infPct = (ward.ward_infection_risk * 100).toFixed(1)
  const detPct = (ward.ward_deterioration_risk * 100).toFixed(1)
  const occupancyPct = ward.total_beds > 0
    ? ((ward.occupied_beds / ward.total_beds) * 100).toFixed(0)
    : '0'

  const infColor =
    ward.ward_infection_risk >= 0.6
      ? 'text-red-600'
      : ward.ward_infection_risk >= 0.4
        ? 'text-orange-500'
        : 'text-green-600'

  const detColor =
    ward.ward_deterioration_risk >= 0.65
      ? 'text-red-600'
      : ward.ward_deterioration_risk >= 0.4
        ? 'text-orange-500'
        : 'text-green-600'

  return (
    <div className="flex flex-wrap items-center gap-3">
      <div className="flex items-center gap-2">
        <span className="text-lg font-semibold text-gray-900">{ward.ward_name}</span>
        <span
          className={`h-2 w-2 rounded-full ${wsConnected ? 'bg-green-500' : 'bg-gray-400'}`}
          title={wsConnected ? 'Live' : 'Disconnected'}
          aria-label={wsConnected ? 'WebSocket connected' : 'WebSocket disconnected'}
        />
      </div>

      <div className="flex flex-wrap gap-3">
        <StatCard
          label="Ward Inf. Risk"
          value={`${infPct}%`}
          sub="mean across patients"
          color={infColor}
        />
        <StatCard
          label="Ward Det. Risk"
          value={`${detPct}%`}
          sub="mean across patients"
          color={detColor}
        />
        <StatCard
          label="High-Risk Patients"
          value={String(ward.high_risk_patient_count)}
          sub="score > 0.6"
          color={ward.high_risk_patient_count > 0 ? 'text-red-600' : 'text-gray-900'}
        />
        <StatCard
          label="Occupancy"
          value={`${ward.occupied_beds} / ${ward.total_beds}`}
          sub={`${occupancyPct}%`}
        />
        {ward.active_exposure_events.length > 0 && (
          <StatCard
            label="Active Exposures"
            value={String(ward.active_exposure_events.length)}
            color="text-purple-600"
          />
        )}
      </div>
    </div>
  )
}
