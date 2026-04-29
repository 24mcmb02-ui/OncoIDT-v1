import { useState, useEffect, useRef, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api } from '@/api/client'
import type { Alert, AlertPriority, AlertStatus } from '@/types'

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const PRIORITY_ORDER: Record<AlertPriority, number> = {
  Critical: 0,
  High: 1,
  Medium: 2,
  Low: 3,
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

function statusBadgeClass(status: AlertStatus): string {
  const map: Record<AlertStatus, string> = {
    active: 'bg-red-100 text-red-700',
    acknowledged: 'bg-gray-100 text-gray-600',
    snoozed: 'bg-yellow-100 text-yellow-700',
    resolved: 'bg-green-100 text-green-700',
  }
  return map[status]
}

// ---------------------------------------------------------------------------
// Web Audio API — short beep for critical alerts
// ---------------------------------------------------------------------------

function playAlertBeep() {
  try {
    const ctx = new AudioContext()
    const osc = ctx.createOscillator()
    const gain = ctx.createGain()
    osc.connect(gain)
    gain.connect(ctx.destination)
    osc.type = 'sine'
    osc.frequency.setValueAtTime(880, ctx.currentTime)
    gain.gain.setValueAtTime(0.3, ctx.currentTime)
    gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.6)
    osc.start(ctx.currentTime)
    osc.stop(ctx.currentTime + 0.6)
  } catch {
    // AudioContext may be blocked before user interaction — silently ignore
  }
}

// ---------------------------------------------------------------------------
// Snooze modal
// ---------------------------------------------------------------------------

interface SnoozeModalProps {
  alertId: string
  onClose: () => void
  onConfirm: (alertId: string, reason: string, minutes: number) => void
}

function SnoozeModal({ alertId, onClose, onConfirm }: SnoozeModalProps) {
  const [reason, setReason] = useState('')
  const [minutes, setMinutes] = useState(30)

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label="Snooze alert"
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/40"
    >
      <div className="w-80 rounded-xl bg-white p-5 shadow-xl">
        <h3 className="mb-3 text-sm font-semibold text-gray-800">Snooze Alert</h3>
        <div className="space-y-3">
          <div>
            <label className="mb-1 block text-xs text-gray-600" htmlFor="snooze-reason">
              Reason (required)
            </label>
            <textarea
              id="snooze-reason"
              value={reason}
              onChange={(e) => setReason(e.target.value)}
              rows={2}
              className="w-full rounded border border-gray-300 px-2 py-1 text-xs"
              placeholder="e.g. Patient already reviewed, monitoring"
            />
          </div>
          <div>
            <label className="mb-1 block text-xs text-gray-600" htmlFor="snooze-duration">
              Duration (minutes)
            </label>
            <select
              id="snooze-duration"
              value={minutes}
              onChange={(e) => setMinutes(Number(e.target.value))}
              className="w-full rounded border border-gray-300 px-2 py-1 text-xs"
            >
              {[15, 30, 60, 120, 240].map((m) => (
                <option key={m} value={m}>{m} min</option>
              ))}
            </select>
          </div>
        </div>
        <div className="mt-4 flex justify-end gap-2">
          <button
            onClick={onClose}
            className="rounded px-3 py-1.5 text-xs text-gray-600 hover:text-gray-900"
          >
            Cancel
          </button>
          <button
            onClick={() => {
              if (!reason.trim()) return
              onConfirm(alertId, reason, minutes)
              onClose()
            }}
            disabled={!reason.trim()}
            className="rounded bg-yellow-500 px-3 py-1.5 text-xs font-semibold text-white hover:bg-yellow-600 disabled:opacity-50"
          >
            Snooze
          </button>
        </div>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Alert row
// ---------------------------------------------------------------------------

interface AlertRowProps {
  alert: Alert
  onAcknowledge: (id: string) => void
  onSnooze: (id: string) => void
  onEscalate: (id: string) => void
}

function AlertRow({ alert, onAcknowledge, onSnooze, onEscalate }: AlertRowProps) {
  const navigate = useNavigate()

  return (
    <div
      className={`flex flex-wrap items-start gap-3 rounded-lg border p-3 transition-colors ${
        alert.priority === 'Critical' && alert.status === 'active'
          ? 'border-red-300 bg-red-50'
          : 'border-gray-200 bg-white'
      }`}
    >
      {/* Priority + status badges */}
      <div className="flex flex-col gap-1 pt-0.5">
        <span className={`rounded-full px-2 py-0.5 text-xs font-semibold ${priorityBadgeClass(alert.priority)}`}>
          {alert.priority}
        </span>
        <span className={`rounded-full px-2 py-0.5 text-xs ${statusBadgeClass(alert.status)}`}>
          {alert.status}
        </span>
      </div>

      {/* Content */}
      <div className="flex-1 min-w-0">
        <p className="text-sm font-medium text-gray-900">{alert.message}</p>
        {alert.top_features.length > 0 && (
          <p className="mt-0.5 text-xs text-gray-500">
            Drivers: {alert.top_features.join(', ')}
          </p>
        )}
        <div className="mt-1 flex flex-wrap gap-2 text-xs text-gray-400">
          <span>{new Date(alert.created_at).toLocaleString()}</span>
          {alert.escalation_count > 1 && (
            <span className="text-orange-500">Escalated ×{alert.escalation_count}</span>
          )}
          {alert.score_value != null && (
            <span>Score: {(alert.score_value * 100).toFixed(1)}%</span>
          )}
          {alert.score_delta != null && (
            <span className={alert.score_delta > 0 ? 'text-red-500' : 'text-green-500'}>
              Δ {alert.score_delta > 0 ? '+' : ''}{(alert.score_delta * 100).toFixed(1)}%
            </span>
          )}
        </div>
      </div>

      {/* Actions */}
      <div className="flex flex-wrap items-center gap-1.5 shrink-0">
        <button
          onClick={() => navigate(`/patient/${alert.patient_id}`)}
          className="rounded border border-gray-300 px-2 py-1 text-xs text-gray-600 hover:bg-gray-50"
        >
          View patient
        </button>
        {alert.status === 'active' && (
          <>
            <button
              onClick={() => onAcknowledge(alert.alert_id)}
              className="rounded bg-green-600 px-2 py-1 text-xs font-medium text-white hover:bg-green-700"
            >
              Acknowledge
            </button>
            <button
              onClick={() => onSnooze(alert.alert_id)}
              className="rounded bg-yellow-500 px-2 py-1 text-xs font-medium text-white hover:bg-yellow-600"
            >
              Snooze
            </button>
            <button
              onClick={() => onEscalate(alert.alert_id)}
              className="rounded bg-red-600 px-2 py-1 text-xs font-medium text-white hover:bg-red-700"
            >
              Escalate
            </button>
          </>
        )}
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Critical alert persistent banner
// ---------------------------------------------------------------------------

interface CriticalBannerProps {
  alerts: Alert[]
  onAcknowledge: (id: string) => void
}

function CriticalBanner({ alerts, onAcknowledge }: CriticalBannerProps) {
  const critical = alerts.filter((a) => a.priority === 'Critical' && a.status === 'active')
  if (critical.length === 0) return null

  const top = critical[0]

  return (
    <div
      role="alertdialog"
      aria-modal="false"
      aria-label="Critical alert banner"
      className="sticky top-0 z-40 flex items-center justify-between gap-4 bg-red-600 px-4 py-3 text-white shadow-lg"
    >
      <div className="flex items-center gap-3">
        <span className="animate-pulse text-xl" aria-hidden="true">🚨</span>
        <div>
          <span className="font-bold">CRITICAL ALERT</span>
          <span className="ml-2 text-sm">{top.message}</span>
          {critical.length > 1 && (
            <span className="ml-2 text-sm opacity-80">+{critical.length - 1} more</span>
          )}
        </div>
      </div>
      <button
        onClick={() => onAcknowledge(top.alert_id)}
        className="rounded bg-white px-3 py-1 text-sm font-semibold text-red-700 hover:bg-red-50 focus:outline-none focus:ring-2 focus:ring-white"
        aria-label="Acknowledge critical alert"
      >
        Acknowledge
      </button>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Main page
// ---------------------------------------------------------------------------

export function AlertCenter() {
  const qc = useQueryClient()
  const [filterPriority, setFilterPriority] = useState<AlertPriority | ''>('')
  const [filterStatus, setFilterStatus] = useState<AlertStatus | 'active'>('active')
  const [snoozeTarget, setSnoozeTarget] = useState<string | null>(null)
  const prevCriticalCount = useRef(0)

  const { data: alerts = [], isLoading } = useQuery<Alert[]>({
    queryKey: ['alerts-center'],
    queryFn: () => api.get<Alert[]>('/alerts'),
    refetchInterval: 15_000,
  })

  // Play beep when new critical alerts arrive
  useEffect(() => {
    const criticalActive = alerts.filter((a) => a.priority === 'Critical' && a.status === 'active').length
    if (criticalActive > prevCriticalCount.current) {
      playAlertBeep()
    }
    prevCriticalCount.current = criticalActive
  }, [alerts])

  const acknowledgeMutation = useMutation({
    mutationFn: (alertId: string) => api.post(`/alerts/${alertId}/acknowledge`, {}),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['alerts-center'] }),
  })

  const snoozeMutation = useMutation({
    mutationFn: ({ alertId, reason, minutes }: { alertId: string; reason: string; minutes: number }) =>
      api.post(`/alerts/${alertId}/snooze`, { reason, duration_minutes: minutes }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['alerts-center'] }),
  })

  const escalateMutation = useMutation({
    mutationFn: (alertId: string) => api.post(`/alerts/${alertId}/escalate`, {}),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['alerts-center'] }),
  })

  const handleAcknowledge = useCallback((id: string) => {
    acknowledgeMutation.mutate(id)
  }, [acknowledgeMutation])

  const handleSnoozeConfirm = useCallback((alertId: string, reason: string, minutes: number) => {
    snoozeMutation.mutate({ alertId, reason, minutes })
  }, [snoozeMutation])

  const handleEscalate = useCallback((id: string) => {
    escalateMutation.mutate(id)
  }, [escalateMutation])

  // Filter + sort
  const filtered = alerts
    .filter((a) => {
      if (filterPriority && a.priority !== filterPriority) return false
      if (filterStatus && a.status !== filterStatus) return false
      return true
    })
    .sort((a, b) => {
      const pd = PRIORITY_ORDER[a.priority] - PRIORITY_ORDER[b.priority]
      if (pd !== 0) return pd
      return new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
    })

  const criticalActiveCount = alerts.filter((a) => a.priority === 'Critical' && a.status === 'active').length

  return (
    <>
      <CriticalBanner alerts={alerts} onAcknowledge={handleAcknowledge} />

      {snoozeTarget && (
        <SnoozeModal
          alertId={snoozeTarget}
          onClose={() => setSnoozeTarget(null)}
          onConfirm={handleSnoozeConfirm}
        />
      )}

      <div className="min-h-screen bg-gray-50 p-4">
        <div className="mx-auto max-w-screen-lg space-y-4">

          {/* Page header */}
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-lg font-bold text-gray-900">Alert Center</h1>
              <p className="text-sm text-gray-500">
                {alerts.filter((a) => a.status === 'active').length} active alerts
                {criticalActiveCount > 0 && (
                  <span className="ml-2 font-semibold text-red-600">
                    · {criticalActiveCount} critical
                  </span>
                )}
              </p>
            </div>
          </div>

          {/* Filter bar */}
          <div className="flex flex-wrap gap-2 rounded-lg bg-white p-3 shadow-sm border border-gray-200">
            <select
              value={filterPriority}
              onChange={(e) => setFilterPriority(e.target.value as AlertPriority | '')}
              className="rounded border border-gray-300 px-2 py-1 text-xs"
              aria-label="Filter by priority"
            >
              <option value="">All priorities</option>
              <option value="Critical">Critical</option>
              <option value="High">High</option>
              <option value="Medium">Medium</option>
              <option value="Low">Low</option>
            </select>

            <select
              value={filterStatus}
              onChange={(e) => setFilterStatus(e.target.value as AlertStatus | 'active')}
              className="rounded border border-gray-300 px-2 py-1 text-xs"
              aria-label="Filter by status"
            >
              <option value="active">Active only</option>
              <option value="">All statuses</option>
              <option value="acknowledged">Acknowledged</option>
              <option value="snoozed">Snoozed</option>
              <option value="resolved">Resolved</option>
            </select>

            <span className="ml-auto self-center text-xs text-gray-400">
              {filtered.length} shown
            </span>
          </div>

          {/* Alert list */}
          {isLoading ? (
            <div className="space-y-2">
              {[...Array(5)].map((_, i) => (
                <div key={i} className="h-20 animate-pulse rounded-lg bg-gray-200" />
              ))}
            </div>
          ) : filtered.length === 0 ? (
            <div className="flex h-40 items-center justify-center rounded-lg border border-gray-200 bg-white text-gray-400">
              No alerts match the current filters.
            </div>
          ) : (
            <div className="space-y-2">
              {filtered.map((alert) => (
                <AlertRow
                  key={alert.alert_id}
                  alert={alert}
                  onAcknowledge={handleAcknowledge}
                  onSnooze={(id) => setSnoozeTarget(id)}
                  onEscalate={handleEscalate}
                />
              ))}
            </div>
          )}

        </div>
      </div>
    </>
  )
}
