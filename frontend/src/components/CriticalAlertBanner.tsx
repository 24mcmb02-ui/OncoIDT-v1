import { useWardStore } from '@/store/wardStore'
import { api } from '@/api/client'

export function CriticalAlertBanner() {
  const alert = useWardStore((s) => s.criticalAlertBanner)
  const dismissCriticalBanner = useWardStore((s) => s.dismissCriticalBanner)
  const acknowledgeAlert = useWardStore((s) => s.acknowledgeAlert)

  if (!alert) return null

  async function handleAcknowledge() {
    if (!alert) return
    try {
      await api.post(`/alerts/${alert.alert_id}/acknowledge`, {})
      acknowledgeAlert(alert.alert_id)
    } catch {
      // Optimistically dismiss anyway
      acknowledgeAlert(alert.alert_id)
    }
  }

  return (
    <div
      role="alertdialog"
      aria-modal="true"
      aria-label="Critical alert"
      className="fixed inset-x-0 top-0 z-50 flex items-center justify-between gap-4 bg-red-600 px-4 py-3 text-white shadow-lg"
    >
      <div className="flex items-center gap-3">
        <span className="animate-pulse text-xl" aria-hidden="true">🚨</span>
        <div>
          <span className="font-bold">CRITICAL ALERT</span>
          <span className="ml-2 text-sm">{alert.message}</span>
        </div>
      </div>
      <button
        onClick={handleAcknowledge}
        className="rounded bg-white px-3 py-1 text-sm font-semibold text-red-700 hover:bg-red-50 focus:outline-none focus:ring-2 focus:ring-white"
        aria-label="Acknowledge critical alert"
      >
        Acknowledge
      </button>
    </div>
  )
}
