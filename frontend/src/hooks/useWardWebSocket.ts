import { useEffect, useRef, useCallback } from 'react'
import { getToken } from '@/api/client'
import { useWardStore } from '@/store/wardStore'
import type { WsMessage } from '@/types'

const MIN_RECONNECT_MS = 1_000
const MAX_RECONNECT_MS = 30_000

/**
 * Connects to ws://host/ws/v1/ward/{wardId} with JWT query param.
 * Implements exponential backoff reconnect (max 30s interval).
 * Dispatches incoming messages to the Zustand ward store.
 */
export function useWardWebSocket(wardId: string): void {
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectDelayRef = useRef<number>(MIN_RECONNECT_MS)
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const unmountedRef = useRef(false)

  const applyScoreUpdate = useWardStore((s) => s.applyScoreUpdate)
  const applyAlertGenerated = useWardStore((s) => s.applyAlertGenerated)
  const applyWardStateChange = useWardStore((s) => s.applyWardStateChange)
  const setWsConnected = useWardStore((s) => s.setWsConnected)

  const connect = useCallback(() => {
    if (unmountedRef.current) return

    const token = getToken()
    const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws'
    const host = window.location.host
    const url = `${protocol}://${host}/ws/v1/ward/${wardId}${token ? `?token=${encodeURIComponent(token)}` : ''}`

    const ws = new WebSocket(url)
    wsRef.current = ws

    ws.onopen = () => {
      if (unmountedRef.current) {
        ws.close()
        return
      }
      reconnectDelayRef.current = MIN_RECONNECT_MS
      setWsConnected(true)
    }

    ws.onmessage = (event: MessageEvent) => {
      let msg: WsMessage
      try {
        msg = JSON.parse(event.data as string) as WsMessage
      } catch {
        console.warn('[WS] Failed to parse message', event.data)
        return
      }

      switch (msg.type) {
        case 'score_update':
          applyScoreUpdate(msg)
          break
        case 'alert_generated':
          applyAlertGenerated(msg)
          break
        case 'ward_state_change':
          applyWardStateChange(msg)
          break
        default:
          console.warn('[WS] Unknown message type', (msg as { type: string }).type)
      }
    }

    ws.onerror = (err) => {
      console.error('[WS] Error', err)
    }

    ws.onclose = () => {
      setWsConnected(false)
      wsRef.current = null

      if (unmountedRef.current) return

      // Exponential backoff reconnect
      const delay = reconnectDelayRef.current
      reconnectDelayRef.current = Math.min(delay * 2, MAX_RECONNECT_MS)

      reconnectTimerRef.current = setTimeout(() => {
        if (!unmountedRef.current) connect()
      }, delay)
    }
  }, [wardId, applyScoreUpdate, applyAlertGenerated, applyWardStateChange, setWsConnected])

  useEffect(() => {
    unmountedRef.current = false
    connect()

    return () => {
      unmountedRef.current = true
      if (reconnectTimerRef.current) {
        clearTimeout(reconnectTimerRef.current)
      }
      if (wsRef.current) {
        wsRef.current.close()
        wsRef.current = null
      }
      setWsConnected(false)
    }
  }, [connect, setWsConnected])
}
