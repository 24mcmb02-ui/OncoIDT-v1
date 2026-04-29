import { useRef, useEffect, useState, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import * as d3 from 'd3'
import type { BedState, Alert } from '@/types'

interface WardHeatmapProps {
  beds: Record<string, BedState>
  patients: Record<string, { patient_id: string; mrn: string }>
  alerts: Alert[]
  /** Collapse to list view below this width (px). Default 1024. */
  collapseWidth?: number
}

interface BedDatum extends BedState {
  displayName: string
  topAlert: Alert | null
  riskScore: number
}

// D3 color scale: green → yellow → red
const riskColor = d3
  .scaleSequential(d3.interpolateRdYlGn)
  .domain([1, 0]) // reversed so high risk = red

const CELL_SIZE = 72
const CELL_GAP = 8
const COLS = 6

function buildBedData(
  beds: Record<string, BedState>,
  patients: Record<string, { patient_id: string; mrn: string }>,
  alerts: Alert[],
): BedDatum[] {
  return Object.values(beds).map((bed) => {
    const patient = bed.patient_id ? patients[bed.patient_id] : null
    const bedAlerts = alerts.filter(
      (a) => a.patient_id === bed.patient_id && a.status === 'active',
    )
    const topAlert =
      bedAlerts.sort((a, b) => {
        const order: Record<string, number> = { Critical: 0, High: 1, Medium: 2, Low: 3 }
        return (order[a.priority] ?? 4) - (order[b.priority] ?? 4)
      })[0] ?? null

    const riskScore = Math.max(
      bed.infection_risk_score ?? 0,
      bed.deterioration_risk_score ?? 0,
    )

    return {
      ...bed,
      displayName: patient ? patient.mrn : bed.bed_id,
      topAlert,
      riskScore,
    }
  })
}

// ---------------------------------------------------------------------------
// SVG Heatmap (desktop)
// ---------------------------------------------------------------------------

function SvgHeatmap({ data, onBedClick }: { data: BedDatum[]; onBedClick: (id: string) => void }) {
  const svgRef = useRef<SVGSVGElement>(null)
  const [tooltip, setTooltip] = useState<{
    x: number
    y: number
    bed: BedDatum
  } | null>(null)

  useEffect(() => {
    if (!svgRef.current) return
    const svg = d3.select(svgRef.current)
    svg.selectAll('*').remove()

    const rows = Math.ceil(data.length / COLS)
    const width = COLS * (CELL_SIZE + CELL_GAP) + CELL_GAP
    const height = rows * (CELL_SIZE + CELL_GAP) + CELL_GAP

    svg.attr('width', width).attr('height', height)

    const cells = svg
      .selectAll<SVGGElement, BedDatum>('g.bed-cell')
      .data(data, (d) => d.bed_id)
      .join('g')
      .attr('class', 'bed-cell')
      .attr('transform', (_, i) => {
        const col = i % COLS
        const row = Math.floor(i / COLS)
        return `translate(${CELL_GAP + col * (CELL_SIZE + CELL_GAP)}, ${CELL_GAP + row * (CELL_SIZE + CELL_GAP)})`
      })
      .style('cursor', (d) => (d.patient_id ? 'pointer' : 'default'))

    // Background rect
    cells
      .append('rect')
      .attr('width', CELL_SIZE)
      .attr('height', CELL_SIZE)
      .attr('rx', 6)
      .attr('fill', (d) => (d.patient_id ? riskColor(d.riskScore) : '#e5e7eb'))
      .attr('stroke', (d) => {
        if (!d.topAlert) return 'transparent'
        const colors: Record<string, string> = {
          Critical: '#dc2626',
          High: '#ea580c',
          Medium: '#ca8a04',
          Low: '#16a34a',
        }
        return colors[d.topAlert.priority] ?? 'transparent'
      })
      .attr('stroke-width', (d) => (d.topAlert ? 2.5 : 0))

    // Bed ID label
    cells
      .append('text')
      .attr('x', CELL_SIZE / 2)
      .attr('y', 20)
      .attr('text-anchor', 'middle')
      .attr('font-size', 11)
      .attr('font-weight', '600')
      .attr('fill', '#1f2937')
      .text((d) => d.bed_id)

    // Risk score label
    cells
      .filter((d) => d.patient_id !== null)
      .append('text')
      .attr('x', CELL_SIZE / 2)
      .attr('y', CELL_SIZE / 2 + 4)
      .attr('text-anchor', 'middle')
      .attr('font-size', 18)
      .attr('font-weight', '700')
      .attr('fill', '#111827')
      .text((d) => (d.riskScore * 100).toFixed(0) + '%')

    // Exposure flag indicator
    cells
      .filter((d) => d.exposure_flag)
      .append('circle')
      .attr('cx', CELL_SIZE - 10)
      .attr('cy', 10)
      .attr('r', 5)
      .attr('fill', '#7c3aed')

    // Interaction
    cells
      .on('mouseenter', (event: MouseEvent, d) => {
        if (!d.patient_id) return
        const rect = (event.target as Element).closest('g')?.getBoundingClientRect()
        setTooltip({
          x: (rect?.left ?? 0) + CELL_SIZE / 2,
          y: (rect?.top ?? 0) - 8,
          bed: d,
        })
      })
      .on('mouseleave', () => setTooltip(null))
      .on('click', (_event, d) => {
        if (d.patient_id) onBedClick(d.patient_id)
      })
  }, [data, onBedClick])

  return (
    <div className="relative">
      <svg ref={svgRef} aria-label="Ward bed heatmap" role="img" />
      {tooltip && (
        <div
          className="pointer-events-none fixed z-50 rounded-lg bg-gray-900 px-3 py-2 text-xs text-white shadow-lg"
          style={{ left: tooltip.x, top: tooltip.y, transform: 'translate(-50%, -100%)' }}
        >
          <div className="font-semibold">{tooltip.bed.displayName}</div>
          <div>Infection: {((tooltip.bed.infection_risk_score ?? 0) * 100).toFixed(1)}%</div>
          <div>Deterioration: {((tooltip.bed.deterioration_risk_score ?? 0) * 100).toFixed(1)}%</div>
          {tooltip.bed.topAlert && (
            <div className="mt-1 font-medium text-yellow-300">
              ⚠ {tooltip.bed.topAlert.priority}: {tooltip.bed.topAlert.alert_type}
            </div>
          )}
          {tooltip.bed.exposure_flag && (
            <div className="text-purple-300">● Exposure flagged</div>
          )}
        </div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// List view (mobile / narrow viewport)
// ---------------------------------------------------------------------------

function BedListView({ data, onBedClick }: { data: BedDatum[]; onBedClick: (id: string) => void }) {
  const priorityColor: Record<string, string> = {
    Critical: 'bg-red-100 text-red-800',
    High: 'bg-orange-100 text-orange-800',
    Medium: 'bg-yellow-100 text-yellow-800',
    Low: 'bg-green-100 text-green-800',
  }

  return (
    <div className="divide-y divide-gray-200 rounded-lg border border-gray-200">
      {data.map((bed) => (
        <div
          key={bed.bed_id}
          className={`flex items-center gap-3 px-4 py-3 ${bed.patient_id ? 'cursor-pointer hover:bg-gray-50' : 'opacity-50'}`}
          onClick={() => bed.patient_id && onBedClick(bed.patient_id)}
          role={bed.patient_id ? 'button' : undefined}
          tabIndex={bed.patient_id ? 0 : undefined}
          onKeyDown={(e) => e.key === 'Enter' && bed.patient_id && onBedClick(bed.patient_id)}
          aria-label={bed.patient_id ? `Navigate to patient ${bed.displayName}` : `Bed ${bed.bed_id} empty`}
        >
          {/* Risk color swatch */}
          <div
            className="h-10 w-10 flex-shrink-0 rounded-md"
            style={{ backgroundColor: bed.patient_id ? riskColor(bed.riskScore) : '#e5e7eb' }}
          />
          <div className="min-w-0 flex-1">
            <div className="flex items-center gap-2">
              <span className="font-medium text-gray-900">{bed.bed_id}</span>
              {bed.patient_id && (
                <span className="truncate text-sm text-gray-500">{bed.displayName}</span>
              )}
              {bed.exposure_flag && (
                <span className="text-xs text-purple-600">● Exposure</span>
              )}
            </div>
            {bed.patient_id && (
              <div className="mt-0.5 flex gap-3 text-xs text-gray-600">
                <span>Inf: {((bed.infection_risk_score ?? 0) * 100).toFixed(1)}%</span>
                <span>Det: {((bed.deterioration_risk_score ?? 0) * 100).toFixed(1)}%</span>
              </div>
            )}
          </div>
          {bed.topAlert && (
            <span className={`rounded-full px-2 py-0.5 text-xs font-medium ${priorityColor[bed.topAlert.priority] ?? ''}`}>
              {bed.topAlert.priority}
            </span>
          )}
        </div>
      ))}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export function WardHeatmap({ beds, patients, alerts, collapseWidth = 1024 }: WardHeatmapProps) {
  const navigate = useNavigate()
  const [isNarrow, setIsNarrow] = useState(window.innerWidth < collapseWidth)

  useEffect(() => {
    const handler = () => setIsNarrow(window.innerWidth < collapseWidth)
    window.addEventListener('resize', handler)
    return () => window.removeEventListener('resize', handler)
  }, [collapseWidth])

  const data = buildBedData(beds, patients, alerts)

  const handleBedClick = useCallback(
    (patientId: string) => navigate(`/patient/${patientId}`),
    [navigate],
  )

  return (
    <div className="overflow-auto">
      {isNarrow ? (
        <BedListView data={data} onBedClick={handleBedClick} />
      ) : (
        <SvgHeatmap data={data} onBedClick={handleBedClick} />
      )}
    </div>
  )
}
