'use client'

import type { HeatmapMatrix } from '@/lib/api'

const BAND_LABELS: Record<string, string> = {
  Poor: 'Poor credit',
  Fair: 'Fair credit',
  Good: 'Good credit',
  Excellent: 'Excellent credit',
}

const ON_TIME = new Set(['Current', 'Prepaid', 'Closed'])
const LATE = new Set(['30DPD', '60DPD', '90DPD'])
const DEFAULTED = new Set(['Default'])

function colIndex(cols: string[], names: Set<string>) {
  return cols.map((col, i) => (names.has(col) ? i : -1)).filter((i) => i >= 0)
}

function sumAt(row: number[], indexes: number[]) {
  return indexes.reduce((total, i) => total + Number(row[i] || 0), 0)
}

function pct(part: number, total: number) {
  if (!total) return 0
  return Math.round((part / total) * 100)
}

function tone(kind: 'ok' | 'late' | 'bad', share: number) {
  const t = Math.min(1, share / 70)
  if (kind === 'ok') return `rgba(14, 119, 117, ${0.12 + t * 0.55})`
  if (kind === 'late') return `rgba(210, 122, 60, ${0.12 + t * 0.65})`
  return `rgba(199, 88, 61, ${0.14 + t * 0.7})`
}

export function toSimpleHealth(matrix?: HeatmapMatrix) {
  if (!matrix?.rows?.length || !matrix.cols?.length) return []
  const okIdx = colIndex(matrix.cols, ON_TIME)
  const lateIdx = colIndex(matrix.cols, LATE)
  const defIdx = colIndex(matrix.cols, DEFAULTED)

  return matrix.rows.map((band, i) => {
    const row = matrix.cells[i] || []
    const ok = sumAt(row, okIdx)
    const late = sumAt(row, lateIdx)
    const defn = sumAt(row, defIdx)
    const total = ok + late + defn
    return {
      band,
      label: BAND_LABELS[band] || band,
      total,
      ok,
      late,
      defn,
      okPct: pct(ok, total),
      latePct: pct(late, total),
      defPct: pct(defn, total),
    }
  }).filter((row) => row.total > 0)
}

export default function SimpleHealthMap({ matrix }: { matrix?: HeatmapMatrix }) {
  const rows = toSimpleHealth(matrix)
  if (!rows.length) {
    return (
      <div className="heatmap-card">
        <h3>Who is paying, and who is not?</h3>
        <div className="empty-data">Portfolio data is still loading.</div>
      </div>
    )
  }

  const worst = [...rows].sort((a, b) => b.defPct + b.latePct - (a.defPct + a.latePct))[0]
  const takeaway = worst
    ? `${worst.label} is the problem pocket: ${worst.latePct}% of those loans are behind, and ${worst.defPct}% have already defaulted.`
    : ''

  return (
    <div className="heatmap-card simple-map">
      <div className="heatmap-head">
        <h3>Who is paying, and who is not?</h3>
        <p>Each row is one credit group. Read left to right: on time, late, defaulted. Darker = a bigger share of that group.</p>
      </div>
      <div className="health-grid">
        <div className="health-axis">
          <span />
          <span>Paying on time</span>
          <span>Behind on payments</span>
          <span>Already defaulted</span>
        </div>
        {rows.map((row) => (
          <div className="health-row" key={row.band}>
            <strong>{row.label}<small>{row.total.toLocaleString()} loans</small></strong>
            <div className="health-cell" style={{ background: tone('ok', row.okPct) }}>
              <b>{row.okPct}%</b>
              <em>on time</em>
            </div>
            <div className="health-cell" style={{ background: tone('late', row.latePct) }}>
              <b>{row.latePct}%</b>
              <em>late</em>
            </div>
            <div className="health-cell bad" style={{ background: tone('bad', row.defPct) }}>
              <b>{row.defPct}%</b>
              <em>defaulted</em>
            </div>
          </div>
        ))}
      </div>
      <p className="map-takeaway">{takeaway}</p>
    </div>
  )
}

export function CreditRiskBars({ matrix }: { matrix?: HeatmapMatrix }) {
  const rows = matrix?.rows?.map((band, i) => ({
    band,
    label: BAND_LABELS[band] || band,
    rate: Number(matrix.cells[i]?.[0] || 0),
  })).filter((row) => row.rate > 0) || []

  if (!rows.length) return null
  const max = Math.max(...rows.map((row) => row.rate), 0.01)

  return (
    <div className="heatmap-card">
      <div className="heatmap-head">
        <h3>If nothing changes, who is most likely to default?</h3>
        <p>Simple read: poorer credit → higher default chance. One number per group.</p>
      </div>
      <div className="bar-list">
        {rows.map((row) => (
          <div className="bar-row" key={row.band}>
            <span>{row.label}</span>
            <div className="bar-track"><i style={{ width: `${(row.rate / max) * 100}%` }} /></div>
            <b>{(row.rate * 100).toFixed(0)}%</b>
          </div>
        ))}
      </div>
    </div>
  )
}
