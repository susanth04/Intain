export type StageKey = 'ingestion' | 'quality' | 'features' | 'prediction' | 'anomaly' | 'explainability' | 'scenarios' | 'reviewer'

export type StageStatus = 'waiting' | 'running' | 'completed' | 'failed'

export type PipelineStage = {
  key: StageKey
  label: string
  status: StageStatus
  output?: unknown
  error?: string
  executionMs?: number
}

export type PipelineResponse = {
  loan_id: string
  stages: PipelineStage[]
  summary?: unknown
  [key: string]: unknown
}

const API_URL = process.env.NEXT_PUBLIC_API_URL || process.env.VITE_API_URL || ''

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...(init?.headers || {}) },
  })
  const body = await response.json().catch(() => ({}))
  if (!response.ok) throw new Error(body.detail || body.message || body.error || `Request failed (${response.status})`)
  return body
}

export function runPipeline(loanId: string) {
  return request<PipelineResponse>('/api/pipeline/run', { method: 'POST', body: JSON.stringify({ loan_id: loanId }) })
}

export function runScenario(loanId: string, scenario: string) {
  return request<unknown>('/api/scenario', { method: 'POST', body: JSON.stringify({ loan_id: loanId, scenario }) })
}

export function askCopilot(loanId: string, question: string) {
  return request<{ answer?: string; response?: string; [key: string]: unknown }>('/api/copilot', {
    method: 'POST', body: JSON.stringify({ loan_id: loanId, question }),
  })
}

export type HeatmapMatrix = {
  rows: string[]
  cols: string[]
  cells: number[][]
  max?: number
  unit?: string
}

export type PortfolioOverview = {
  total_loans?: number
  tier_summary?: Record<string, number>
  status_heatmap?: HeatmapMatrix
  dpd_heatmap?: HeatmapMatrix
  scenario_heatmap?: HeatmapMatrix
  state_heatmap?: HeatmapMatrix
  exception_mix?: { type: string; count: number }[]
  top_loans?: Record<string, unknown>[]
  metrics?: { highlights?: Record<string, unknown>[] }
  demo_note?: string
  [key: string]: unknown
}

export function fetchPortfolio(limit = 24) {
  return request<PortfolioOverview>(`/api/portfolio/overview?limit=${limit}`)
}

export function retryStage(loanId: string, stage: StageKey) {
  return request<PipelineResponse>('/api/pipeline/run', { method: 'POST', body: JSON.stringify({ loan_id: loanId, stage }) })
}

export function apiBaseUrl() { return API_URL || 'Relative API path' }

export function valueAt(object: unknown, ...keys: string[]): unknown {
  if (!object || typeof object !== 'object') return undefined
  const record = object as Record<string, unknown>
  for (const key of keys) if (record[key] !== undefined && record[key] !== null) return record[key]
  return undefined
}

export function displayValue(value: unknown, fallback = '—') {
  if (value === undefined || value === null || value === '') return fallback
  if (typeof value === 'object') return JSON.stringify(value)
  return String(value)
}

export const stageLabels: Record<StageKey, string> = {
  ingestion: 'Data ingestion', quality: 'Data quality', features: 'Features', prediction: 'Risk model',
  anomaly: 'Anomaly', explainability: 'Explainability', scenarios: 'Scenarios', reviewer: 'AI reviewer',
}
export const stageOrder: StageKey[] = ['ingestion', 'quality', 'features', 'prediction', 'anomaly', 'explainability', 'scenarios', 'reviewer']

export function normalizeStages(result: PipelineResponse): PipelineStage[] {
  const source = Array.isArray(result.stages) ? result.stages : []
  return stageOrder.map((key, index) => {
    const found = source.find((item) => item.key === key || item.label?.toLowerCase().includes(stageLabels[key].split(' ')[0]))
    return found || { key, label: stageLabels[key], status: index === 0 ? 'completed' : 'waiting' }
  })
}

export function errorText(error: unknown) { return error instanceof Error ? error.message : 'Backend request failed' }

type UnknownRecord = Record<string, unknown>
export function asRecord(value: unknown): UnknownRecord { return value && typeof value === 'object' ? value as UnknownRecord : {} }
export function entries(value: unknown): [string, unknown][] { return Object.entries(asRecord(value)) }

export { API_URL }

export type { UnknownRecord }

type _KeepTypes = never
void (0 as unknown as _KeepTypes)
