'use client'

import { useEffect, useMemo, useState, useRef } from 'react'
import * as d3 from 'd3'
import { AlertTriangle, ArrowRight, BarChart3, Check, ChevronRight, Circle, Layers3, Menu, MessageSquareText, RefreshCw, Search, ShieldCheck, Sparkles, X, XCircle, Moon, Sun } from 'lucide-react'
import CopilotAnswer from '@/components/copilot-answer'
import SimpleHealthMap, { CreditRiskBars } from '@/components/heatmap-grid'
import { LedgerCard } from '@/components/ui/ledger-card'
import { askCopilot, asRecord, displayValue, entries, errorText, fetchModelComparison, fetchPortfolio, fetchSurvival, normalizeStages, retryStage, runScenario, stageLabels, stageOrder, type ModelComparison, type PipelineResponse, type PipelineStage, type PortfolioOverview, type StageKey, type StageStatus, type SurvivalResponse, valueAt } from '@/lib/api'

type DetailDrawerProps = { stage: PipelineStage; loanId: string; onClose: () => void; onRetry: () => void }
type NavKey = 'Overview' | 'Loan Intelligence' | 'Anomaly Detection' | 'Scenario Analysis' | 'AI Reviewer' | 'Model Cards'

const stageNumbers = ['01', '02', '03', '04', '05', '06', '07', '08']
const prompts = ['Why was this loan flagged?', 'What are the main risk drivers?', 'Explain the model prediction.', 'What data-quality issues were found?']
const navItems: NavKey[] = ['Overview', 'Loan Intelligence', 'Anomaly Detection', 'Scenario Analysis', 'AI Reviewer', 'Model Cards']

type ModelCardKey = 'next_3m_delinquency_lgbm' | 'next_6m_delinquency_lgbm' | 'next_12m_default_lr' | 'next_12m_prepayment_lr' | 'next_state_lgbm'

function StatusIcon({ status }: { status: StageStatus }) {
  if (status === 'completed') return <span className="status-icon completed"><Check /></span>
  if (status === 'failed') return <span className="status-icon failed"><X /></span>
  if (status === 'running') return <span className="status-icon running"><span /></span>
  return <span className="status-icon waiting"><Circle /></span>
}

function statusLabel(status: StageStatus) { return status.toUpperCase() }

function DataGrid({ data }: { data: unknown }) {
  const rows = entries(data)
  if (!rows.length) return <div className="empty-data">No backend output returned for this stage.</div>
  return <div className="data-grid">{rows.map(([key, value]) => <DataRow key={key} label={key} value={value} />)}</div>
}

function DataRow({ label, value }: { label: string; value: unknown }) {
  if (Array.isArray(value)) {
    return <div className="data-row data-row-stack"><span>{label.replaceAll('_', ' ')}</span><div className="data-list">{value.map((item, index) => <div key={index}>{typeof item === 'object' ? <DataGrid data={item} /> : displayValue(item)}</div>)}</div></div>
  }
  if (value && typeof value === 'object') return <div className="data-row data-row-stack"><span>{label.replaceAll('_', ' ')}</span><DataGrid data={value} /></div>
  return <div className="data-row"><span>{label.replaceAll('_', ' ')}</span><strong>{displayValue(value)}</strong></div>
}

function MetricsPanel({ portfolio, inline }: { portfolio: PortfolioOverview | null; inline?: boolean }) {
  const metrics = asRecord(portfolio?.metrics)
  const highlights = Array.isArray(metrics.highlights) ? metrics.highlights.map(asRecord) : []
  const inner = (
    <>
      <div className="metric-list">{highlights.map((metric, index) => <div className="metric-row" key={index}><div><strong>{displayValue(metric.target)}</strong><span>{displayValue(metric.note)}</span></div><b>{typeof metric.lgbm_test_roc === 'number' ? `ROC-AUC ${(metric.lgbm_test_roc * 100).toFixed(1)}%` : displayValue(metric.lgbm_test_roc)}</b></div>)}</div>
      <div className="metric-row" style={{marginTop: '16px', borderTop: '1px solid var(--line)', paddingTop: '16px'}}><span>Survival model concordance</span><strong>{typeof metrics.cox_concordance === 'number' ? metrics.cox_concordance.toFixed(4) : '—'}</strong></div>
    </>
  )
  if (inline) return inner
  return <LedgerCard className="metrics-panel"><div className="ledger-card-title"><div><p className="eyebrow">VALIDATION</p><h3>Model performance</h3></div><p className="body-text">Real evaluation results from the strict out-of-time test window. These are model diagnostics, not loan-level predictions.</p></div>{inner}</LedgerCard>
}

function PortfolioPulse({ portfolio }: { portfolio: PortfolioOverview | null }) {
  const containerRef = useRef<HTMLDivElement>(null)
  
  useEffect(() => {
    if (!containerRef.current || !portfolio?.status_heatmap) return
    const rows = portfolio.status_heatmap.rows || []
    const cells = portfolio.status_heatmap.cells || []
    const data = rows.map((label, index) => ({ 
      label, 
      value: (cells[index] || []).reduce((sum, v) => sum + Number(v || 0), 0) 
    })).filter(d => d.value > 0).reverse()
    
    if (data.length === 0) return
    const container = containerRef.current
    container.innerHTML = ''
    
    const width = container.clientWidth
    const height = data.length * 32
    const margin = { top: 0, right: 60, bottom: 0, left: 100 }
    
    const svg = d3.select(container).append('svg')
      .attr('width', '100%')
      .attr('height', height)
      .attr('viewBox', `0 0 ${width} ${height}`)
      .attr('preserveAspectRatio', 'xMinYMin meet')
      
    const x = d3.scaleLinear().domain([0, d3.max(data, d => d.value) || 1]).range([margin.left, width - margin.right])
    const y = d3.scaleBand().domain(data.map(d => d.label)).range([0, height]).padding(0.4)
      
    const defs = svg.append("defs")
    const gradient = defs.append("linearGradient").attr("id", "bar-grad").attr("x1", "0%").attr("y1", "0%").attr("x2", "100%").attr("y2", "0%")
    gradient.append("stop").attr("offset", "0%").attr("stop-color", "var(--border-line-subtle)")
    gradient.append("stop").attr("offset", "100%").attr("stop-color", "var(--teal)")

    svg.selectAll('.bg-bar').data(data).enter().append('rect').attr('class', 'bg-bar')
      .attr('x', margin.left).attr('y', d => y(d.label) || 0)
      .attr('width', width - margin.left - margin.right).attr('height', y.bandwidth())
      .attr('fill', 'var(--border-line-subtle)').attr('rx', 4)

    svg.selectAll('.bar').data(data).enter().append('rect').attr('class', 'bar')
      .attr('x', margin.left).attr('y', d => y(d.label) || 0)
      .attr('width', d => Math.max(x(d.value) - margin.left, 4)).attr('height', y.bandwidth())
      .attr('fill', 'url(#bar-grad)').attr('rx', 4)
      
    svg.selectAll('.label').data(data).enter().append('text').attr('class', 'label')
      .attr('x', margin.left - 12).attr('y', d => (y(d.label) || 0) + y.bandwidth() / 2)
      .attr('dy', '0.35em').attr('text-anchor', 'end')
      .attr('fill', 'var(--text-primary)').attr('font-size', '12px').text(d => d.label)
      
    svg.selectAll('.val').data(data).enter().append('text').attr('class', 'val')
      .attr('x', width).attr('y', d => (y(d.label) || 0) + y.bandwidth() / 2)
      .attr('dy', '0.35em').attr('text-anchor', 'end')
      .attr('fill', 'var(--teal)').attr('font-size', '12px')
      .attr('font-family', 'var(--mono)').attr('font-weight', '500')
      .text(d => d.value.toLocaleString())
  }, [portfolio])
  
  if (!portfolio?.status_heatmap?.rows?.length) return <div className="empty-data">Portfolio distribution is loading.</div>
  
  return <div className="portfolio-pulse" aria-label="Portfolio distribution by credit band">
    <div className="pulse-head"><span>CREDIT BAND</span><span /><span>LOAN COUNT</span></div>
    <div ref={containerRef} style={{ width: '100%', minHeight: '128px' }} />
  </div>
}

function MiniD3Bar({ value }: { value: number | null }) {
  const containerRef = useRef<HTMLDivElement>(null)
  
  useEffect(() => {
    if (!containerRef.current) return
    const container = containerRef.current
    container.innerHTML = ''
    
    const width = container.clientWidth
    const height = 6
    const svg = d3.select(container).append('svg')
      .attr('width', '100%')
      .attr('height', height)
      .attr('viewBox', `0 0 ${width} ${height}`)
      .attr('preserveAspectRatio', 'none')
      
    svg.append('rect')
      .attr('width', '100%')
      .attr('height', height)
      .attr('fill', 'var(--border-line-subtle)')
      .attr('rx', 3)
      
    if (value !== null && value !== undefined) {
      svg.append('rect')
        .attr('width', `${Math.max(value * 100, 2)}%`)
        .attr('height', height)
        .attr('fill', 'var(--teal)')
        .attr('rx', 3)
    }
  }, [value])
  
  return <div ref={containerRef} style={{ width: '100%', marginTop: '12px', opacity: value !== null ? 1 : 0.2 }} />
}

function SurvivalCard({ data, loading, error }: { data: SurvivalResponse | null; loading: boolean; error: string }) {
  const hazardRatio = data?.cox_hazard_ratio
  const hazardLevel = typeof hazardRatio !== 'number' ? '' : hazardRatio < 1 ? 'low' : hazardRatio <= 1.5 ? 'moderate' : 'high'
  const ageProgress = data && data.km_median_survival_months > 0
    ? Math.min(data.loan_age_months / data.km_median_survival_months, 1)
    : null

  return <LedgerCard>
    <div className="ledger-card-title">
      <div><p className="eyebrow">TIME TO EXIT</p><h3>Survival outlook</h3></div>
      <p className="body-text">Kaplan-Meier median and Cox relative hazard for this loan.</p>
    </div>
    {loading ? <div className="loading-state"><RefreshCw className="spin" /> Loading survival estimate…</div> : error ? <div className="error-box"><AlertTriangle />{error}</div> : data ? <>
      <div className="survival-metrics">
        <div className="survival-metric"><span>KM MEDIAN SURVIVAL</span><strong>{displayValue(data.km_median_survival_months)} months</strong></div>
        <div className="survival-metric"><span>CREDIT BAND</span><strong>{data.credit_band}</strong></div>
      </div>
      <div className="survival-age">
        <div><span>Loan age</span><strong>{displayValue(data.loan_age_months)} / {displayValue(data.km_median_survival_months)} months</strong></div>
        <MiniD3Bar value={ageProgress} />
      </div>
      <div className="survival-remaining"><span>ESTIMATED MONTHS REMAINING</span><strong>{displayValue(data.estimated_months_remaining)} months</strong></div>
      <div className="survival-hazard-row"><span>Cox hazard ratio</span>{typeof hazardRatio === 'number' ? <span className={`survival-hazard ${hazardLevel}`}>{hazardRatio.toFixed(2)} · {displayValue(data.cox_interpretation)}</span> : <strong>Unavailable</strong>}</div>
      <p className="survival-concordance">Cox concordance {typeof data.cox_concordance === 'number' ? data.cox_concordance.toFixed(2) : 'unavailable'} - modest ranking power</p>
      <p className="body-text survival-label">{data.survival_label}</p>
    </> : <div className="empty-data">Run loan analysis to see its survival outlook.</div>}
  </LedgerCard>
}

function PredictionCards({ output }: { output: unknown }) {
  const data = asRecord(output)
  const preds = asRecord(valueAt(data, 'predictions'))
  const getProb = (key: string) => {
    const modelOut = asRecord(valueAt(preds, key))
    const prob = valueAt(modelOut, 'probability')
    if (typeof prob === 'number') return `${(prob * 100).toFixed(1)}%`
    return prob
  }
  const getRawProb = (key: string) => {
    const modelOut = asRecord(valueAt(preds, key))
    const prob = valueAt(modelOut, 'probability')
    if (typeof prob === 'number') return prob
    return null
  }
  const cards = [
    ['3M DELINQUENCY', 'next_3m_delinquency_flag'],
    ['6M DELINQUENCY', 'next_6m_delinquency_flag'],
    ['12M DEFAULT', 'next_12m_default_flag'],
    ['12M PREPAYMENT', 'next_12m_prepayment_flag'],
  ]
  return <div className="prediction-grid">{cards.map(([label, key]) => <div className="prediction-card" key={label}><span>{label}</span><strong>{displayValue(getProb(key))}</strong><MiniD3Bar value={getRawProb(key)} /></div>)}</div>
}

function Explainability({ output }: { output: unknown }) {
  const data = asRecord(output)
  const explanations = asRecord(valueAt(data, 'explanations'))
  const available = entries(explanations).filter(([, value]) => asRecord(value).available === true)
  if (!available.length) return <div className="empty-data">Explainability unavailable for this model.<br /><span>{displayValue(valueAt(data, 'reason', 'message'))}</span></div>
  return <div className="explanation-groups">{available.map(([key, value]) => {
    const explanation = asRecord(value)
    const features = Array.isArray(explanation.top_features) ? explanation.top_features : []
    return <section className="explanation-group" key={key}>
      <div className="explanation-group-head"><strong>{displayValue(explanation.label, key.replaceAll('_', ' '))}</strong><span>{displayValue(explanation.model_used)}</span></div>
      <div className="shap-list">{features.map((item, index) => {
        const row = asRecord(item)
        const contribution = Number(valueAt(row, 'shap_value', 'contribution', 'value') || 0)
        return <div className="shap-row" key={index}><div><span>{displayValue(valueAt(row, 'feature', 'name'), `Feature ${index + 1}`)}</span><b className={contribution < 0 ? 'negative' : ''}>{contribution > 0 ? '+' : ''}{contribution.toFixed(3)}</b></div><div className="shap-track"><i className={contribution < 0 ? 'negative' : ''} style={{ width: `${Math.min(Math.abs(contribution) * 100, 100)}%` }} /></div></div>
      })}</div>
    </section>
  })}</div>
}

function ScenarioPanel({ loanId, initial }: { loanId: string; initial: unknown }) {
  const [scenario, setScenario] = useState('BASE')
  const [result, setResult] = useState(initial)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  async function change(value: string) {
    setScenario(value)
    setLoading(true)
    setError('')
    try { setResult(await runScenario(loanId, value)) } catch (e) { setError(errorText(e)) } finally { setLoading(false) }
  }
  return <div className="scenario-panel"><div className="segmented">{['BASE', 'ADVERSE CREDIT', 'HIGH PREPAYMENT'].map(value => <button key={value} className={scenario === value ? 'active' : ''} onClick={() => change(value)}>{value}</button>)}</div>{loading ? <div className="loading-state"><RefreshCw className="spin" /> Requesting scenario from backend…</div> : error ? <div className="error-box"><AlertTriangle />{error}</div> : <DataGrid data={result} />}</div>
}

function DetailDrawer({ stage, loanId, onClose, onRetry }: DetailDrawerProps) {
  const output = stage.output || {}
  return (
    <div className="drawer-backdrop" onClick={onClose}>
      <aside className="drawer" onClick={(event) => event.stopPropagation()}>
        <div className="drawer-head">
          <div>
            <p className="eyebrow">{stage.key === 'explainability' ? 'SHAP EXPLAINABILITY' : `STAGE ${stageNumbers[stageOrder.indexOf(stage.key)]}`}</p>
            <h2>{stage.label}</h2>
          </div>
          <button className="icon-button" onClick={onClose} aria-label="Close stage details"><X /></button>
        </div>
        <div className="drawer-status"><StatusIcon status={stage.status} /><span>{statusLabel(stage.status)}</span>{stage.executionMs !== undefined && <em>{stage.executionMs} ms</em>}</div>
        {stage.status === 'failed' && <div className="error-box"><XCircle /><div><strong>Stage failed</strong><p>{stage.error || 'The backend did not provide an error message.'}</p></div><button className="text-button" onClick={onRetry}><RefreshCw /> Retry stage</button></div>}
        {stage.key === 'prediction' && stage.status === 'completed' && <PredictionCards output={output} />}
        {stage.key === 'explainability' && stage.status === 'completed' && <><h3>Why did the model make this prediction?</h3><Explainability output={output} /></>}
        {stage.key === 'scenarios' && stage.status === 'completed' && <ScenarioPanel loanId={loanId} initial={output} />}
        {stage.key !== 'prediction' && stage.key !== 'explainability' && stage.key !== 'scenarios' && stage.status === 'completed' && <DataGrid data={output} />}
        {stage.status === 'waiting' && <div className="empty-data">This stage has not completed yet. Run the full analysis to receive backend output.</div>}
      </aside>
    </div>
  )
}

const TIER_ROWS = [
  { key: 'high', label: 'High', color: 'var(--orange)' },
  { key: 'elevated', label: 'Elevated', color: '#e8a040' },
  { key: 'moderate', label: 'Moderate', color: 'var(--teal)' },
  { key: 'low', label: 'Healthy', color: '#0d8b76' },
] as const

function TierBars({ tiers }: { tiers: Record<string, number> }) {
  const max = Math.max(...TIER_ROWS.map((row) => Number(tiers[row.key] || 0)), 1)
  return (
    <div className="tier-bars">
      {TIER_ROWS.map((row) => {
        const value = Number(tiers[row.key] || 0)
        const width = Math.max((value / max) * 100, value > 0 ? 3 : 0)
        return (
          <div className="bar-row" key={row.key}>
            <span>{row.label}</span>
            <div className="bar-track">
              <i style={{ width: `${width}%`, background: row.color }} />
            </div>
            <b>{value.toLocaleString()}</b>
          </div>
        )
      })}
    </div>
  )
}

function OverviewView({ portfolio, loading, onOpenLoan }: { portfolio: PortfolioOverview | null; loading: boolean; onOpenLoan: (id: string) => void }) {
  const tiers = portfolio?.tier_summary || {}
  return (
    <section className="demo-view">
      <div className="section-heading">
        <p className="eyebrow">PORTFOLIO SNAPSHOT</p>
      </div>
      {loading && <div className="loading-state"><RefreshCw className="spin" /> Loading the book…</div>}

      {/* Signal summary row — full width bar chart + loans side by side */}
      <div className="grid-2">
        <LedgerCard as="article" className="tier-card">
          <div className="ledger-card-title">
            <span className="eyebrow">RISK TIER DISTRIBUTION</span>
            <h3>Portfolio signal breakdown</h3>
            <p className="body-text">Count of loans per risk band. High risk loans need immediate review.</p>
          </div>
          <TierBars tiers={tiers} />
          <div className="tier-stats">
            {TIER_ROWS.map((row) => (
              <div key={row.key} className="tier-stat">
                <strong style={{ color: row.color }}>{Number(tiers[row.key] || 0).toLocaleString()}</strong>
                <span>{row.label}</span>
              </div>
            ))}
          </div>
        </LedgerCard>

        <LedgerCard>
          <div className="ledger-card-title">
            <h3>Loans to open first</h3>
            <p className="body-text">Highest-risk accounts. Click one, then run the full analysis.</p>
          </div>
          <div className="watch-list">
            {(portfolio?.top_loans || []).slice(0, 6).map((loan) => {
              const row = asRecord(loan)
              return (
                <button key={String(row.loan_id)} className="watch-row" onClick={() => onOpenLoan(String(row.loan_id))}>
                  <div>
                    <strong>{String(row.loan_id)}</strong>
                    <span>{String(row.credit_band)} credit · {String(row.current_status)} · {String(row.days_past_due)} days late</span>
                  </div>
                  <em className={`pill ${String(row.risk_tier)}`}>{String(row.risk_tier)}</em>
                </button>
              )
            })}
          </div>
        </LedgerCard>
      </div>

      <div className="health-grid-section" style={{ marginTop: '32px' }}>
        <SimpleHealthMap matrix={portfolio?.status_heatmap} />
      </div>
      <div className="grid-2 section-spacing" style={{ alignItems: 'stretch' }}>
        <CreditRiskBars matrix={portfolio?.scenario_heatmap} />
        <MetricsPanel portfolio={portfolio} />
      </div>
    </section>
  )
}
function ComparisonBenchmarkD3({ modelComparison }: { modelComparison: ModelComparison }) {
  const containerRef = useRef<HTMLDivElement>(null)
  
  useEffect(() => {
    if (!containerRef.current || !modelComparison) return
    const container = containerRef.current
    container.innerHTML = ''
    
    const targets = Object.keys(modelComparison.results)
    const margin = { top: 0, right: 40, bottom: 0, left: 60 }
    
    let yOffset = 0
    const totalHeight = targets.length * 120
    
    const width = container.clientWidth
    const svg = d3.select(container).append('svg')
      .attr('width', '100%')
      .attr('height', totalHeight)
      .attr('viewBox', `0 0 ${width} ${totalHeight}`)
      .attr('preserveAspectRatio', 'xMinYMin meet')
      
    const x = d3.scaleLinear().domain([0, 1]).range([margin.left, width - margin.right])
    
    targets.forEach((target) => {
      const winner = modelComparison.winners[target]
      const modelsData = Object.entries(modelComparison.results[target])
        .filter(([key]) => key.endsWith('_test'))
        .map(([key, metric]) => ({ model: key.replace('_test', ''), metric: metric as any }))
      
      const groupHeight = modelsData.length * 28 + 32
      const group = svg.append('g').attr('transform', `translate(0, ${yOffset})`)
      
      group.append('text').attr('x', 0).attr('y', 16)
        .attr('fill', 'var(--text-primary)').attr('font-size', '14px').attr('font-weight', '600')
        .text(target.replaceAll('_', ' ').toUpperCase())
      
      group.append('text').attr('x', width).attr('y', 16)
        .attr('text-anchor', 'end')
        .attr('fill', 'var(--teal)').attr('font-size', '11px').attr('font-family', 'var(--mono)')
        .text(`Best: ${winner.model} · ${winner.test_roc_auc.toFixed(4)}`)
        
      const y = d3.scaleBand().domain(modelsData.map(d => d.model)).range([30, groupHeight - 10]).padding(0.5)
      
      group.selectAll('.bg-bar').data(modelsData).enter().append('rect').attr('class', 'bg-bar')
        .attr('x', margin.left).attr('y', d => y(d.model) || 0)
        .attr('width', width - margin.left - margin.right).attr('height', y.bandwidth())
        .attr('fill', 'var(--border-line-subtle)').attr('rx', 3)
        
      group.selectAll('.bar').data(modelsData).enter().append('rect').attr('class', 'bar')
        .attr('x', margin.left).attr('y', d => y(d.model) || 0)
        .attr('width', d => Math.max(x(d.metric.roc_auc) - margin.left, 4)).attr('height', y.bandwidth())
        .attr('fill', d => d.model === winner.model ? 'var(--teal)' : 'var(--mint)')
        .attr('rx', 3)
        
      group.selectAll('.label').data(modelsData).enter().append('text').attr('class', 'label')
        .attr('x', margin.left - 12).attr('y', d => (y(d.model) || 0) + y.bandwidth() / 2)
        .attr('dy', '0.35em').attr('text-anchor', 'end')
        .attr('fill', 'var(--muted-ink)').attr('font-size', '11px').text(d => d.model)
        
      group.selectAll('.val').data(modelsData).enter().append('text').attr('class', 'val')
        .attr('x', width).attr('y', d => (y(d.model) || 0) + y.bandwidth() / 2)
        .attr('dy', '0.35em').attr('text-anchor', 'end')
        .attr('fill', 'var(--text-primary)').attr('font-size', '11px').attr('font-family', 'var(--mono)')
        .text(d => d.metric.roc_auc.toFixed(4))
        
      yOffset += groupHeight + 20
    })
    
    svg.attr('height', yOffset)
    svg.attr('viewBox', `0 0 ${width} ${yOffset}`)
  }, [modelComparison])

  return <div ref={containerRef} style={{ width: '100%', minHeight: '300px' }} />
}

export default function LoanIntelligenceDashboard() {
  const [loanId, setLoanId] = useState('LN0000298')
  const [result, setResult] = useState<PipelineResponse | null>(null)
  const [survival, setSurvival] = useState<SurvivalResponse | null>(null)
  const [survivalLoading, setSurvivalLoading] = useState(false)
  const [survivalError, setSurvivalError] = useState('')
  const [stages, setStages] = useState<PipelineStage[]>([])
  const [selected, setSelected] = useState<PipelineStage | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [theme, setTheme] = useState<'dark' | 'light'>('dark')
  const [activeNav, setActiveNav] = useState<NavKey>('Overview')
  const [mobileOpen, setMobileOpen] = useState(false)
  const [question, setQuestion] = useState('')
  const [answer, setAnswer] = useState('')
  const [copilotLoading, setCopilotLoading] = useState(false)
  const [portfolio, setPortfolio] = useState<PortfolioOverview | null>(null)
  const [portfolioLoading, setPortfolioLoading] = useState(true)
  const [modelComparison, setModelComparison] = useState<ModelComparison | null>(null)
  const [expandedModelCard, setExpandedModelCard] = useState<ModelCardKey | null>(null)

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
  }, [theme])

  const completed = stages.filter((stage) => stage.status === 'completed').length
  const prediction = useMemo(() => asRecord(stages.find((stage) => stage.key === 'prediction')?.output), [stages])

  useEffect(() => {
    let cancelled = false
    fetchPortfolio(24)
      .then((data) => { if (!cancelled) setPortfolio(data) })
      .catch((e) => { if (!cancelled) setError(errorText(e)) })
      .finally(() => { if (!cancelled) setPortfolioLoading(false) })
    return () => { cancelled = true }
  }, [])

  useEffect(() => {
    fetchModelComparison().then(setModelComparison).catch(() => setModelComparison(null))
  }, [])

  async function run() {
    if (!loanId.trim()) return
    const requestedLoanId = loanId.trim()
    setLoading(true)
    setError('')
    setSelected(null)
    setResult(null)
    setSurvival(null)
    setSurvivalError('')
    setSurvivalLoading(true)
    setActiveNav('Loan Intelligence')
    let currentStages = stageOrder.map((key) => ({ key, label: stageLabels[key], status: 'waiting' as StageStatus }))
    setStages([...currentStages])
    const cumulativeResult: PipelineResponse = { loan_id: requestedLoanId, stages: [] }
    void fetchSurvival(requestedLoanId)
      .then((data) => {
        if (data.status === 'error') {
          setSurvivalError(data.error || 'Survival estimate unavailable')
          return
        }
        setSurvival(data)
      })
      .catch((e) => setSurvivalError(errorText(e)))
      .finally(() => setSurvivalLoading(false))
    try {
      for (let i = 0; i < stageOrder.length; i++) {
        const key = stageOrder[i]
        currentStages[i] = { ...currentStages[i], status: 'running' }
        setStages([...currentStages])
        const data = await retryStage(requestedLoanId, key)
        const stageData = data.stages?.[0]
        if (stageData) {
          currentStages[i] = stageData
          cumulativeResult.stages.push(stageData)
          setResult({ ...data, stages: cumulativeResult.stages })
        } else {
          currentStages[i] = { ...currentStages[i], status: 'completed' }
        }
        setStages([...currentStages])
        if (currentStages[i].status === 'failed') break
        await new Promise((r) => setTimeout(r, 250))
      }
    } catch (e) {
      setError(errorText(e))
    } finally {
      setLoading(false)
    }
  }

  async function retry(stage: StageKey) {
    setSelected(null)
    setError('')
    try {
      const data = await retryStage(loanId, stage)
      setResult(data)
      setStages(normalizeStages(data))
    } catch (e) {
      setError(errorText(e))
    }
  }

  async function ask(questionText = question) {
    if (!questionText.trim()) return
    setQuestion(questionText)
    setCopilotLoading(true)
    try {
      const response = await askCopilot(loanId, questionText)
      setAnswer(displayValue(valueAt(response, 'answer', 'response', 'message')))
    } catch (e) {
      setAnswer(`Backend error: ${errorText(e)}`)
    } finally {
      setCopilotLoading(false)
    }
  }

  function openLoan(id: string) {
    setLoanId(id)
    setActiveNav('Loan Intelligence')
    setMobileOpen(false)
  }

  function toggleModelCard(card: ModelCardKey) {
    setExpandedModelCard((current) => current === card ? null : card)
  }

  function activateModelCard(event: React.KeyboardEvent<HTMLDivElement>, card: ModelCardKey) {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault()
      toggleModelCard(card)
    }
  }

  const reviewerCard = (
    <LedgerCard className="reviewer-card">
      <div className="reviewer-title">
        <div className="reviewer-icon"><Sparkles /></div>
        <div>
          <strong>AI reviewer copilot</strong>
          <span className="eyebrow" style={{marginTop: '4px'}}>Short grounded brief · Human review required</span>
        </div>
      </div>
      <div className="prompt-list">{prompts.map((prompt) => <button key={prompt} onClick={() => ask(prompt)}>{prompt}<ArrowRight /></button>)}</div>
      <div className="chat-input">
        <MessageSquareText />
        <input value={question} onChange={(event) => setQuestion(event.target.value)} onKeyDown={(event) => { if (event.key === 'Enter' && !event.nativeEvent.isComposing && event.keyCode !== 229) ask() }} placeholder="Ask about this loan..." />
        <button onClick={() => ask()} disabled={copilotLoading}>{copilotLoading ? <RefreshCw className="spin" /> : <ArrowRight />}</button>
      </div>
      {answer && <div className="copilot-answer"><span>AI-GENERATED</span><CopilotAnswer text={answer} /></div>}
    </LedgerCard>
  )

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="topbar-inner">
          <a href="#" className="brand">
            <div className="brand-text"><strong>INTAIN</strong></div>
          </a>
          <nav>
            {navItems.map((item) => (
              <button key={item} className={activeNav === item ? 'active' : ''} onClick={() => setActiveNav(item)}>{item}</button>
            ))}
          </nav>
          <div className="topbar-actions">
            <button onClick={() => setTheme(t => t === 'dark' ? 'light' : 'dark')} className="icon-button" style={{ border: 'none' }} aria-label="Toggle theme">
              {theme === 'dark' ? <Sun size={18} /> : <Moon size={18} />}
            </button>
            <button className="mobile-menu" aria-label="Open navigation" onClick={() => setMobileOpen(!mobileOpen)}><Menu /></button>
          </div>
        </div>
      </header>
      {mobileOpen && (
        <div className="mobile-nav">
          {navItems.map((item) => (
            <button key={item} onClick={() => { setActiveNav(item); setMobileOpen(false) }}>{item}</button>
          ))}
        </div>
      )}
      <div className="page-container">
        {activeNav === 'Overview' && <section className="hero">
          <div className="hero-copy-block">
            <p className="eyebrow"><span className="live-dot" /> PORTFOLIO INTELLIGENCE / LIVE SYSTEM</p>
            <h1>See the risk surface<br /><em>before it moves.</em></h1>
            <p className="hero-copy">A decision layer for loan performance: find the pockets that need attention, understand why, and move from signal to review without losing the thread.</p>
            <div className="hero-metadata">
              <div className="hero-metadata-item"><span>BOOK SIZE</span><strong>{portfolio?.total_loans?.toLocaleString() || '—'} LOANS</strong></div>
              <div className="hero-metadata-item"><span>LAST REFRESH</span><strong>LIVE / OUT-OF-TIME</strong></div>
            </div>
          </div>
          <LedgerCard className="hero-pulse"><div className="hero-pulse-head"><span>PORTFOLIO SHAPE</span><b>LIVE</b></div><PortfolioPulse portfolio={portfolio} /><p>Loan concentration by credit band. The shape of the book is the first risk signal.</p></LedgerCard>
        </section>}
        {error && <div className="global-error"><AlertTriangle /> <span>{error}</span></div>}

        {activeNav === 'Overview' && <OverviewView portfolio={portfolio} loading={portfolioLoading} onOpenLoan={openLoan} />}

        {activeNav === 'Loan Intelligence' && (
          <section className="workspace">
            <div className="loan-run-bar">
              <section className="run-panel">
                <div className="run-label"><Search /><div><span>SELECT A LOAN</span><strong>Run full analysis</strong></div></div>
                <div className="search-control">
                  <input value={loanId} onChange={(event) => setLoanId(event.target.value)} onKeyDown={(event) => { if (event.key === 'Enter' && !event.nativeEvent.isComposing && event.keyCode !== 229) run() }} placeholder="Search Loan ID..." aria-label="Loan ID" />
                  <span>⌘ K</span>
                </div>
                <button className="run-button" onClick={run} disabled={loading}>{loading ? <RefreshCw className="spin" /> : <ArrowRight />} {loading ? 'Running pipeline' : 'Run full analysis'}</button>
              </section>
            </div>
            <div className="pipeline-centered">
              <div className="section-heading">
                <div><p className="eyebrow">ORCHESTRATION</p><h2>Live pipeline</h2></div>
                <div className="pipeline-count"><strong>{completed.toString().padStart(2, '0')}</strong><span>/ 08 stages complete</span></div>
              </div>
              <div className="pipeline">
                {stageOrder.map((key, index) => {
                  const stage = stages[index] || { key, label: stageLabels[key], status: 'waiting' as StageStatus }
                  return (
                    <div className="stage-wrap" key={key}>
                      <button className={`stage-card ${stage.status}`} onClick={() => stage.status !== 'waiting' && setSelected(stage)}>
                        <div className="stage-index">{stageNumbers[index]}</div>
                        <StatusIcon status={stage.status} />
                        <div className="stage-copy">
                          <strong>{stage.label}</strong>
                          <span>{statusLabel(stage.status)}{stage.executionMs !== undefined ? ` · ${stage.executionMs} ms` : ''}</span>
                        </div>
                        <ChevronRight className="stage-chevron" />
                      </button>
                      {index < 7 && <div className={`connector ${stage.status === 'completed' ? 'complete' : ''}`}><span /></div>}
                    </div>
                  )
                })}
              </div>
              <div className="pipeline-summary-row">
                <div className="section-heading" style={{ marginBottom: 0 }}>
                  <p className="eyebrow">OUTPUT</p>
                  <h2>Analysis summary</h2>
                </div>
                <LedgerCard className="summary-card" style={{padding: 0}}>
                  {result ? (
                    <>
                      <div className="summary-top"><span>ANALYSIS COMPLETE</span><ShieldCheck /><small>{loanId}</small></div>
                      <div className="summary-risk"><span>MODEL OUTPUT</span><strong>{displayValue((prediction?.next_state as { predicted_state?: string } | undefined)?.predicted_state)}</strong></div>
                      <PredictionCards output={prediction} />
                      <div className="summary-foot">
                        <span>Every value is returned by the inference API</span>
                        <button className="text-button" onClick={() => stages.find((stage) => stage.key === 'prediction') && setSelected(stages.find((stage) => stage.key === 'prediction')!)}>View full analysis <ArrowRight /></button>
                      </div>
                    </>
                  ) : (
                    <div className="summary-empty"><Layers3 /><strong>Awaiting loan analysis</strong><p className="body-text">Select a loan and run the pipeline to populate model outputs.</p></div>
                  )}
                </LedgerCard>
                <SurvivalCard data={survival} loading={survivalLoading} error={survivalError} />
                {reviewerCard}
              </div>
            </div>
          </section>
        )}

        {activeNav === 'Anomaly Detection' && (
          <section className="demo-view">
            <div className="section-heading">
              <p className="eyebrow">EXCEPTIONS</p>
              <h2>Odd loans the rules caught</h2>
              <p className="body-text" style={{marginTop: '8px'}}>These are accounts that look unusual — missing documents, strange balances, or statistical outliers. Open one to see why.</p>
            </div>
            <div className="grid-asymmetric section-spacing">
              <LedgerCard>
                <div className="ledger-card-title">
                  <h3>Why they were flagged</h3>
                </div>
                <div className="metric-list">
                  {(portfolio?.exception_mix || []).map((item) => (
                    <div className="metric-row" key={item.type}><div><strong>{item.type || 'Unusual pattern'}</strong><span>Number of flagged loans in this bucket</span></div><b>{item.count}</b></div>
                  ))}
                </div>
              </LedgerCard>
              <LedgerCard>
                <div className="ledger-card-title">
                  <h3>Start with these</h3>
                </div>
              <div className="watch-list">
                {(portfolio?.top_loans || []).slice(0, 8).map((loan) => {
                  const row = asRecord(loan)
                  return (
                    <button key={String(row.loan_id)} className="watch-row" onClick={() => openLoan(String(row.loan_id))}>
                      <div>
                        <strong>{String(row.loan_id)}</strong>
                        <span>{displayValue(row.exception_type, 'unusual pattern')} · score {displayValue(row.anomaly_score)}</span>
                      </div>
                      <em className={`pill ${String(row.risk_tier)}`}>{String(row.risk_tier)}</em>
                    </button>
                  )
                })}
              </div>
              </LedgerCard>
            </div>
          </section>
        )}

        {activeNav === 'Scenario Analysis' && (
          <section className="demo-view">
            <div className="section-heading">
              <p className="eyebrow">STRESS</p>
              <h2>What if credit gets worse?</h2>
              <p className="body-text" style={{marginTop: '8px'}}>Pick a loan above, run analysis, then tap Base / Adverse / High prepayment to see how that one loan shifts.</p>
            </div>
            <div className="grid-2 section-spacing">
              <CreditRiskBars matrix={portfolio?.scenario_heatmap} />
              <LedgerCard>
                <div className="ledger-card-title">
                  <h3>Stress this loan ({loanId})</h3>
                </div>
                <ScenarioPanel loanId={loanId} initial={{}} />
              </LedgerCard>
            </div>
          </section>
        )}

        {activeNav === 'AI Reviewer' && (
          <section className="demo-view reviewer-page">
            <div className="section-heading">
              <p className="eyebrow">COPILOT</p>
              <h2>Reviewer brief for {loanId}</h2>
            </div>
            {reviewerCard}
          </section>
        )}

        {activeNav === 'Model Cards' && (
          <section className="demo-view model-cards-page">
            <div className="section-heading">
              <p className="eyebrow">MODEL DOCUMENTATION</p>
              <h2>Model Cards</h2>
            </div>
            <div className="model-cards-container section-spacing">
              <LedgerCard className="model-cards-intro">
                <p className="body-text">Comprehensive documentation for each machine learning model used in the Loan Performance Intelligence Engine. Model cards provide transparency into model performance, limitations, and intended use cases.</p>
              </LedgerCard>
              {modelComparison && <LedgerCard className="comparison-panel">
                <div className="comparison-heading"><div><p className="eyebrow">HELD-OUT TEST BENCHMARK</p><h3>Four models, target by target</h3></div><span className="mono-numeral">ROC-AUC · higher is better</span></div>
                <div className="comparison-list">
                  <ComparisonBenchmarkD3 modelComparison={modelComparison} />
                </div>
                <p className="comparison-note">Models use the same time-aware train, validation, and test split. Winners are selected by held-out test ROC-AUC.</p>
              </LedgerCard>}
              <div className="model-cards-grid">
                <LedgerCard className="model-card-item" as="div" role="button" tabIndex={0} onClick={() => toggleModelCard('next_3m_delinquency_lgbm')} onKeyDown={(event) => activateModelCard(event, 'next_3m_delinquency_lgbm')}>
                  <div className="model-card-header">
                    <strong>3-Month Delinquency (XGBoost)</strong>
                    <span className="model-card-metric">ROC-AUC: 0.7206</span>
                  </div>
                  <p>Predicts 3-month delinquency probability using gradient boosting. Strong performance with comprehensive feature engineering.</p>
                  <button className="text-button" onClick={(event) => { event.stopPropagation(); toggleModelCard('next_3m_delinquency_lgbm') }}>
                    {expandedModelCard === 'next_3m_delinquency_lgbm' ? 'Collapse' : 'View Full Card'}
                  </button>
                  {expandedModelCard === 'next_3m_delinquency_lgbm' && (
                    <div className="model-card-details">
                      <div className="model-card-section">
                        <h4>Performance Metrics</h4>
                        <ul>
                          <li>Test ROC-AUC: 0.7206</li>
                          <li>Test PR-AUC: 0.5732</li>
                          <li>Test F1: 0.4604</li>
                          <li>Brier Score: 0.1973</li>
                        </ul>
                      </div>
                      <div className="model-card-section">
                        <h4>Key Features</h4>
                        <ul>
                          <li>Rolling delinquency status windows</li>
                          <li>Balance utilization ratios</li>
                          <li>Credit score × LTV interactions</li>
                          <li>Rate spread calculations</li>
                        </ul>
                      </div>
                      <div className="model-card-section">
                        <h4>Limitations</h4>
                        <ul>
                          <li>Trained on synthetic data</li>
                          <li>3-month horizon may miss longer-term risks</li>
                          <li>Does not incorporate economic indicators</li>
                        </ul>
                      </div>
                    </div>
                  )}
                </LedgerCard>
                <LedgerCard className="model-card-item" as="div" role="button" tabIndex={0} onClick={() => toggleModelCard('next_6m_delinquency_lgbm')} onKeyDown={(event) => activateModelCard(event, 'next_6m_delinquency_lgbm')}>
                  <div className="model-card-header">
                    <strong>6-Month Delinquency (XGBoost)</strong>
                    <span className="model-card-metric">ROC-AUC: 0.7263</span>
                  </div>
                  <p>Extended horizon delinquency prediction with excellent precision-recall tradeoff. Best performing model in the system.</p>
                  <button className="text-button" onClick={(event) => { event.stopPropagation(); toggleModelCard('next_6m_delinquency_lgbm') }}>
                    {expandedModelCard === 'next_6m_delinquency_lgbm' ? 'Collapse' : 'View Full Card'}
                  </button>
                  {expandedModelCard === 'next_6m_delinquency_lgbm' && (
                    <div className="model-card-details">
                      <div className="model-card-section">
                        <h4>Performance Metrics</h4>
                        <ul>
                          <li>Test ROC-AUC: 0.7263</li>
                          <li>Test PR-AUC: 0.6873</li>
                          <li>Test F1: 0.6801</li>
                          <li>Brier Score: 0.2161</li>
                        </ul>
                      </div>
                      <div className="model-card-section">
                        <h4>Key Features</h4>
                        <ul>
                          <li>Extended 6-month rolling windows</li>
                          <li>Seasoning bucket analysis</li>
                          <li>High-risk combo features</li>
                          <li>Historical payment patterns</li>
                        </ul>
                      </div>
                      <div className="model-card-section">
                        <h4>Strengths</h4>
                        <ul>
                          <li>Best performing model in system</li>
                          <li>Excellent precision-recall tradeoff</li>
                          <li>Strong F1 score (0.68)</li>
                        </ul>
                      </div>
                    </div>
                  )}
                </LedgerCard>
                <LedgerCard className="model-card-item" as="div" role="button" tabIndex={0} onClick={() => toggleModelCard('next_12m_default_lr')} onKeyDown={(event) => activateModelCard(event, 'next_12m_default_lr')}>
                  <div className="model-card-header">
                    <strong>12-Month Default (LightGBM)</strong>
                    <span className="model-card-metric">ROC-AUC: 0.7968</span>
                  </div>
                  <p>Baseline model for default prediction. Limited signal in target; used as fallback. Requires feature engineering improvements.</p>
                  <button className="text-button" onClick={(event) => { event.stopPropagation(); toggleModelCard('next_12m_default_lr') }}>
                    {expandedModelCard === 'next_12m_default_lr' ? 'Collapse' : 'View Full Card'}
                  </button>
                  {expandedModelCard === 'next_12m_default_lr' && (
                    <div className="model-card-details">
                      <div className="model-card-section">
                        <h4>Performance Metrics</h4>
                        <ul>
                          <li>Test ROC-AUC: 0.7968</li>
                          <li>Test PR-AUC: 0.2743</li>
                          <li>Test F1: 0.1665</li>
                          <li>Brier Score: 0.1222</li>
                        </ul>
                      </div>
                      <div className="model-card-section">
                        <h4>Known Issues</h4>
                        <ul>
                          <li>Weak signal in target (PR-AUC 0.27)</li>
                          <li>Identical performance to LightGBM</li>
                          <li>Cannot achieve 80% precision with meaningful recall</li>
                        </ul>
                      </div>
                      <div className="model-card-section">
                        <h4>Recommendations</h4>
                        <ul>
                          <li>Investigate target definition and data quality</li>
                          <li>Develop features specifically for default prediction</li>
                          <li>Consider survival analysis for time-to-default</li>
                        </ul>
                      </div>
                    </div>
                  )}
                </LedgerCard>
                <LedgerCard className="model-card-item" as="div" role="button" tabIndex={0} onClick={() => toggleModelCard('next_12m_prepayment_lr')} onKeyDown={(event) => activateModelCard(event, 'next_12m_prepayment_lr')}>
                  <div className="model-card-header">
                    <strong>12-Month Prepayment (Random Forest)</strong>
                    <span className="model-card-metric">ROC-AUC: 0.7364</span>
                  </div>
                  <p>Weak performance on prepayment prediction. Requires interest rate forecasts and economic indicators for improvement.</p>
                  <button className="text-button" onClick={(event) => { event.stopPropagation(); toggleModelCard('next_12m_prepayment_lr') }}>
                    {expandedModelCard === 'next_12m_prepayment_lr' ? 'Collapse' : 'View Full Card'}
                  </button>
                  {expandedModelCard === 'next_12m_prepayment_lr' && (
                    <div className="model-card-details">
                      <div className="model-card-section">
                        <h4>Performance Metrics</h4>
                        <ul>
                          <li>Test ROC-AUC: 0.7364</li>
                          <li>Test PR-AUC: 0.315</li>
                          <li>Test F1: 0.4377</li>
                          <li>Brier Score: 0.2459</li>
                        </ul>
                      </div>
                      <div className="model-card-section">
                        <h4>Known Issues</h4>
                        <ul>
                          <li>Very weak signal (ROC-AUC ~0.57)</li>
                          <li>Not sensitive to interest rate changes</li>
                          <li>Does not capture economic conditions</li>
                        </ul>
                      </div>
                      <div className="model-card-section">
                        <h4>Required Improvements</h4>
                        <ul>
                          <li>Add interest rate forecasts and market indicators</li>
                          <li>Incorporate housing market indices</li>
                          <li>Consider different prepayment definitions</li>
                        </ul>
                      </div>
                    </div>
                  )}
                </LedgerCard>
                <LedgerCard className="model-card-item" as="div" role="button" tabIndex={0} onClick={() => toggleModelCard('next_state_lgbm')} onKeyDown={(event) => activateModelCard(event, 'next_state_lgbm')}>
                  <div className="model-card-header">
                    <strong>Next State (LightGBM)</strong>
                    <span className="model-card-metric">Macro F1: 0.71</span>
                  </div>
                  <p>Multiclass prediction of loan state transitions. Predicts distribution over 7 possible states for comprehensive trajectory analysis.</p>
                  <button className="text-button" onClick={(event) => { event.stopPropagation(); toggleModelCard('next_state_lgbm') }}>
                    {expandedModelCard === 'next_state_lgbm' ? 'Collapse' : 'View Full Card'}
                  </button>
                  {expandedModelCard === 'next_state_lgbm' && (
                    <div className="model-card-details">
                      <div className="model-card-section">
                        <h4>Performance Metrics</h4>
                        <ul>
                          <li>Macro F1: 0.71</li>
                          <li>Weighted F1: 0.68</li>
                          <li>7 possible output states</li>
                        </ul>
                      </div>
                      <div className="model-card-section">
                        <h4>Possible States</h4>
                        <ul>
                          <li>Current, 30DPD, 60DPD, 90DPD</li>
                          <li>Default, Prepaid, Closed</li>
                        </ul>
                      </div>
                      <div className="model-card-section">
                        <h4>Known Limitations</h4>
                        <ul>
                          <li>SHAP explainability currently limited</li>
                          <li>Poor performance on rare state transitions</li>
                          <li>May not capture rapid state changes</li>
                        </ul>
                      </div>
                    </div>
                  )}
                </LedgerCard>
              </div>
            </div>
          </section>
        )}
      </div>
      {selected && <DetailDrawer stage={selected} loanId={loanId} onClose={() => setSelected(null)} onRetry={() => retry(selected.key)} />}
    </main>
  )
}
