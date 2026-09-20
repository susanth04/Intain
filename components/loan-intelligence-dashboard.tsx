'use client'

import { useEffect, useMemo, useState } from 'react'
import { AlertTriangle, ArrowRight, BarChart3, Check, ChevronRight, Circle, Layers3, Menu, MessageSquareText, Presentation, RefreshCw, Search, ShieldCheck, Sparkles, X, XCircle } from 'lucide-react'
import CopilotAnswer from '@/components/copilot-answer'
import SimpleHealthMap, { CreditRiskBars } from '@/components/heatmap-grid'
import { askCopilot, asRecord, displayValue, entries, errorText, fetchPortfolio, normalizeStages, retryStage, runScenario, stageLabels, stageOrder, type PipelineResponse, type PipelineStage, type PortfolioOverview, type StageKey, type StageStatus, valueAt } from '@/lib/api'

type DetailDrawerProps = { stage: PipelineStage; loanId: string; onClose: () => void; onRetry: () => void }
type NavKey = 'Overview' | 'Loan Intelligence' | 'Anomaly Detection' | 'Scenario Analysis' | 'AI Reviewer'

const stageNumbers = ['01', '02', '03', '04', '05', '06', '07', '08']
const prompts = ['Why was this loan flagged?', 'What are the main risk drivers?', 'Explain the model prediction.', 'What data-quality issues were found?']
const navItems: NavKey[] = ['Overview', 'Loan Intelligence', 'Anomaly Detection', 'Scenario Analysis', 'AI Reviewer']

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
  return <div className="data-grid">{rows.map(([key, value]) => <div className="data-row" key={key}><span>{key.replaceAll('_', ' ')}</span><strong>{displayValue(value)}</strong></div>)}</div>
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
  const cards = [
    ['3M DELINQUENCY', 'next_3m_delinquency_flag'],
    ['6M DELINQUENCY', 'next_6m_delinquency_flag'],
    ['12M DEFAULT', 'next_12m_default_flag'],
    ['12M PREPAYMENT', 'next_12m_prepayment_flag'],
  ]
  return <div className="prediction-grid">{cards.map(([label, key]) => <div className="prediction-card" key={label}><span>{label}</span><strong>{displayValue(getProb(key))}</strong></div>)}</div>
}

function Explainability({ output }: { output: unknown }) {
  const data = asRecord(output)
  const shapWrapper = asRecord(valueAt(data, 'shap_values'))
  const shap = valueAt(shapWrapper, 'features')
  if (!shap) return <div className="empty-data">Explainability unavailable for this model.<br /><span>{displayValue(valueAt(data, 'reason', 'message'))}</span></div>
  const rows = Array.isArray(shap) ? shap : Object.entries(shap).map(([feature, contribution]) => ({ feature, contribution }))
  return <div className="shap-list">{rows.map((item, index) => { const row = asRecord(item); const contribution = Number(valueAt(row, 'contribution', 'value', 'shap_value') || 0); return <div className="shap-row" key={index}><div><span>{displayValue(valueAt(row, 'feature', 'name'), `Feature ${index + 1}`)}</span><b className={contribution < 0 ? 'negative' : ''}>{contribution > 0 ? '+' : ''}{contribution || '—'}</b></div><div className="shap-track"><i className={contribution < 0 ? 'negative' : ''} style={{ width: `${Math.min(Math.abs(contribution) * 100, 100)}%` }} /></div></div> })}</div>
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

function OverviewView({ portfolio, loading, onOpenLoan }: { portfolio: PortfolioOverview | null; loading: boolean; onOpenLoan: (id: string) => void }) {
  const tiers = portfolio?.tier_summary || {}
  return (
    <section className="demo-view">
      <div className="section-heading">
        <div>
          <p className="eyebrow">PORTFOLIO SNAPSHOT</p>
          <h2>Can they pay?</h2>
        </div>
        <span className="chip">{portfolio?.total_loans ?? '—'} loans in the book</span>
      </div>
      {loading && <div className="loading-state"><RefreshCw className="spin" /> Loading the book…</div>}
      <div className="tier-row">
        {[['high', 'Needs attention'], ['elevated', 'Watch closely'], ['moderate', 'Keep an eye'], ['low', 'Healthy']].map(([key, label]) => (
          <div className={`tier-card ${key}`} key={key}><span>{label}</span><strong>{tiers[key] ?? 0}</strong></div>
        ))}
      </div>
      <SimpleHealthMap matrix={portfolio?.status_heatmap} />
      <div className="split-panels">
        <CreditRiskBars matrix={portfolio?.scenario_heatmap} />
        <div className="panel-card">
          <h3>Loans to open first</h3>
          <p className="panel-copy">Highest-risk accounts. Click one, then run the full analysis.</p>
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
        </div>
      </div>
    </section>
  )
}

export default function LoanIntelligenceDashboard() {
  const [loanId, setLoanId] = useState('LN0000298')
  const [result, setResult] = useState<PipelineResponse | null>(null)
  const [stages, setStages] = useState<PipelineStage[]>([])
  const [selected, setSelected] = useState<PipelineStage | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [presentation, setPresentation] = useState(false)
  const [activeNav, setActiveNav] = useState<NavKey>('Overview')
  const [mobileOpen, setMobileOpen] = useState(false)
  const [question, setQuestion] = useState('')
  const [answer, setAnswer] = useState('')
  const [copilotLoading, setCopilotLoading] = useState(false)
  const [portfolio, setPortfolio] = useState<PortfolioOverview | null>(null)
  const [portfolioLoading, setPortfolioLoading] = useState(true)

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

  async function run() {
    if (!loanId.trim()) return
    setLoading(true)
    setError('')
    setSelected(null)
    setResult(null)
    setActiveNav('Loan Intelligence')
    let currentStages = stageOrder.map((key) => ({ key, label: stageLabels[key], status: 'waiting' as StageStatus }))
    setStages([...currentStages])
    const cumulativeResult: PipelineResponse = { loan_id: loanId.trim(), stages: [] }
    try {
      for (let i = 0; i < stageOrder.length; i++) {
        const key = stageOrder[i]
        currentStages[i] = { ...currentStages[i], status: 'running' }
        setStages([...currentStages])
        const data = await retryStage(loanId.trim(), key)
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

  const reviewerCard = (
    <div className="reviewer-card">
      <div className="reviewer-title">
        <div className="reviewer-icon"><Sparkles /></div>
        <div>
          <strong>AI reviewer copilot</strong>
          <span>Short grounded brief · Human review required</span>
        </div>
      </div>
      <div className="prompt-list">{prompts.map((prompt) => <button key={prompt} onClick={() => ask(prompt)}>{prompt}<ArrowRight /></button>)}</div>
      <div className="chat-input">
        <MessageSquareText />
        <input value={question} onChange={(event) => setQuestion(event.target.value)} onKeyDown={(event) => { if (event.key === 'Enter' && !event.nativeEvent.isComposing && event.keyCode !== 229) ask() }} placeholder="Ask about this loan..." />
        <button onClick={() => ask()} disabled={copilotLoading}>{copilotLoading ? <RefreshCw className="spin" /> : <ArrowRight />}</button>
      </div>
      {answer && <div className="copilot-answer"><span>AI-GENERATED</span><CopilotAnswer text={answer} /></div>}
    </div>
  )

  return (
    <main className={presentation ? 'app-shell presentation' : 'app-shell'}>
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">I</div>
          <div><strong>INTAIN</strong><span>LOAN PERFORMANCE<br />INTELLIGENCE ENGINE</span></div>
        </div>
        <nav>
          {navItems.map((item) => (
            <button key={item} className={activeNav === item ? 'active' : ''} onClick={() => setActiveNav(item)}>{item}</button>
          ))}
        </nav>
        <button className="presentation-button" onClick={() => setPresentation(!presentation)}><Presentation /> {presentation ? 'Exit presentation' : 'Presentation mode'}</button>
        <button className="mobile-menu" aria-label="Open navigation" onClick={() => setMobileOpen(!mobileOpen)}><Menu /></button>
      </header>
      {mobileOpen && (
        <div className="mobile-nav">
          {navItems.map((item) => (
            <button key={item} onClick={() => { setActiveNav(item); setMobileOpen(false) }}>{item}</button>
          ))}
        </div>
      )}
      <div className="page-wrap">
        <section className="hero">
          <div>
            <p className="eyebrow"><span className="live-dot" /> PORTFOLIO INTELLIGENCE / LIVE SYSTEM</p>
            <h1>Loan intelligence,<br /><em>without the guesswork.</em></h1>
            <p className="hero-copy">See which credit groups are late, then open one loan and walk the full explainable pipeline.</p>
          </div>
          <div className="hero-meta">
            <span>BACKEND STATUS</span>
            <strong><i /> CONNECTED</strong>
            <small>{process.env.NEXT_PUBLIC_API_URL || 'API · relative / local'}</small>
          </div>
        </section>
        <section className="run-panel">
          <div className="run-label"><Search /><div><span>SELECT A LOAN</span><strong>Run full analysis</strong></div></div>
          <div className="search-control">
            <input value={loanId} onChange={(event) => setLoanId(event.target.value)} onKeyDown={(event) => { if (event.key === 'Enter' && !event.nativeEvent.isComposing && event.keyCode !== 229) run() }} placeholder="Search Loan ID..." aria-label="Loan ID" />
            <span>⌘ K</span>
          </div>
          <button className="run-button" onClick={run} disabled={loading}>{loading ? <RefreshCw className="spin" /> : <ArrowRight />} {loading ? 'Running pipeline' : 'Run full analysis'}</button>
        </section>
        {error && <div className="global-error"><AlertTriangle /> <span>{error}</span></div>}

        {activeNav === 'Overview' && <OverviewView portfolio={portfolio} loading={portfolioLoading} onOpenLoan={openLoan} />}

        {activeNav === 'Loan Intelligence' && (
          <section className="workspace">
            <div className="pipeline-column">
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
                        <div className="stage-copy"><strong>{stage.label}</strong><span>{statusLabel(stage.status)}{stage.executionMs !== undefined ? ` · ${stage.executionMs} ms` : ''}</span></div>
                        <ChevronRight className="stage-chevron" />
                      </button>
                      {index < 7 && <div className={`connector ${stage.status === 'completed' ? 'complete' : ''}`}><span /></div>}
                    </div>
                  )
                })}
              </div>
            </div>
            <aside className="summary-column">
              <div className="section-heading"><div><p className="eyebrow">OUTPUT</p><h2>Analysis summary</h2></div><BarChart3 /></div>
              <div className="summary-card">
                {result ? (
                  <>
                    <div className="summary-top"><span>ANALYSIS COMPLETE</span><ShieldCheck /><small>{loanId}</small></div>
                    <div className="summary-risk"><span>MODEL OUTPUT</span><strong>{displayValue((prediction?.next_state as { predicted_state?: string } | undefined)?.predicted_state)}</strong></div>
                    <PredictionCards output={prediction} />
                    <div className="summary-foot">
                      <span>Every value is returned by the inference API</span>
                      <button onClick={() => stages.find((stage) => stage.key === 'prediction') && setSelected(stages.find((stage) => stage.key === 'prediction')!)}>View full analysis <ArrowRight /></button>
                    </div>
                  </>
                ) : (
                  <div className="summary-empty"><Layers3 /><strong>Awaiting loan analysis</strong><p>Select a loan and run the pipeline to populate model outputs.</p></div>
                )}
              </div>
              {reviewerCard}
            </aside>
          </section>
        )}

        {activeNav === 'Anomaly Detection' && (
          <section className="demo-view">
            <div className="section-heading"><div><p className="eyebrow">EXCEPTIONS</p><h2>Odd loans the rules caught</h2></div></div>
            <p className="demo-note">These are accounts that look unusual — missing documents, strange balances, or statistical outliers. Open one to see why.</p>
            <div className="panel-card">
              <h3>Why they were flagged</h3>
              <div className="metric-list">
                {(portfolio?.exception_mix || []).map((item) => (
                  <div className="metric-row" key={item.type}><div><strong>{item.type || 'Unusual pattern'}</strong><span>Number of flagged loans in this bucket</span></div><b>{item.count}</b></div>
                ))}
              </div>
            </div>
            <div className="panel-card">
              <h3>Start with these</h3>
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
            </div>
          </section>
        )}

        {activeNav === 'Scenario Analysis' && (
          <section className="demo-view">
            <div className="section-heading"><div><p className="eyebrow">STRESS</p><h2>What if credit gets worse?</h2></div></div>
            <CreditRiskBars matrix={portfolio?.scenario_heatmap} />
            <p className="demo-note">Pick a loan above, run analysis, then tap Base / Adverse / High prepayment to see how that one loan shifts.</p>
            <div className="panel-card">
              <h3>Stress this loan ({loanId})</h3>
              <ScenarioPanel loanId={loanId} initial={{}} />
            </div>
          </section>
        )}

        {activeNav === 'AI Reviewer' && (
          <section className="demo-view reviewer-page">
            <div className="section-heading"><div><p className="eyebrow">COPILOT</p><h2>Reviewer brief for {loanId}</h2></div></div>
            {reviewerCard}
          </section>
        )}
      </div>
      {selected && <DetailDrawer stage={selected} loanId={loanId} onClose={() => setSelected(null)} onRetry={() => retry(selected.key)} />}
    </main>
  )
}
