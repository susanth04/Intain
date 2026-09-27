'use client'

import { useEffect, useMemo, useState } from 'react'
import { AlertTriangle, ArrowRight, BarChart3, Check, ChevronRight, Circle, Layers3, Menu, MessageSquareText, RefreshCw, Search, ShieldCheck, Sparkles, X, XCircle } from 'lucide-react'
import CopilotAnswer from '@/components/copilot-answer'
import SimpleHealthMap, { CreditRiskBars } from '@/components/heatmap-grid'
import AnimatedGradient from '@/components/ui/animated-gradient'
import { ServiceCard } from '@/components/ui/service-card'
import { askCopilot, asRecord, displayValue, entries, errorText, fetchModelComparison, fetchPortfolio, normalizeStages, retryStage, runScenario, stageLabels, stageOrder, type ModelComparison, type PipelineResponse, type PipelineStage, type PortfolioOverview, type StageKey, type StageStatus, valueAt } from '@/lib/api'

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

function MetricsPanel({ portfolio }: { portfolio: PortfolioOverview | null }) {
  const metrics = asRecord(portfolio?.metrics)
  const highlights = Array.isArray(metrics.highlights) ? metrics.highlights.map(asRecord) : []
  return <div className="panel-card metrics-panel"><div className="metrics-heading"><div><p className="eyebrow">VALIDATION</p><h3>Model performance</h3></div><span>Held-out test set</span></div><p className="panel-copy">Real evaluation results from the strict out-of-time test window. These are model diagnostics, not loan-level predictions.</p><div className="metric-list">{highlights.map((metric, index) => <div className="metric-row" key={index}><div><strong>{displayValue(metric.target)}</strong><span>{displayValue(metric.note)}</span></div><b>{typeof metric.lgbm_test_roc === 'number' ? `ROC-AUC ${(metric.lgbm_test_roc * 100).toFixed(1)}%` : displayValue(metric.lgbm_test_roc)}</b></div>)}</div><div className="metrics-foot"><span>Survival model concordance</span><strong>{typeof metrics.cox_concordance === 'number' ? metrics.cox_concordance.toFixed(4) : '—'}</strong></div></div>
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

function OverviewView({ portfolio, loading, onOpenLoan }: { portfolio: PortfolioOverview | null; loading: boolean; onOpenLoan: (id: string) => void }) {
  const tiers = portfolio?.tier_summary || {}
  return (
    <section className="demo-view">
      <div className="section-heading">
        <div>
          <p className="eyebrow">PORTFOLIO SNAPSHOT</p>
        </div>
      </div>
      {loading && <div className="loading-state"><RefreshCw className="spin" /> Loading the book…</div>}
      <div className="tier-row">
        {[['red', 'Needs attention', 'Accounts requiring immediate review.'], ['blue', 'Watch closely', 'Elevated signals across the book.'], ['default', 'Keep an eye', 'Moderate risk worth monitoring.'], ['gray', 'Healthy', 'No urgent risk signal detected.']].map(([variant, label, description]) => (
          <ServiceCard key={label} title={label} href="#portfolio-signals" variant={variant as 'red' | 'blue' | 'default' | 'gray'} value={tiers[label === 'Needs attention' ? 'high' : label === 'Watch closely' ? 'elevated' : label === 'Keep an eye' ? 'moderate' : 'low'] ?? 0} description={description} />
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
      <MetricsPanel portfolio={portfolio} />
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
  const [activeNav, setActiveNav] = useState<NavKey>('Overview')
  const [mobileOpen, setMobileOpen] = useState(false)
  const [question, setQuestion] = useState('')
  const [answer, setAnswer] = useState('')
  const [copilotLoading, setCopilotLoading] = useState(false)
  const [portfolio, setPortfolio] = useState<PortfolioOverview | null>(null)
  const [portfolioLoading, setPortfolioLoading] = useState(true)
  const [modelComparison, setModelComparison] = useState<ModelComparison | null>(null)
  const [expandedModelCard, setExpandedModelCard] = useState<ModelCardKey | null>(null)

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
    <main className="app-shell">
      <AnimatedGradient className="animated-gradient" />
      <header className="topbar">
        <nav>
          {navItems.map((item) => (
            <button key={item} className={activeNav === item ? 'active' : ''} onClick={() => setActiveNav(item)}>{item}</button>
          ))}
        </nav>
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
          </div>
        </section>
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

        {activeNav === 'Model Cards' && (
          <section className="demo-view model-cards-page">
            <div className="section-heading"><div><p className="eyebrow">MODEL DOCUMENTATION</p><h2>Model Cards</h2></div></div>
            <div className="model-cards-container">
              <div className="model-cards-intro">
                <p>Comprehensive documentation for each machine learning model used in the Loan Performance Intelligence Engine. Model cards provide transparency into model performance, limitations, and intended use cases.</p>
              </div>
              {modelComparison && <div className="comparison-panel">
                <div className="comparison-heading"><div><p className="eyebrow">HELD-OUT TEST BENCHMARK</p><h3>Best model by target</h3></div><span>ROC-AUC winner</span></div>
                <div className="comparison-list">
                  {Object.entries(modelComparison.winners).map(([target, winner]) => (
                    <div className="comparison-row" key={target}>
                      <div><strong>{target.replaceAll('_', ' ')}</strong><span>{winner.models_compared.join(' · ')}</span></div>
                      <b>{winner.model} · {winner.test_roc_auc.toFixed(4)}</b>
                    </div>
                  ))}
                </div>
                <p className="comparison-note">Models use the same time-aware train, validation, and test split. Winners are selected by held-out test ROC-AUC.</p>
              </div>}
              <div className="model-cards-grid">
                <div className="model-card-item" role="button" tabIndex={0} onClick={() => toggleModelCard('next_3m_delinquency_lgbm')} onKeyDown={(event) => activateModelCard(event, 'next_3m_delinquency_lgbm')}>
                  <div className="model-card-header">
                    <strong>3-Month Delinquency (LightGBM)</strong>
                    <span className="model-card-metric">ROC-AUC: 0.7129</span>
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
                          <li>Test ROC-AUC: 0.7129</li>
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
                </div>
                <div className="model-card-item" role="button" tabIndex={0} onClick={() => toggleModelCard('next_6m_delinquency_lgbm')} onKeyDown={(event) => activateModelCard(event, 'next_6m_delinquency_lgbm')}>
                  <div className="model-card-header">
                    <strong>6-Month Delinquency (LightGBM)</strong>
                    <span className="model-card-metric">ROC-AUC: 0.7098</span>
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
                          <li>Test ROC-AUC: 0.7098</li>
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
                </div>
                <div className="model-card-item" role="button" tabIndex={0} onClick={() => toggleModelCard('next_12m_default_lr')} onKeyDown={(event) => activateModelCard(event, 'next_12m_default_lr')}>
                  <div className="model-card-header">
                    <strong>12-Month Default (Logistic Regression)</strong>
                    <span className="model-card-metric">ROC-AUC: 0.6966</span>
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
                          <li>Test ROC-AUC: 0.6966</li>
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
                </div>
                <div className="model-card-item" role="button" tabIndex={0} onClick={() => toggleModelCard('next_12m_prepayment_lr')} onKeyDown={(event) => activateModelCard(event, 'next_12m_prepayment_lr')}>
                  <div className="model-card-header">
                    <strong>12-Month Prepayment (Logistic Regression)</strong>
                    <span className="model-card-metric">ROC-AUC: 0.5652</span>
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
                          <li>Test ROC-AUC: 0.5652 (barely above random)</li>
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
                </div>
                <div className="model-card-item" role="button" tabIndex={0} onClick={() => toggleModelCard('next_state_lgbm')} onKeyDown={(event) => activateModelCard(event, 'next_state_lgbm')}>
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
                </div>
              </div>
            </div>
          </section>
        )}
      </div>
      {selected && <DetailDrawer stage={selected} loanId={loanId} onClose={() => setSelected(null)} onRetry={() => retry(selected.key)} />}
    </main>
  )
}
