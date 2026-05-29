import { useEffect, useState, useRef } from 'react'
import {
  BarChart2,
  Play,
  Square,
  ChevronDown,
  ChevronRight,
  Download,
  Plus,
  CheckCircle,
  XCircle,
  Loader,
  RefreshCw,
} from 'lucide-react'
import { useBenchmarkStore, BenchmarkRunSummary, CaseDetail, RunReport } from '../store/benchmarkStore'
import { useSocketStore } from '../store/socketStore'

// ---------------------------------------------------------------------------
// C.L.A.S.S. column helpers
// ---------------------------------------------------------------------------

const CLASS_COLUMNS = [
  { key: 'accuracy_mean', label: 'Accuracy', format: (v: any) => (v != null ? `${(v * 100).toFixed(1)}%` : '—') },
  { key: 'latency_ms_avg', label: 'Latency (ms)', format: (v: any) => (v != null ? v.toFixed(0) : '—') },
  { key: 'cost_usd', label: 'Cost ($)', format: (v: any) => (v != null ? `$${Number(v).toFixed(4)}` : '—') },
  { key: 'stability_std_dev', label: 'Stability σ', format: (v: any) => (v != null ? v.toFixed(3) : '—') },
  { key: 'security', label: 'Security', format: (v: any) => (v == null ? '—' : v === 'n/a' ? '✓' : `${v} issues`) },
  { key: 'case_count', label: 'Cases', format: (v: any) => v ?? '—' },
]

function scoreColor(val: number): string {
  if (val >= 0.8) return 'text-green-400'
  if (val >= 0.5) return 'text-yellow-400'
  return 'text-red-400'
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function StatusBadge({ status }: { status: string }) {
  const cls: Record<string, string> = {
    starting: 'bg-blue-900 text-blue-300',
    running: 'bg-yellow-900 text-yellow-300',
    completed: 'bg-green-900 text-green-300',
    stopped: 'bg-gray-700 text-gray-300',
    failed: 'bg-red-900 text-red-300',
  }
  return (
    <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${cls[status] ?? 'bg-gray-700 text-gray-300'}`}>
      {status}
    </span>
  )
}

function ProgressBar({ completed, total }: { completed: number; total: number }) {
  const pct = total > 0 ? Math.min(100, Math.round((completed / total) * 100)) : 0
  return (
    <div className="w-full bg-gray-700 rounded-full h-2.5">
      <div
        className="bg-primary-500 h-2.5 rounded-full transition-all duration-300"
        style={{ width: `${pct}%` }}
      />
    </div>
  )
}

// ---------------------------------------------------------------------------
// Config editor
// ---------------------------------------------------------------------------

type ConfigTab = 'select' | 'custom'

function ConfigEditor({ onStart }: { onStart: () => void }) {
  const { configs, selectedConfig, fetchConfigs, getConfig } = useBenchmarkStore()
  const { emit, connected } = useSocketStore()

  const [tab, setTab] = useState<ConfigTab>('select')
  const [selectedName, setSelectedName] = useState('')
  const [rounds, setRounds] = useState<string>('1')
  const [maxCases, setMaxCases] = useState<string>('')
  const [dryRun, setDryRun] = useState(false)
  const [customYaml, setCustomYaml] = useState('')
  const [customError, setCustomError] = useState('')
  const [starting, setStarting] = useState(false)

  const {
    handleBenchmarkRunId,
    handleBenchmarkStarted,
    handleBenchmarkProgress,
    handleBenchmarkComplete,
    handleBenchmarkError,
  } = useBenchmarkStore()
  const { on, off } = useSocketStore()

  useEffect(() => {
    fetchConfigs()
  }, [])

  useEffect(() => {
    if (tab === 'select' && selectedName) {
      getConfig(selectedName)
    }
  }, [selectedName, tab])

  // Wire SocketIO events for the lifetime of this component
  useEffect(() => {
    const onRunId = (data: any) => { handleBenchmarkRunId(data); onStart() }
    const onStarted = (data: any) => handleBenchmarkStarted(data)
    const onProgress = (data: any) => handleBenchmarkProgress(data)
    const onComplete = (data: any) => handleBenchmarkComplete(data)
    const onError = (data: any) => handleBenchmarkError(data)

    on('benchmark_run_id', onRunId)
    on('benchmark_started', onStarted)
    on('benchmark_progress', onProgress)
    on('benchmark_complete', onComplete)
    on('benchmark_error', onError)

    return () => {
      off('benchmark_run_id', onRunId)
      off('benchmark_started', onStarted)
      off('benchmark_progress', onProgress)
      off('benchmark_complete', onComplete)
      off('benchmark_error', onError)
    }
  }, [on, off, handleBenchmarkRunId, handleBenchmarkStarted, handleBenchmarkProgress, handleBenchmarkComplete, handleBenchmarkError, onStart])

  const handleRun = () => {
    if (!connected) return

    const options: Record<string, any> = { dry_run: dryRun }
    if (rounds && parseInt(rounds) > 0) options.rounds = parseInt(rounds)
    if (maxCases && parseInt(maxCases) > 0) options.max_cases = parseInt(maxCases)

    if (tab === 'select') {
      if (!selectedConfig) return
      setStarting(true)
      emit('start_benchmark', { config: selectedConfig, options })
    } else {
      // Parse custom YAML/JSON
      setCustomError('')
      let config: any
      try {
        // Try JSON first, then YAML (simple key:value lines)
        config = JSON.parse(customYaml)
      } catch {
        setCustomError('Invalid JSON. Please provide a valid JSON config.')
        return
      }
      setStarting(true)
      emit('start_benchmark', { config, options })
    }
  }

  const canRun = connected && !starting && (
    (tab === 'select' && Boolean(selectedConfig)) ||
    (tab === 'custom' && customYaml.trim().length > 0)
  )

  return (
    <div className="flex flex-col h-full p-6 space-y-6">
      <div>
        <h2 className="text-xl font-semibold text-gray-100">New Benchmark</h2>
        <p className="text-sm text-gray-400 mt-1">Select an existing config or provide a custom one.</p>
      </div>

      {/* Tabs */}
      <div className="flex border-b border-gray-700">
        {(['select', 'custom'] as ConfigTab[]).map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`px-4 py-2 text-sm font-medium transition-colors ${
              tab === t
                ? 'text-primary-400 border-b-2 border-primary-400'
                : 'text-gray-400 hover:text-gray-200'
            }`}
          >
            {t === 'select' ? 'Select Config' : 'Custom Config (JSON)'}
          </button>
        ))}
      </div>

      {tab === 'select' ? (
        <div className="space-y-4 flex-1 overflow-y-auto">
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-1">Config file</label>
            <select
              value={selectedName}
              onChange={(e) => setSelectedName(e.target.value)}
              className="w-full bg-gray-700 border border-gray-600 rounded-md px-3 py-2 text-gray-100 text-sm"
            >
              <option value="">-- Select a config --</option>
              {configs.map((c) => (
                <option key={c.name} value={c.name}>
                  {c.name} ({c.subject_count} subjects, {c.task_count} tasks)
                </option>
              ))}
            </select>
          </div>

          {selectedConfig && (
            <div className="bg-gray-750 border border-gray-700 rounded-lg p-4 text-sm space-y-2">
              <p className="font-medium text-gray-200">{selectedConfig.name}</p>
              {selectedConfig.subjects && (
                <p className="text-gray-400">
                  Subjects: <span className="text-gray-200">{Object.keys(selectedConfig.subjects).join(', ')}</span>
                </p>
              )}
              {selectedConfig.tasks && (
                <p className="text-gray-400">
                  Tasks: <span className="text-gray-200">{Object.keys(selectedConfig.tasks).join(', ')}</span>
                </p>
              )}
              {selectedConfig.execution && (
                <p className="text-gray-400">
                  Default rounds: <span className="text-gray-200">{selectedConfig.execution.rounds ?? 1}</span>
                </p>
              )}
            </div>
          )}
        </div>
      ) : (
        <div className="flex-1 flex flex-col space-y-2">
          <label className="block text-sm font-medium text-gray-300">
            Paste config as JSON
          </label>
          <textarea
            value={customYaml}
            onChange={(e) => { setCustomYaml(e.target.value); setCustomError('') }}
            placeholder={'{\n  "name": "my-bench",\n  "subjects": {},\n  "tasks": {},\n  "execution": { "rounds": 1 }\n}'}
            className="flex-1 min-h-[200px] bg-gray-700 border border-gray-600 rounded-md p-3 text-gray-100 text-sm font-mono resize-none"
          />
          {customError && <p className="text-red-400 text-xs">{customError}</p>}
        </div>
      )}

      {/* Run options */}
      <div className="space-y-3 border-t border-gray-700 pt-4">
        <p className="text-sm font-medium text-gray-300">Run options</p>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-xs text-gray-400 mb-1">Rounds</label>
            <input
              type="number"
              min={1}
              value={rounds}
              onChange={(e) => setRounds(e.target.value)}
              className="w-full bg-gray-700 border border-gray-600 rounded px-2 py-1.5 text-sm text-gray-100"
            />
          </div>
          <div>
            <label className="block text-xs text-gray-400 mb-1">Max cases per task</label>
            <input
              type="number"
              min={1}
              placeholder="unlimited"
              value={maxCases}
              onChange={(e) => setMaxCases(e.target.value)}
              className="w-full bg-gray-700 border border-gray-600 rounded px-2 py-1.5 text-sm text-gray-100"
            />
          </div>
        </div>
        <label className="flex items-center gap-2 text-sm text-gray-300 cursor-pointer">
          <input
            type="checkbox"
            checked={dryRun}
            onChange={(e) => setDryRun(e.target.checked)}
            className="accent-primary-500"
          />
          Dry run (no outputs written to disk)
        </label>
      </div>

      <button
        onClick={handleRun}
        disabled={!canRun}
        className="flex items-center gap-2 px-5 py-2.5 rounded-lg bg-primary-600 text-white font-medium text-sm hover:bg-primary-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
      >
        {starting ? (
          <><Loader className="w-4 h-4 animate-spin" /> Starting…</>
        ) : (
          <><Play className="w-4 h-4" /> Run Benchmark</>
        )}
      </button>

      {!connected && (
        <p className="text-xs text-yellow-400">Not connected to server. Reconnect to run benchmarks.</p>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Live run view
// ---------------------------------------------------------------------------

function LiveRunView({ onViewResults }: { onViewResults: () => void }) {
  const { activeRun, stopRunHttp } = useBenchmarkStore()
  const { emit } = useSocketStore()
  const logEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [activeRun?.progress_log.length])

  useEffect(() => {
    if (activeRun?.status === 'completed') {
      // Small delay before auto-transitioning so user sees "completed"
      const t = setTimeout(onViewResults, 1200)
      return () => clearTimeout(t)
    }
  }, [activeRun?.status, onViewResults])

  if (!activeRun) {
    return (
      <div className="flex items-center justify-center h-full text-gray-500">
        No active benchmark.
      </div>
    )
  }

  const handleStop = () => {
    if (activeRun.run_id) {
      emit('stop_benchmark', { run_id: activeRun.run_id })
      stopRunHttp(activeRun.run_id)
    }
  }

  const isRunning = activeRun.status === 'running' || activeRun.status === 'starting'

  return (
    <div className="flex flex-col h-full p-6 space-y-4">
      <div className="flex items-start justify-between">
        <div>
          <h2 className="text-xl font-semibold text-gray-100">
            {activeRun.config_name || 'Running…'}
          </h2>
          <div className="flex items-center gap-3 mt-1">
            <StatusBadge status={activeRun.status} />
            <span className="text-sm text-gray-400">
              {activeRun.completed_cases} / {activeRun.total_cases || '?'} cases
            </span>
          </div>
        </div>
        <div className="flex gap-2">
          {isRunning && (
            <button
              onClick={handleStop}
              className="flex items-center gap-2 px-4 py-2 rounded-lg bg-red-700 hover:bg-red-600 text-white text-sm font-medium transition-colors"
            >
              <Square className="w-3.5 h-3.5" /> Stop
            </button>
          )}
          {(activeRun.status === 'completed' || activeRun.status === 'stopped' || activeRun.status === 'failed') && (
            <button
              onClick={onViewResults}
              className="flex items-center gap-2 px-4 py-2 rounded-lg bg-primary-600 hover:bg-primary-500 text-white text-sm font-medium transition-colors"
            >
              View Results
            </button>
          )}
        </div>
      </div>

      <ProgressBar
        completed={activeRun.completed_cases}
        total={activeRun.total_cases || 1}
      />

      {activeRun.error && (
        <div className="bg-red-900/30 border border-red-700 rounded-lg p-3 text-red-300 text-sm">
          {activeRun.error}
        </div>
      )}

      {/* Live case log */}
      <div className="flex-1 overflow-hidden flex flex-col">
        <p className="text-xs font-medium text-gray-400 uppercase tracking-wide mb-2">
          Case Log
        </p>
        <div className="flex-1 overflow-y-auto bg-gray-800 rounded-lg border border-gray-700">
          <table className="w-full text-xs">
            <thead className="sticky top-0 bg-gray-800 border-b border-gray-700">
              <tr>
                <th className="text-left px-3 py-2 text-gray-400 font-medium">Subject</th>
                <th className="text-left px-3 py-2 text-gray-400 font-medium">Task</th>
                <th className="text-left px-3 py-2 text-gray-400 font-medium">Case</th>
                <th className="text-center px-3 py-2 text-gray-400 font-medium">Status</th>
                <th className="text-right px-3 py-2 text-gray-400 font-medium">Score</th>
              </tr>
            </thead>
            <tbody>
              {activeRun.progress_log.map((ev, i) => (
                <tr key={i} className="border-b border-gray-700/50 hover:bg-gray-750">
                  <td className="px-3 py-1.5 text-gray-300">{ev.subject}</td>
                  <td className="px-3 py-1.5 text-gray-400">{ev.task}</td>
                  <td className="px-3 py-1.5 text-gray-500 font-mono">{ev.case_id}</td>
                  <td className="px-3 py-1.5 text-center">
                    {ev.status === 'ok' ? (
                      <CheckCircle className="w-3.5 h-3.5 text-green-400 inline" />
                    ) : (
                      <XCircle className="w-3.5 h-3.5 text-red-400 inline" />
                    )}
                  </td>
                  <td className={`px-3 py-1.5 text-right font-mono ${ev.score != null ? scoreColor(ev.score) : 'text-gray-500'}`}>
                    {ev.score != null ? ev.score.toFixed(2) : '—'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          {activeRun.progress_log.length === 0 && (
            <div className="flex items-center justify-center h-24 text-gray-500 text-sm">
              Waiting for first case…
            </div>
          )}
          <div ref={logEndRef} />
        </div>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Results view
// ---------------------------------------------------------------------------

function SubjectSection({
  subject,
  row,
  stats,
  issues,
  cases,
}: {
  subject: string
  row: Record<string, any>
  stats: Record<string, any>
  issues: any[]
  cases: CaseDetail[]
}) {
  const [expanded, setExpanded] = useState(false)
  const [showCases, setShowCases] = useState(false)

  const passRate = stats?.pass_rate ?? 0
  const accuracy = row?.accuracy_mean ?? 0

  return (
    <div className="border border-gray-700 rounded-lg overflow-hidden">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center gap-3 px-4 py-3 bg-gray-800 hover:bg-gray-750 transition-colors text-left"
      >
        {expanded ? (
          <ChevronDown className="w-4 h-4 text-gray-400 flex-shrink-0" />
        ) : (
          <ChevronRight className="w-4 h-4 text-gray-400 flex-shrink-0" />
        )}
        <span className="font-medium text-gray-200 flex-1">{subject}</span>
        <span className={`text-sm font-mono font-semibold ${scoreColor(accuracy)}`}>
          {(accuracy * 100).toFixed(1)}%
        </span>
        <span className="text-xs text-gray-500 ml-2">
          {row?.case_count ?? 0} cases
        </span>
      </button>

      {expanded && (
        <div className="p-4 space-y-4 bg-gray-800/50">
          {/* Stats cards */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {[
              { label: 'Accuracy', value: `${(accuracy * 100).toFixed(1)}%` },
              { label: 'Pass rate', value: `${(passRate * 100).toFixed(1)}%` },
              { label: 'Latency avg', value: row?.latency_ms_avg != null ? `${row.latency_ms_avg.toFixed(0)} ms` : '—' },
              { label: 'Cost', value: row?.cost_usd != null ? `$${Number(row.cost_usd).toFixed(4)}` : '—' },
            ].map(({ label, value }) => (
              <div key={label} className="bg-gray-800 border border-gray-700 rounded-lg p-3">
                <p className="text-xs text-gray-500 uppercase tracking-wide">{label}</p>
                <p className="text-lg font-semibold text-gray-100 mt-1">{value}</p>
              </div>
            ))}
          </div>

          {/* CI */}
          {stats?.ci_lower != null && (
            <p className="text-xs text-gray-400">
              95% CI: [{(stats.ci_lower * 100).toFixed(1)}%, {(stats.ci_upper * 100).toFixed(1)}%] &nbsp;|&nbsp;
              σ: {(stats.std_dev_score ?? 0).toFixed(3)}
            </p>
          )}

          {/* Issues */}
          {issues && issues.length > 0 && (
            <div>
              <p className="text-xs font-medium text-gray-400 uppercase tracking-wide mb-2">
                Issues ({issues.length})
              </p>
              <div className="space-y-1 max-h-40 overflow-y-auto">
                {issues.map((issue, i) => (
                  <div key={i} className="flex items-start gap-2 text-xs bg-gray-800 rounded px-3 py-1.5">
                    <span className={`font-medium ${issue.severity === 'error' ? 'text-red-400' : 'text-yellow-400'}`}>
                      [{issue.severity}]
                    </span>
                    <span className="text-gray-400">{issue.analyzer}:</span>
                    <span className="text-gray-300 flex-1">{issue.message}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Per-case detail */}
          {cases && cases.length > 0 && (
            <div>
              <button
                onClick={() => setShowCases(!showCases)}
                className="text-xs text-primary-400 hover:text-primary-300 flex items-center gap-1 mb-2"
              >
                {showCases ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
                {showCases ? 'Hide' : 'Show'} {cases.length} case details
              </button>
              {showCases && (
                <div className="overflow-x-auto">
                  <table className="w-full text-xs border border-gray-700 rounded">
                    <thead className="bg-gray-800">
                      <tr>
                        <th className="text-left px-2 py-1.5 text-gray-400">Case</th>
                        <th className="text-left px-2 py-1.5 text-gray-400">Task</th>
                        <th className="text-center px-2 py-1.5 text-gray-400">Round</th>
                        <th className="text-center px-2 py-1.5 text-gray-400">Score</th>
                        <th className="text-center px-2 py-1.5 text-gray-400">Passed</th>
                        <th className="text-center px-2 py-1.5 text-gray-400">Issues</th>
                      </tr>
                    </thead>
                    <tbody>
                      {cases.map((c, i) => (
                        <tr key={i} className="border-t border-gray-700/50">
                          <td className="px-2 py-1.5 font-mono text-gray-400">{c.case_id}</td>
                          <td className="px-2 py-1.5 text-gray-400">{c.task_type}</td>
                          <td className="px-2 py-1.5 text-center text-gray-400">{c.round_num}</td>
                          <td className={`px-2 py-1.5 text-center font-mono ${c.score != null ? scoreColor(c.score) : 'text-gray-500'}`}>
                            {c.score != null ? c.score.toFixed(2) : '—'}
                          </td>
                          <td className="px-2 py-1.5 text-center">
                            {c.passed == null ? '—' : c.passed ? (
                              <CheckCircle className="w-3.5 h-3.5 text-green-400 inline" />
                            ) : (
                              <XCircle className="w-3.5 h-3.5 text-red-400 inline" />
                            )}
                          </td>
                          <td className="px-2 py-1.5 text-center text-gray-400">{c.issue_count}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function ResultsView({
  report,
  casesBySubject,
  runName,
}: {
  report: RunReport
  casesBySubject: Record<string, CaseDetail[]>
  runName?: string
}) {
  const [sortCol, setSortCol] = useState<string>('accuracy_mean')
  const [sortAsc, setSortAsc] = useState(false)

  const classReport = report.class_report ?? {}
  const perSubjectStats = report.per_subject_stats ?? {}
  const issuesBySubject = report.issues_by_subject ?? {}

  const rows = Object.values(classReport) as Array<Record<string, any>>
  const sorted = [...rows].sort((a, b) => {
    const av = a[sortCol] ?? 0
    const bv = b[sortCol] ?? 0
    if (typeof av === 'string') return sortAsc ? av.localeCompare(bv) : bv.localeCompare(av)
    return sortAsc ? av - bv : bv - av
  })

  const handleSort = (key: string) => {
    if (sortCol === key) setSortAsc(!sortAsc)
    else { setSortCol(key); setSortAsc(false) }
  }

  const exportCSV = () => {
    if (rows.length === 0) return
    const keys = Object.keys(rows[0])
    const header = keys.join(',')
    const lines = rows.map((r) => keys.map((k) => JSON.stringify(r[k] ?? '')).join(','))
    const csv = [header, ...lines].join('\n')
    const blob = new Blob([csv], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `class_report_${report.config_name ?? 'eval'}.csv`
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="flex flex-col h-full p-6 space-y-6 overflow-y-auto">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h2 className="text-xl font-semibold text-gray-100">
            {report.config_name ?? runName ?? 'Benchmark Results'}
          </h2>
          {report.generated_at && (
            <p className="text-sm text-gray-400 mt-0.5">
              {new Date(report.generated_at).toLocaleString()}
            </p>
          )}
        </div>
        <button
          onClick={exportCSV}
          disabled={rows.length === 0}
          className="flex items-center gap-2 px-3 py-1.5 text-sm rounded-lg bg-gray-700 hover:bg-gray-600 text-gray-300 disabled:opacity-40 transition-colors"
        >
          <Download className="w-4 h-4" /> Export CSV
        </button>
      </div>

      {/* C.L.A.S.S. table */}
      {rows.length > 0 ? (
        <div className="overflow-x-auto rounded-lg border border-gray-700">
          <table className="w-full text-sm">
            <thead className="bg-gray-800 border-b border-gray-700">
              <tr>
                <th className="text-left px-4 py-3 text-gray-300 font-medium">Subject</th>
                {CLASS_COLUMNS.map((col) => (
                  <th
                    key={col.key}
                    onClick={() => handleSort(col.key)}
                    className="text-right px-4 py-3 text-gray-300 font-medium cursor-pointer hover:text-gray-100 select-none whitespace-nowrap"
                  >
                    {col.label}
                    {sortCol === col.key && (
                      <span className="ml-1 text-primary-400">{sortAsc ? '↑' : '↓'}</span>
                    )}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {sorted.map((row, i) => (
                <tr key={i} className="border-b border-gray-700/50 hover:bg-gray-800/50">
                  <td className="px-4 py-2.5 text-gray-200 font-medium">{row.subject}</td>
                  {CLASS_COLUMNS.map((col) => (
                    <td
                      key={col.key}
                      className={`px-4 py-2.5 text-right font-mono ${
                        col.key === 'accuracy_mean' && row[col.key] != null
                          ? scoreColor(row[col.key])
                          : 'text-gray-300'
                      }`}
                    >
                      {col.format(row[col.key])}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="bg-gray-800 rounded-lg border border-gray-700 p-6 text-center text-gray-500 text-sm">
          No class report data available.
        </div>
      )}

      {/* Per-subject sections */}
      {Object.keys(classReport).length > 0 && (
        <div className="space-y-3">
          <p className="text-sm font-medium text-gray-400 uppercase tracking-wide">Per-subject details</p>
          {Object.entries(classReport).map(([subject, row]) => (
            <SubjectSection
              key={subject}
              subject={subject}
              row={row}
              stats={perSubjectStats[subject] ?? {}}
              issues={issuesBySubject[subject] ?? []}
              cases={casesBySubject[subject] ?? []}
            />
          ))}
        </div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Run sidebar item
// ---------------------------------------------------------------------------

function RunItem({
  run,
  selected,
  onClick,
}: {
  run: BenchmarkRunSummary
  selected: boolean
  onClick: () => void
}) {
  return (
    <button
      onClick={onClick}
      className={`w-full text-left px-3 py-2.5 rounded-lg transition-colors ${
        selected ? 'bg-primary-700/30 border border-primary-700' : 'hover:bg-gray-750 border border-transparent'
      }`}
    >
      <p className="text-sm font-medium text-gray-200 truncate">{run.config_name || run.name}</p>
      <p className="text-xs text-gray-500 truncate mt-0.5">{run.name}</p>
      <div className="flex items-center gap-2 mt-1">
        <span className="text-xs text-gray-600">
          {run.generated_at ? new Date(run.generated_at).toLocaleDateString() : '—'}
        </span>
        <span className="text-xs text-gray-600">·</span>
        <span className="text-xs text-gray-600">{run.case_count} cases</span>
      </div>
    </button>
  )
}

// ---------------------------------------------------------------------------
// Main Benchmarks page
// ---------------------------------------------------------------------------

type View = 'idle' | 'config' | 'live' | 'results'

export default function Benchmarks() {
  const {
    runs,
    selectedRunDetail,
    activeRun,
    fetchRuns,
    getRunDetail,
    clearSelectedRunDetail,
  } = useBenchmarkStore()

  const [view, setView] = useState<View>('idle')
  const [selectedRunPath, setSelectedRunPath] = useState<string | null>(null)
  const { on, off } = useSocketStore()

  useEffect(() => {
    fetchRuns()
  }, [])

  // Wire up SocketIO handlers for page-level events (live view transitions)
  useEffect(() => {
    const onProgress = () => { if (view !== 'live') setView('live') }
    const onComplete = () => { /* auto-transition handled inside LiveRunView */ }
    const onError = () => { /* keep live view so user sees the error */ }

    on('benchmark_progress', onProgress)
    on('benchmark_complete', onComplete)
    on('benchmark_error', onError)

    return () => {
      off('benchmark_progress', onProgress)
      off('benchmark_complete', onComplete)
      off('benchmark_error', onError)
    }
  }, [view, on, off])

  const handleSelectRun = async (run: BenchmarkRunSummary) => {
    setSelectedRunPath(run.path)
    await getRunDetail(run.path)
    setView('results')
  }

  const handleNewBenchmark = () => {
    clearSelectedRunDetail()
    setSelectedRunPath(null)
    setView('config')
  }

  const handleStarted = () => {
    setView('live')
  }

  const handleViewResults = () => {
    setView('results')
    // Refresh past runs list after a completed run
    fetchRuns()
  }

  // Determine which report/cases to display in Results view
  const displayReport =
    view === 'results' && activeRun?.report
      ? activeRun.report
      : selectedRunDetail?.report ?? null

  const displayCases =
    view === 'results' && activeRun?.report
      ? {}  // live run cases come from progress_log, not a detail load
      : selectedRunDetail?.cases_by_subject ?? {}

  return (
    <div className="flex h-full bg-gray-900 text-gray-100">
      {/* ── Left sidebar ── */}
      <div className="w-64 flex-shrink-0 bg-gray-800 border-r border-gray-700 flex flex-col">
        <div className="p-4 border-b border-gray-700 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <BarChart2 className="w-5 h-5 text-primary-400" />
            <span className="font-semibold text-gray-100">Benchmarks</span>
          </div>
          <button
            onClick={() => fetchRuns()}
            className="text-gray-400 hover:text-gray-200 transition-colors"
            title="Refresh runs"
          >
            <RefreshCw className="w-4 h-4" />
          </button>
        </div>

        <div className="p-3 border-b border-gray-700">
          <button
            onClick={handleNewBenchmark}
            className="w-full flex items-center justify-center gap-2 px-3 py-2 rounded-lg bg-primary-600 hover:bg-primary-500 text-white text-sm font-medium transition-colors"
          >
            <Plus className="w-4 h-4" /> New Benchmark
          </button>
        </div>

        {/* Active run indicator */}
        {activeRun && (
          <div
            className="mx-3 mt-3 px-3 py-2.5 rounded-lg bg-yellow-900/30 border border-yellow-700 cursor-pointer hover:bg-yellow-900/50 transition-colors"
            onClick={() => setView(activeRun.status === 'completed' || activeRun.status === 'stopped' || activeRun.status === 'failed' ? 'results' : 'live')}
          >
            <div className="flex items-center gap-2">
              {(activeRun.status === 'running' || activeRun.status === 'starting') && (
                <Loader className="w-3.5 h-3.5 text-yellow-400 animate-spin" />
              )}
              <span className="text-xs font-medium text-yellow-300 truncate">
                {activeRun.config_name || 'Active run'}
              </span>
            </div>
            <div className="mt-1">
              <ProgressBar completed={activeRun.completed_cases} total={activeRun.total_cases || 1} />
            </div>
            <div className="flex justify-between mt-1">
              <StatusBadge status={activeRun.status} />
              <span className="text-xs text-yellow-500">{activeRun.completed_cases}/{activeRun.total_cases || '?'}</span>
            </div>
          </div>
        )}

        {/* Past runs list */}
        <div className="flex-1 overflow-y-auto p-3 space-y-1">
          {runs.length === 0 ? (
            <p className="text-xs text-gray-500 text-center mt-4">No past runs found.</p>
          ) : (
            runs.map((run) => (
              <RunItem
                key={run.path}
                run={run}
                selected={selectedRunPath === run.path}
                onClick={() => handleSelectRun(run)}
              />
            ))
          )}
        </div>
      </div>

      {/* ── Main area ── */}
      <div className="flex-1 overflow-hidden">
        {view === 'idle' && (
          <div className="flex flex-col items-center justify-center h-full text-center space-y-4 text-gray-500">
            <BarChart2 className="w-16 h-16 text-gray-700" />
            <p className="text-lg font-medium text-gray-400">No benchmark selected</p>
            <p className="text-sm">Start a new benchmark or select a past run from the sidebar.</p>
            <button
              onClick={handleNewBenchmark}
              className="flex items-center gap-2 px-5 py-2.5 rounded-lg bg-primary-600 hover:bg-primary-500 text-white font-medium text-sm transition-colors"
            >
              <Plus className="w-4 h-4" /> New Benchmark
            </button>
          </div>
        )}

        {view === 'config' && (
          <ConfigEditor onStart={handleStarted} />
        )}

        {view === 'live' && (
          <LiveRunView onViewResults={handleViewResults} />
        )}

        {view === 'results' && displayReport && (
          <ResultsView
            report={displayReport}
            casesBySubject={displayCases}
            runName={selectedRunPath?.split('/').pop() ?? selectedRunPath?.split('\\').pop()}
          />
        )}

        {view === 'results' && !displayReport && (
          <div className="flex items-center justify-center h-full text-gray-500 text-sm">
            No report data available.
          </div>
        )}
      </div>
    </div>
  )
}
