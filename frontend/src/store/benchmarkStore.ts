/**
 * Benchmark store — manages eval configs, past run listings, and live
 * benchmark execution state for the Benchmarks tab.
 */

import { create } from 'zustand'

// ---------------------------------------------------------------------------
// Interfaces
// ---------------------------------------------------------------------------

export interface BenchmarkConfig {
  name: string
  path: string
  subject_count: number
  task_count: number
  subject_names: string[]
  task_names: string[]
}

export interface BenchmarkRunSummary {
  name: string
  path: string
  config_name: string
  generated_at: string
  case_count: number
  subject_names: string[]
}

export interface ProgressEvent {
  run_id: string
  round_num: number
  subject: string
  task: string
  case_id: string
  status: 'ok' | 'error'
  score: number | null
  passed: boolean | null
  completed: number
  total: number
}

export interface CaseDetail {
  subject_name: string
  case_id: string
  round_num: number
  task_type: string
  output: string
  score: number | null
  passed: boolean | null
  error: string | null
  issue_count: number
}

export interface RunReport {
  config_name?: string
  generated_at?: string
  class_report?: Record<string, Record<string, any>>
  per_subject_stats?: Record<string, Record<string, number>>
  issues_by_subject?: Record<
    string,
    Array<{ analyzer: string; code: string; message: string; severity: string }>
  >
}

export interface RunDetail {
  report: RunReport
  cases_by_subject: Record<string, CaseDetail[]>
}

export interface ActiveRun {
  run_id: string
  config_name: string
  status: 'starting' | 'running' | 'completed' | 'stopped' | 'failed'
  started_at?: string
  total_cases: number
  completed_cases: number
  progress_log: ProgressEvent[]
  report?: RunReport
  error?: string
}

// ---------------------------------------------------------------------------
// State shape
// ---------------------------------------------------------------------------

interface BenchmarkState {
  /** Available eval config files. */
  configs: BenchmarkConfig[]
  /** Full parsed YAML for the currently selected config (for the editor). */
  selectedConfig: Record<string, any> | null
  /** Metadata list of past/completed benchmark runs. */
  runs: BenchmarkRunSummary[]
  /** Detailed results for a run selected from the history list. */
  selectedRunDetail: RunDetail | null
  /** Live state for the benchmark currently being executed. */
  activeRun: ActiveRun | null
  loading: boolean
  error: string | null

  // ----- REST actions -------------------------------------------------------

  /** Fetch the list of available eval configs. */
  fetchConfigs: () => Promise<void>
  /** Load a specific config's full YAML by name or path. */
  getConfig: (name: string) => Promise<void>
  /** Fetch the list of past benchmark run directories. */
  fetchRuns: () => Promise<void>
  /** Load the full report + case details for a past run. */
  getRunDetail: (runDir: string) => Promise<void>
  /** Send stop signal via REST (fallback when SocketIO not connected). */
  stopRunHttp: (runId: string) => Promise<void>

  // ----- SocketIO event handlers --------------------------------------------

  /** Called when server confirms the run_id (just after start_benchmark emit). */
  handleBenchmarkRunId: (data: { run_id: string }) => void
  /** Called when the background thread emits benchmark_started. */
  handleBenchmarkStarted: (data: {
    run_id: string
    config_name: string
    started_at: string
    total_cases: number
  }) => void
  /** Called for each completed case during the run. */
  handleBenchmarkProgress: (data: ProgressEvent) => void
  /** Called when the run finishes successfully. */
  handleBenchmarkComplete: (data: {
    run_id: string
    report: RunReport
    case_count: number
  }) => void
  /** Called when the run encounters an error or is stopped. */
  handleBenchmarkError: (data: { run_id: string; error: string }) => void

  // ----- UI helpers ---------------------------------------------------------

  clearActiveRun: () => void
  clearSelectedRunDetail: () => void
  clearSelectedConfig: () => void
}

// ---------------------------------------------------------------------------
// Store
// ---------------------------------------------------------------------------

export const useBenchmarkStore = create<BenchmarkState>((set, get) => ({
  configs: [],
  selectedConfig: null,
  runs: [],
  selectedRunDetail: null,
  activeRun: null,
  loading: false,
  error: null,

  // -------------------------------------------------------------------------
  // REST actions
  // -------------------------------------------------------------------------

  fetchConfigs: async () => {
    set({ loading: true, error: null })
    try {
      const resp = await fetch('/api/v1/eval/configs')
      if (!resp.ok) throw new Error('Failed to fetch eval configs')
      const data = await resp.json()
      set({ configs: data.configs, loading: false })
    } catch (err: any) {
      set({ error: err.message, loading: false })
    }
  },

  getConfig: async (name: string) => {
    set({ loading: true, error: null })
    try {
      const resp = await fetch(`/api/v1/eval/configs/${encodeURIComponent(name)}`)
      if (!resp.ok) throw new Error(`Config "${name}" not found`)
      const data = await resp.json()
      set({ selectedConfig: data.config, loading: false })
    } catch (err: any) {
      set({ error: err.message, loading: false })
    }
  },

  fetchRuns: async () => {
    set({ loading: true, error: null })
    try {
      const resp = await fetch('/api/v1/eval/runs')
      if (!resp.ok) throw new Error('Failed to fetch benchmark runs')
      const data = await resp.json()
      set({ runs: data.runs, loading: false })
    } catch (err: any) {
      set({ error: err.message, loading: false })
    }
  },

  getRunDetail: async (runDir: string) => {
    set({ loading: true, error: null })
    try {
      const resp = await fetch(
        `/api/v1/eval/runs/detail?run_dir=${encodeURIComponent(runDir)}`
      )
      if (!resp.ok) throw new Error(`Run not found: ${runDir}`)
      const data: RunDetail = await resp.json()
      set({ selectedRunDetail: data, loading: false })
    } catch (err: any) {
      set({ error: err.message, loading: false })
    }
  },

  stopRunHttp: async (runId: string) => {
    try {
      const resp = await fetch(`/api/v1/eval/runs/active/${runId}`, {
        method: 'DELETE',
      })
      if (!resp.ok) {
        const body = await resp.json().catch(() => ({}))
        throw new Error(body.error ?? 'Failed to stop run')
      }
    } catch (err: any) {
      set({ error: err.message })
    }
  },

  // -------------------------------------------------------------------------
  // SocketIO event handlers
  // -------------------------------------------------------------------------

  handleBenchmarkRunId: ({ run_id }) => {
    // Server acknowledged the start — optimistically create the active run record.
    if (!get().activeRun) {
      set({
        activeRun: {
          run_id,
          config_name: '',
          status: 'starting',
          total_cases: 0,
          completed_cases: 0,
          progress_log: [],
        },
      })
    }
  },

  handleBenchmarkStarted: ({ run_id, config_name, started_at, total_cases }) => {
    set((state) => ({
      activeRun: {
        ...(state.activeRun ?? {
          run_id,
          progress_log: [],
          completed_cases: 0,
        }),
        run_id,
        config_name,
        status: 'running',
        started_at,
        total_cases,
      },
    }))
  },

  handleBenchmarkProgress: (event: ProgressEvent) => {
    set((state) => {
      if (!state.activeRun || state.activeRun.run_id !== event.run_id)
        return state
      return {
        activeRun: {
          ...state.activeRun,
          status: 'running',
          completed_cases: event.completed,
          total_cases: event.total || state.activeRun.total_cases,
          progress_log: [...state.activeRun.progress_log, event],
        },
      }
    })
  },

  handleBenchmarkComplete: ({ run_id, report, case_count }) => {
    set((state) => {
      if (!state.activeRun || state.activeRun.run_id !== run_id) return state
      return {
        activeRun: {
          ...state.activeRun,
          status: 'completed',
          completed_cases: case_count,
          report,
        },
      }
    })
  },

  handleBenchmarkError: ({ run_id, error }) => {
    set((state) => {
      if (!state.activeRun || state.activeRun.run_id !== run_id) return state
      const isStopped = error.toLowerCase().includes('stopped')
      return {
        activeRun: {
          ...state.activeRun,
          status: isStopped ? 'stopped' : 'failed',
          error,
        },
      }
    })
  },

  // -------------------------------------------------------------------------
  // UI helpers
  // -------------------------------------------------------------------------

  clearActiveRun: () => set({ activeRun: null }),
  clearSelectedRunDetail: () => set({ selectedRunDetail: null }),
  clearSelectedConfig: () => set({ selectedConfig: null }),
}))
