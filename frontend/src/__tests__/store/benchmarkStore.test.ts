import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import {
  useBenchmarkStore,
  BenchmarkConfig,
  BenchmarkRunSummary,
  ProgressEvent,
} from '../../store/benchmarkStore'

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const initialState = {
  configs: [],
  selectedConfig: null,
  runs: [],
  selectedRunDetail: null,
  activeRun: null,
  loading: false,
  error: null,
}

const mockConfig: BenchmarkConfig = {
  name: 'model_prices',
  path: '/configs/eval/model_prices.yaml',
  subject_count: 2,
  task_count: 3,
  subject_names: ['llama3', 'gpt4o'],
  task_names: ['code', 'reason', 'summarize'],
}

const mockRun: BenchmarkRunSummary = {
  name: 'run_20250101',
  path: '/results/run_20250101',
  config_name: 'model_prices',
  generated_at: '2025-01-01T12:00:00',
  case_count: 20,
  subject_names: ['llama3'],
}

beforeEach(() => {
  useBenchmarkStore.setState(initialState)
  vi.restoreAllMocks()
})

afterEach(() => {
  vi.restoreAllMocks()
})

// ---------------------------------------------------------------------------
// fetchConfigs
// ---------------------------------------------------------------------------

describe('fetchConfigs', () => {
  it('stores configs on success', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(JSON.stringify({ configs: [mockConfig] }), { status: 200 })
    )
    await useBenchmarkStore.getState().fetchConfigs()
    expect(useBenchmarkStore.getState().configs).toEqual([mockConfig])
    expect(useBenchmarkStore.getState().loading).toBe(false)
    expect(useBenchmarkStore.getState().error).toBeNull()
  })

  it('sets error on HTTP failure', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response('', { status: 500 })
    )
    await useBenchmarkStore.getState().fetchConfigs()
    expect(useBenchmarkStore.getState().error).toBeTruthy()
    expect(useBenchmarkStore.getState().loading).toBe(false)
  })

  it('sets error on network failure', async () => {
    vi.spyOn(globalThis, 'fetch').mockRejectedValue(new Error('offline'))
    await useBenchmarkStore.getState().fetchConfigs()
    expect(useBenchmarkStore.getState().error).toBe('offline')
  })
})

// ---------------------------------------------------------------------------
// getConfig
// ---------------------------------------------------------------------------

describe('getConfig', () => {
  it('stores selected config on success', async () => {
    const rawConfig = { name: 'model_prices', subjects: {}, tasks: {} }
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(JSON.stringify({ config: rawConfig }), { status: 200 })
    )
    await useBenchmarkStore.getState().getConfig('model_prices')
    expect(useBenchmarkStore.getState().selectedConfig).toEqual(rawConfig)
  })

  it('sets error on 404', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response('', { status: 404 })
    )
    await useBenchmarkStore.getState().getConfig('missing')
    expect(useBenchmarkStore.getState().error).toBeTruthy()
  })
})

// ---------------------------------------------------------------------------
// fetchRuns
// ---------------------------------------------------------------------------

describe('fetchRuns', () => {
  it('stores runs on success', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(JSON.stringify({ runs: [mockRun] }), { status: 200 })
    )
    await useBenchmarkStore.getState().fetchRuns()
    expect(useBenchmarkStore.getState().runs).toEqual([mockRun])
    expect(useBenchmarkStore.getState().loading).toBe(false)
  })

  it('stores empty array when no runs', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(JSON.stringify({ runs: [] }), { status: 200 })
    )
    await useBenchmarkStore.getState().fetchRuns()
    expect(useBenchmarkStore.getState().runs).toEqual([])
  })
})

// ---------------------------------------------------------------------------
// getRunDetail
// ---------------------------------------------------------------------------

describe('getRunDetail', () => {
  it('stores run detail on success', async () => {
    const detail = {
      report: { config_name: 'cfg1', generated_at: '2025-01-01T00:00:00' },
      cases_by_subject: { sub_a: [] },
    }
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(JSON.stringify(detail), { status: 200 })
    )
    await useBenchmarkStore.getState().getRunDetail('/results/run_001')
    expect(useBenchmarkStore.getState().selectedRunDetail?.report.config_name).toBe('cfg1')
  })

  it('sets error on 404', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(JSON.stringify({ error: 'not found' }), { status: 404 })
    )
    await useBenchmarkStore.getState().getRunDetail('/results/missing')
    expect(useBenchmarkStore.getState().error).toBeTruthy()
  })
})

// ---------------------------------------------------------------------------
// SocketIO event handlers
// ---------------------------------------------------------------------------

describe('handleBenchmarkRunId', () => {
  it('creates activeRun with starting status', () => {
    useBenchmarkStore.getState().handleBenchmarkRunId({ run_id: 'run-abc' })
    const { activeRun } = useBenchmarkStore.getState()
    expect(activeRun).not.toBeNull()
    expect(activeRun?.run_id).toBe('run-abc')
    expect(activeRun?.status).toBe('starting')
  })

  it('does not overwrite an existing activeRun', () => {
    useBenchmarkStore.setState({
      activeRun: {
        run_id: 'existing',
        config_name: 'cfg',
        status: 'running',
        total_cases: 10,
        completed_cases: 3,
        progress_log: [],
      },
    })
    useBenchmarkStore.getState().handleBenchmarkRunId({ run_id: 'new-run' })
    // Should not overwrite since activeRun is already set
    expect(useBenchmarkStore.getState().activeRun?.run_id).toBe('existing')
  })
})

describe('handleBenchmarkStarted', () => {
  it('transitions status to running and stores metadata', () => {
    useBenchmarkStore.setState({
      activeRun: {
        run_id: 'r1',
        config_name: '',
        status: 'starting',
        total_cases: 0,
        completed_cases: 0,
        progress_log: [],
      },
    })
    useBenchmarkStore.getState().handleBenchmarkStarted({
      run_id: 'r1',
      config_name: 'cfg-x',
      started_at: '2025-01-01T00:00:00',
      total_cases: 15,
    })
    const { activeRun } = useBenchmarkStore.getState()
    expect(activeRun?.status).toBe('running')
    expect(activeRun?.config_name).toBe('cfg-x')
    expect(activeRun?.total_cases).toBe(15)
  })
})

describe('handleBenchmarkProgress', () => {
  beforeEach(() => {
    useBenchmarkStore.setState({
      activeRun: {
        run_id: 'r1',
        config_name: 'cfg',
        status: 'running',
        total_cases: 5,
        completed_cases: 0,
        progress_log: [],
      },
    })
  })

  it('appends to progress_log and updates completed_cases', () => {
    const event: ProgressEvent = {
      run_id: 'r1',
      round_num: 1,
      subject: 'sub_a',
      task: 'code',
      case_id: 'c1',
      status: 'ok',
      score: 1.0,
      passed: true,
      completed: 1,
      total: 5,
    }
    useBenchmarkStore.getState().handleBenchmarkProgress(event)
    const { activeRun } = useBenchmarkStore.getState()
    expect(activeRun?.progress_log).toHaveLength(1)
    expect(activeRun?.completed_cases).toBe(1)
    expect(activeRun?.progress_log[0].case_id).toBe('c1')
  })

  it('ignores events for other run_ids', () => {
    const event: ProgressEvent = {
      run_id: 'other-run',
      round_num: 1,
      subject: 's',
      task: 't',
      case_id: 'cx',
      status: 'ok',
      score: 1,
      passed: true,
      completed: 1,
      total: 5,
    }
    useBenchmarkStore.getState().handleBenchmarkProgress(event)
    expect(useBenchmarkStore.getState().activeRun?.progress_log).toHaveLength(0)
  })
})

describe('handleBenchmarkComplete', () => {
  it('sets status to completed and stores report', () => {
    useBenchmarkStore.setState({
      activeRun: {
        run_id: 'r1',
        config_name: 'cfg',
        status: 'running',
        total_cases: 5,
        completed_cases: 4,
        progress_log: [],
      },
    })
    const report = { config_name: 'cfg', class_report: { sub_a: { accuracy_mean: 0.9 } } }
    useBenchmarkStore.getState().handleBenchmarkComplete({
      run_id: 'r1',
      report,
      case_count: 5,
    })
    const { activeRun } = useBenchmarkStore.getState()
    expect(activeRun?.status).toBe('completed')
    expect(activeRun?.report?.config_name).toBe('cfg')
    expect(activeRun?.completed_cases).toBe(5)
  })
})

describe('handleBenchmarkError', () => {
  it('sets status to failed on generic error', () => {
    useBenchmarkStore.setState({
      activeRun: {
        run_id: 'r1',
        config_name: 'cfg',
        status: 'running',
        total_cases: 5,
        completed_cases: 2,
        progress_log: [],
      },
    })
    useBenchmarkStore.getState().handleBenchmarkError({
      run_id: 'r1',
      error: 'Something went wrong',
    })
    expect(useBenchmarkStore.getState().activeRun?.status).toBe('failed')
    expect(useBenchmarkStore.getState().activeRun?.error).toBe('Something went wrong')
  })

  it('sets status to stopped when error message contains "stopped"', () => {
    useBenchmarkStore.setState({
      activeRun: {
        run_id: 'r1',
        config_name: 'cfg',
        status: 'running',
        total_cases: 5,
        completed_cases: 2,
        progress_log: [],
      },
    })
    useBenchmarkStore.getState().handleBenchmarkError({
      run_id: 'r1',
      error: 'Run stopped by user',
    })
    expect(useBenchmarkStore.getState().activeRun?.status).toBe('stopped')
  })
})

// ---------------------------------------------------------------------------
// UI helpers
// ---------------------------------------------------------------------------

describe('clearActiveRun', () => {
  it('nulls the activeRun', () => {
    useBenchmarkStore.setState({
      activeRun: {
        run_id: 'r1',
        config_name: 'c',
        status: 'completed',
        total_cases: 1,
        completed_cases: 1,
        progress_log: [],
      },
    })
    useBenchmarkStore.getState().clearActiveRun()
    expect(useBenchmarkStore.getState().activeRun).toBeNull()
  })
})

describe('clearSelectedRunDetail', () => {
  it('nulls selectedRunDetail', () => {
    useBenchmarkStore.setState({
      selectedRunDetail: { report: {}, cases_by_subject: {} },
    })
    useBenchmarkStore.getState().clearSelectedRunDetail()
    expect(useBenchmarkStore.getState().selectedRunDetail).toBeNull()
  })
})
