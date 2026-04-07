import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { useAgentStore, AgentConfig } from '../../store/agentStore'

const initialState = {
  agents: [],
  currentAgent: null,
  loading: false,
  error: null,
}

function makeConfig(overrides: Partial<AgentConfig> = {}): AgentConfig {
  return { id: 'agent-1', model: 'llama3', ...overrides }
}

beforeEach(() => {
  useAgentStore.setState(initialState)
  vi.restoreAllMocks()
})

afterEach(() => {
  vi.restoreAllMocks()
})

// ─── Pure state mutations ────────────────────────────────────────────────────

describe('setCurrentAgent', () => {
  it('sets the current agent', () => {
    const config = makeConfig()
    useAgentStore.getState().setCurrentAgent(config)
    expect(useAgentStore.getState().currentAgent).toEqual(config)
  })

  it('clears the current agent when passed null', () => {
    useAgentStore.setState({ currentAgent: makeConfig() })
    useAgentStore.getState().setCurrentAgent(null)
    expect(useAgentStore.getState().currentAgent).toBeNull()
  })
})

// ─── Fetch-backed actions ────────────────────────────────────────────────────

describe('fetchAgents', () => {
  it('replaces the agents list on success', async () => {
    const agentList = [{ id: 'a1', name: 'Alpha', model: 'llama3', source: 'file' as const }]
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(JSON.stringify({ agents: agentList }), { status: 200 })
    )

    await useAgentStore.getState().fetchAgents()

    expect(useAgentStore.getState().agents).toEqual(agentList)
    expect(useAgentStore.getState().loading).toBe(false)
    expect(useAgentStore.getState().error).toBeNull()
  })

  it('sets error and clears loading on HTTP failure', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response('', { status: 500 })
    )

    await useAgentStore.getState().fetchAgents()

    expect(useAgentStore.getState().error).toBeTruthy()
    expect(useAgentStore.getState().loading).toBe(false)
  })

  it('sets error on network failure', async () => {
    vi.spyOn(globalThis, 'fetch').mockRejectedValue(new Error('No network'))

    await useAgentStore.getState().fetchAgents()

    expect(useAgentStore.getState().error).toBe('No network')
    expect(useAgentStore.getState().loading).toBe(false)
  })
})

describe('getAgent', () => {
  it('stores the agent config from the API', async () => {
    const config = makeConfig({ name: 'Beta', system_prompt: 'Be helpful' })
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(JSON.stringify({ agent: { config } }), { status: 200 })
    )

    await useAgentStore.getState().getAgent('agent-1')

    expect(useAgentStore.getState().currentAgent).toEqual(config)
    expect(useAgentStore.getState().loading).toBe(false)
  })

  it('sets error when the agent is not found', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response('', { status: 404 })
    )

    await useAgentStore.getState().getAgent('missing')

    expect(useAgentStore.getState().error).toBeTruthy()
  })
})

describe('createAgent', () => {
  it('returns the new agent_id and refreshes the list', async () => {
    vi.spyOn(globalThis, 'fetch')
      .mockResolvedValueOnce(
        new Response(JSON.stringify({ agent_id: 'new-agent' }), { status: 200 })
      )
      .mockResolvedValueOnce(
        new Response(JSON.stringify({ agents: [] }), { status: 200 })
      )

    const id = await useAgentStore.getState().createAgent(makeConfig())
    expect(id).toBe('new-agent')
  })

  it('sets error and rethrows on failure', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response('', { status: 500 })
    )

    await expect(useAgentStore.getState().createAgent(makeConfig())).rejects.toThrow()
    expect(useAgentStore.getState().error).toBeTruthy()
  })
})

describe('updateAgent', () => {
  it('sets loading false on success and refreshes the list', async () => {
    vi.spyOn(globalThis, 'fetch')
      .mockResolvedValueOnce(new Response('{}', { status: 200 }))
      .mockResolvedValueOnce(new Response(JSON.stringify({ agents: [] }), { status: 200 }))

    await useAgentStore.getState().updateAgent('agent-1', makeConfig())
    expect(useAgentStore.getState().loading).toBe(false)
    expect(useAgentStore.getState().error).toBeNull()
  })

  it('sets error and rethrows on failure', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response('', { status: 500 })
    )

    await expect(useAgentStore.getState().updateAgent('agent-1', makeConfig())).rejects.toThrow()
    expect(useAgentStore.getState().error).toBeTruthy()
  })
})

describe('deleteAgent', () => {
  it('sets loading false on success and refreshes the list', async () => {
    vi.spyOn(globalThis, 'fetch')
      .mockResolvedValueOnce(new Response('{}', { status: 200 }))
      .mockResolvedValueOnce(new Response(JSON.stringify({ agents: [] }), { status: 200 }))

    await useAgentStore.getState().deleteAgent('agent-1')
    expect(useAgentStore.getState().loading).toBe(false)
  })

  it('sets error and rethrows on failure', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response('', { status: 500 })
    )

    await expect(useAgentStore.getState().deleteAgent('agent-1')).rejects.toThrow()
    expect(useAgentStore.getState().error).toBeTruthy()
  })
})

describe('testAgent', () => {
  it('returns the result from the API on success', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(JSON.stringify({ result: 'The answer is 42' }), { status: 200 })
    )

    const result = await useAgentStore.getState().testAgent('agent-1', 'What is 6x7?')
    expect(result).toBe('The answer is 42')
  })

  it('sets error and rethrows on failure', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(JSON.stringify({ error: 'Agent busy' }), { status: 500 })
    )

    await expect(useAgentStore.getState().testAgent('agent-1', 'hello')).rejects.toThrow()
    expect(useAgentStore.getState().error).toBeTruthy()
  })
})
