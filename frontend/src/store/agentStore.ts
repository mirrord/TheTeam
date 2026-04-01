/**
 * Agent store for managing agent configurations.
 */

import { create } from 'zustand'
import { fetchWithToast } from '../lib/fetchWithToast'

export interface Agent {
  id: string
  name: string
  model: string
  source: 'file' | 'runtime'
  path?: string
}

export interface AgentConfig {
  id: string
  name?: string
  model: string
  system_prompt?: string
  temperature?: number
  tools?: string[]
  flowchart?: string
  tool_auto_loop?: boolean
  tool_max_iterations?: number
  memory_enabled?: boolean
  current_context?: string
  save_to_file?: boolean
  compaction?: {
    enabled: boolean
    threshold?: number
    keep_last?: number
    summary_model?: string | null
    memory_category?: string
  }
  recall?: {
    enabled: boolean
    sources?: string[]
    n_results?: number
    recall_model?: string | null
    categories?: string[]
    min_relevance?: number
  }
  [key: string]: any
}

interface AgentState {
  agents: Agent[]
  currentAgent: AgentConfig | null
  loading: boolean
  error: string | null

  // Actions
  fetchAgents: () => Promise<void>
  getAgent: (id: string) => Promise<void>
  createAgent: (config: AgentConfig) => Promise<string>
  updateAgent: (id: string, config: AgentConfig) => Promise<void>
  deleteAgent: (id: string) => Promise<void>
  testAgent: (id: string, prompt: string) => Promise<any>
  setCurrentAgent: (agent: AgentConfig | null) => void
}

export const useAgentStore = create<AgentState>((set, get) => ({
  agents: [],
  currentAgent: null,
  loading: false,
  error: null,

  fetchAgents: async () => {
    set({ loading: true, error: null })
    try {
      const response = await fetch('/api/v1/agents/')
      if (!response.ok) throw new Error('Failed to fetch agents')
      const data = await response.json()
      set({ agents: data.agents, loading: false })
    } catch (error: any) {
      set({ error: error.message, loading: false })
    }
  },

  getAgent: async (id: string) => {
    set({ loading: true, error: null })
    try {
      const response = await fetch(`/api/v1/agents/${id}`)
      if (!response.ok) throw new Error('Failed to fetch agent')
      const data = await response.json()
      set({ currentAgent: data.agent.config, loading: false })
    } catch (error: any) {
      set({ error: error.message, loading: false })
    }
  },

  createAgent: async (config: AgentConfig) => {
    set({ loading: true, error: null })
    try {
      const response = await fetch('/api/v1/agents/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config),
      })
      if (!response.ok) throw new Error('Failed to create agent')
      const data = await response.json()
      await get().fetchAgents() // Refresh list
      set({ loading: false })
      return data.agent_id
    } catch (error: any) {
      set({ error: error.message, loading: false })
      throw error
    }
  },

  updateAgent: async (id: string, config: AgentConfig) => {
    set({ loading: true, error: null })
    try {
      const response = await fetch(`/api/v1/agents/${id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config),
      })
      if (!response.ok) throw new Error('Failed to update agent')
      await get().fetchAgents() // Refresh list
      set({ loading: false })
    } catch (error: any) {
      set({ error: error.message, loading: false })
      throw error
    }
  },

  deleteAgent: async (id: string) => {
    set({ loading: true, error: null })
    try {
      const response = await fetch(`/api/v1/agents/${id}`, {
        method: 'DELETE',
      })
      if (!response.ok) throw new Error('Failed to delete agent')
      await get().fetchAgents() // Refresh list
      set({ loading: false })
    } catch (error: any) {
      set({ error: error.message, loading: false })
      throw error
    }
  },

  testAgent: async (id: string, prompt: string) => {
    set({ loading: true, error: null })
    try {
      const response = await fetchWithToast(`/api/v1/agents/${id}/test`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt }),
      })
      if (!response.ok) throw new Error('Failed to test agent')
      const data = await response.json()
      set({ loading: false })
      return data.result
    } catch (error: any) {
      set({ error: error.message, loading: false })
      throw error
    }
  },

  setCurrentAgent: (agent: AgentConfig | null) => {
    set({ currentAgent: agent })
  },
}))
