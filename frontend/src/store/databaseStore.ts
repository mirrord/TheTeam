/**
 * Database admin store. Wraps the /api/v1/database endpoints.
 */

import { create } from 'zustand'

export interface DatabaseInfo {
  name: string
  type: string
  path: string
  size_bytes: number
  available: boolean
  error: string | null
}

export interface SearchResultItem {
  database: string
  type: string
  content: string
  metadata: Record<string, any>
  relevance_score: number
  match_type: string
}

export interface MemoryResultItem {
  id: string
  category: string
  content: string
  metadata: Record<string, any>
  distance: number
  relevance_score: number
}

interface DatabaseState {
  databases: DatabaseInfo[]
  searchResults: Record<string, SearchResultItem[]>
  memoryResults: Record<string, MemoryResultItem[]>
  categories: string[]
  loading: boolean
  error: string | null

  fetchInfo: () => Promise<void>
  clear: (database: 'memory' | 'history' | 'flowcharts' | 'all') => Promise<void>
  search: (query: string, exact?: boolean, databases?: string[]) => Promise<void>
  fetchCategories: () => Promise<void>
  searchMemory: (query: string, categories?: string[]) => Promise<void>
  resetSearch: () => void
}

export const useDatabaseStore = create<DatabaseState>((set) => ({
  databases: [],
  searchResults: {},
  memoryResults: {},
  categories: [],
  loading: false,
  error: null,

  fetchInfo: async () => {
    set({ loading: true, error: null })
    try {
      const response = await fetch('/api/v1/database/info')
      if (!response.ok) throw new Error('Failed to fetch database info')
      const data = await response.json()
      set({ databases: data.databases, loading: false })
    } catch (error: any) {
      set({ error: error.message, loading: false })
    }
  },

  clear: async (database) => {
    set({ loading: true, error: null })
    try {
      const response = await fetch(`/api/v1/database/clear/${database}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ confirm: true }),
      })
      if (!response.ok) {
        const err = await response.json().catch(() => ({}))
        throw new Error(err.error || 'Failed to clear database')
      }
      set({ loading: false })
    } catch (error: any) {
      set({ error: error.message, loading: false })
      throw error
    }
  },

  search: async (query, exact = false, databases) => {
    set({ loading: true, error: null })
    try {
      const response = await fetch('/api/v1/database/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, exact, databases }),
      })
      if (!response.ok) {
        const err = await response.json().catch(() => ({}))
        throw new Error(err.error || 'Search failed')
      }
      const data = await response.json()
      set({ searchResults: data.results || {}, loading: false })
    } catch (error: any) {
      set({ error: error.message, loading: false })
    }
  },

  fetchCategories: async () => {
    try {
      const response = await fetch('/api/v1/database/memory/categories')
      if (!response.ok) return
      const data = await response.json()
      set({ categories: data.categories || [] })
    } catch {
      // Categories are advisory; ignore failures.
    }
  },

  searchMemory: async (query, categories) => {
    set({ loading: true, error: null })
    try {
      const response = await fetch('/api/v1/database/memory/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, categories }),
      })
      if (!response.ok) {
        const err = await response.json().catch(() => ({}))
        throw new Error(err.error || 'Memory search failed')
      }
      const data = await response.json()
      set({ memoryResults: data.results || {}, loading: false })
    } catch (error: any) {
      set({ error: error.message, loading: false })
    }
  },

  resetSearch: () => set({ searchResults: {}, memoryResults: {} }),
}))
