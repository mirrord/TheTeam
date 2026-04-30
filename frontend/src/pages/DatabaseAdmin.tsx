/**
 * Database administration page.
 *
 * - Lists every database (memory, history, flowcharts) with size and status.
 * - Allows clearing a specific database (with explicit confirmation).
 * - Allows running a search across databases or a memory-only search.
 */

import { useEffect, useState } from 'react'
import { Database, Trash2, Search, RefreshCw, AlertTriangle } from 'lucide-react'
import { useDatabaseStore } from '../store/databaseStore'

type ClearTarget = 'memory' | 'history' | 'flowcharts' | 'all'

function formatBytes(n: number): string {
  if (!Number.isFinite(n) || n < 0) return '—'
  if (n < 1024) return `${n} B`
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`
  if (n < 1024 * 1024 * 1024) return `${(n / 1024 / 1024).toFixed(1)} MB`
  return `${(n / 1024 / 1024 / 1024).toFixed(1)} GB`
}

export default function DatabaseAdmin() {
  const {
    databases,
    searchResults,
    memoryResults,
    loading,
    error,
    fetchInfo,
    clear,
    search,
    searchMemory,
    resetSearch,
  } = useDatabaseStore()

  const [confirmTarget, setConfirmTarget] = useState<ClearTarget | null>(null)
  const [query, setQuery] = useState('')
  const [exact, setExact] = useState(false)
  const [memoryQuery, setMemoryQuery] = useState('')

  useEffect(() => {
    fetchInfo()
  }, [fetchInfo])

  const handleClear = async (target: ClearTarget) => {
    try {
      await clear(target)
      await fetchInfo()
    } finally {
      setConfirmTarget(null)
    }
  }

  return (
    <div className="flex-1 overflow-y-auto p-6">
      <div className="max-w-6xl mx-auto space-y-6">
        <header className="flex items-center justify-between">
          <h1 className="text-2xl font-bold flex items-center gap-2">
            <Database className="w-6 h-6" /> Database Administration
          </h1>
          <button
            onClick={() => fetchInfo()}
            className="flex items-center gap-2 px-3 py-2 rounded bg-gray-700 hover:bg-gray-600"
          >
            <RefreshCw className="w-4 h-4" /> Refresh
          </button>
        </header>

        {error && (
          <div className="bg-red-900/40 border border-red-700 text-red-200 px-4 py-2 rounded">
            {error}
          </div>
        )}

        {/* Info table */}
        <section className="bg-gray-800 rounded-lg overflow-hidden">
          <table className="w-full text-sm">
            <thead className="bg-gray-700 text-gray-300">
              <tr>
                <th className="px-4 py-2 text-left">Name</th>
                <th className="px-4 py-2 text-left">Type</th>
                <th className="px-4 py-2 text-left">Path</th>
                <th className="px-4 py-2 text-right">Size</th>
                <th className="px-4 py-2 text-center">Status</th>
                <th className="px-4 py-2 text-right">Actions</th>
              </tr>
            </thead>
            <tbody>
              {databases.length === 0 && (
                <tr>
                  <td colSpan={6} className="text-center text-gray-400 py-6">
                    {loading ? 'Loading…' : 'No databases reported.'}
                  </td>
                </tr>
              )}
              {databases.map((db) => (
                <tr key={db.name} className="border-t border-gray-700">
                  <td className="px-4 py-2 font-mono">{db.name}</td>
                  <td className="px-4 py-2 text-gray-300">{db.type}</td>
                  <td className="px-4 py-2 text-gray-400 font-mono text-xs">{db.path}</td>
                  <td className="px-4 py-2 text-right">{formatBytes(db.size_bytes)}</td>
                  <td className="px-4 py-2 text-center">
                    {db.available ? (
                      <span className="text-green-400">available</span>
                    ) : (
                      <span className="text-red-400" title={db.error || ''}>
                        unavailable
                      </span>
                    )}
                  </td>
                  <td className="px-4 py-2 text-right">
                    {(db.name === 'memory' || db.name === 'history' || db.name === 'flowcharts') && (
                      <button
                        onClick={() => setConfirmTarget(db.name as ClearTarget)}
                        className="inline-flex items-center gap-1 px-2 py-1 rounded text-red-300 hover:bg-red-900/40"
                      >
                        <Trash2 className="w-4 h-4" /> Clear
                      </button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="p-3 bg-gray-900 text-right border-t border-gray-700">
            <button
              onClick={() => setConfirmTarget('all')}
              className="inline-flex items-center gap-1 px-3 py-1 rounded text-red-300 hover:bg-red-900/40 border border-red-700"
            >
              <Trash2 className="w-4 h-4" /> Clear All Databases
            </button>
          </div>
        </section>

        {/* Cross-database search */}
        <section className="bg-gray-800 rounded-lg p-4 space-y-3">
          <h2 className="font-semibold flex items-center gap-2">
            <Search className="w-4 h-4" /> Cross-database Search
          </h2>
          <div className="flex gap-2">
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search query…"
              className="flex-1 px-3 py-2 rounded bg-gray-900 border border-gray-700"
              onKeyDown={(e) => {
                if (e.key === 'Enter' && query.trim()) search(query.trim(), exact)
              }}
            />
            <label className="flex items-center gap-1 text-sm text-gray-300">
              <input type="checkbox" checked={exact} onChange={(e) => setExact(e.target.checked)} />
              Exact
            </label>
            <button
              onClick={() => query.trim() && search(query.trim(), exact)}
              disabled={!query.trim() || loading}
              className="px-4 py-2 rounded bg-primary-600 hover:bg-primary-500 disabled:bg-gray-600"
            >
              Search
            </button>
            <button
              onClick={resetSearch}
              className="px-3 py-2 rounded bg-gray-700 hover:bg-gray-600"
            >
              Clear
            </button>
          </div>
          {Object.entries(searchResults).map(([db, items]) => (
            <div key={db}>
              <div className="text-xs uppercase text-gray-400 mt-2">{db}</div>
              {items.length === 0 ? (
                <div className="text-gray-500 text-sm">No results.</div>
              ) : (
                <ul className="space-y-1">
                  {items.map((item, idx) => (
                    <li key={idx} className="text-sm bg-gray-900 px-3 py-2 rounded">
                      <div className="text-gray-200 break-words">{item.content}</div>
                      <div className="text-xs text-gray-500">
                        score {item.relevance_score?.toFixed(3)} · {item.match_type}
                      </div>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          ))}
        </section>

        {/* Memory-specific search */}
        <section className="bg-gray-800 rounded-lg p-4 space-y-3">
          <h2 className="font-semibold">Memory Search</h2>
          <div className="flex gap-2">
            <input
              value={memoryQuery}
              onChange={(e) => setMemoryQuery(e.target.value)}
              placeholder="Memory query…"
              className="flex-1 px-3 py-2 rounded bg-gray-900 border border-gray-700"
              onKeyDown={(e) => {
                if (e.key === 'Enter' && memoryQuery.trim()) searchMemory(memoryQuery.trim())
              }}
            />
            <button
              onClick={() => memoryQuery.trim() && searchMemory(memoryQuery.trim())}
              disabled={!memoryQuery.trim() || loading}
              className="px-4 py-2 rounded bg-primary-600 hover:bg-primary-500 disabled:bg-gray-600"
            >
              Search
            </button>
          </div>
          {Object.entries(memoryResults).map(([category, items]) => (
            <div key={category}>
              <div className="text-xs uppercase text-gray-400 mt-2">{category}</div>
              {items.length === 0 ? (
                <div className="text-gray-500 text-sm">No results.</div>
              ) : (
                <ul className="space-y-1">
                  {items.map((item) => (
                    <li key={item.id} className="text-sm bg-gray-900 px-3 py-2 rounded">
                      <div className="text-gray-200 break-words">{item.content}</div>
                      <div className="text-xs text-gray-500">
                        score {item.relevance_score?.toFixed(3)} · distance {item.distance?.toFixed(3)}
                      </div>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          ))}
        </section>

        {/* Confirm-clear modal */}
        {confirmTarget && (
          <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50">
            <div className="bg-gray-800 rounded-lg p-6 max-w-md w-full space-y-4 border border-red-700">
              <h3 className="text-lg font-semibold flex items-center gap-2 text-red-300">
                <AlertTriangle className="w-5 h-5" /> Confirm clear: {confirmTarget}
              </h3>
              <p className="text-sm text-gray-300">
                This will permanently delete data from
                {confirmTarget === 'all' ? ' all databases' : ` the "${confirmTarget}" database`}.
                This action cannot be undone.
              </p>
              <div className="flex justify-end gap-2">
                <button
                  onClick={() => setConfirmTarget(null)}
                  className="px-3 py-2 rounded bg-gray-700 hover:bg-gray-600"
                >
                  Cancel
                </button>
                <button
                  onClick={() => handleClear(confirmTarget)}
                  className="px-3 py-2 rounded bg-red-600 hover:bg-red-500"
                >
                  Clear
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
