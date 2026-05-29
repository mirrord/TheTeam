import { X } from 'lucide-react'

export interface ToolInfo {
  name: string
  description?: string
  platform?: string
  source?: string
}

interface Props {
  isOpen: boolean
  onClose: () => void
  tools: ToolInfo[]
  enabledTools: string[]
  onToggle: (name: string) => void
  onEnableAll: () => void
  onDisableAll: () => void
}

export default function ToolsSidebar({
  isOpen,
  onClose,
  tools,
  enabledTools,
  onToggle,
  onEnableAll,
  onDisableAll,
}: Props) {
  const enabledSet = new Set(enabledTools)

  // Group tools by source for clearer browsing
  const grouped = tools.reduce<Record<string, ToolInfo[]>>((acc, t) => {
    const key = t.source || 'other'
    if (!acc[key]) acc[key] = []
    acc[key].push(t)
    return acc
  }, {})

  return (
    <>
      {/* Backdrop */}
      {isOpen && (
        <div
          className="fixed inset-0 bg-black bg-opacity-30 z-40"
          onClick={onClose}
        />
      )}

      {/* Sliding panel */}
      <div
        className={`fixed right-0 top-0 h-full w-96 bg-gray-800 border-l border-gray-700 shadow-2xl z-50 transform transition-transform duration-300 ease-in-out flex flex-col ${
          isOpen ? 'translate-x-0' : 'translate-x-full'
        }`}
      >
        <div className="p-4 border-b border-gray-700 flex items-center justify-between">
          <div>
            <h3 className="font-semibold text-lg">Tools</h3>
            <p className="text-xs text-gray-400">
              {enabledTools.length} of {tools.length} enabled
            </p>
          </div>
          <button
            onClick={onClose}
            className="p-1 hover:bg-gray-700 rounded"
            aria-label="Close tools panel"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="p-3 border-b border-gray-700 flex gap-2">
          <button
            onClick={onEnableAll}
            className="flex-1 px-2 py-1 text-xs bg-gray-700 hover:bg-gray-600 rounded"
          >
            Enable all
          </button>
          <button
            onClick={onDisableAll}
            className="flex-1 px-2 py-1 text-xs bg-gray-700 hover:bg-gray-600 rounded"
          >
            Disable all
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-3 space-y-4">
          {tools.length === 0 && (
            <p className="text-sm text-gray-400 text-center mt-8">
              No tools available
            </p>
          )}
          {Object.entries(grouped).map(([source, items]) => (
            <div key={source}>
              <div className="text-xs uppercase tracking-wide text-gray-500 mb-2">
                {source}
              </div>
              <div className="space-y-1">
                {items.map((tool) => {
                  const enabled = enabledSet.has(tool.name)
                  return (
                    <label
                      key={tool.name}
                      className="flex items-start gap-3 p-2 rounded hover:bg-gray-700 cursor-pointer"
                    >
                      <input
                        type="checkbox"
                        checked={enabled}
                        onChange={() => onToggle(tool.name)}
                        className="mt-1 accent-primary-500"
                      />
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2">
                          <span className="font-mono text-sm font-medium truncate">
                            {tool.name}
                          </span>
                          {tool.platform && tool.platform !== 'cross-platform' && (
                            <span className="text-[10px] px-1.5 py-0.5 bg-gray-700 rounded text-gray-400">
                              {tool.platform}
                            </span>
                          )}
                        </div>
                        {tool.description && (
                          <p className="text-xs text-gray-400 mt-0.5 line-clamp-2">
                            {tool.description}
                          </p>
                        )}
                      </div>
                    </label>
                  )
                })}
              </div>
            </div>
          ))}
        </div>
      </div>
    </>
  )
}
