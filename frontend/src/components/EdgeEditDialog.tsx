import { useState, useEffect } from 'react'
import { X } from 'lucide-react'

interface EdgeData {
  condition?: {
    type?: string
    limit?: number
    enabled?: boolean
  }
  priority?: number
  output_key?: string
  input_key?: string
}

interface EdgeEditDialogProps {
  isOpen: boolean
  onClose: () => void
  edgeData: EdgeData
  onSave: (updatedData: EdgeData) => void
}

const CONDITION_TYPES = [
  'AlwaysCondition',
  'CountCondition',
  'NeverCondition',
  'CustomCondition',
  'RegexCondition',
  'ContextCondition',
  'LoopCondition',
  'ErrorCondition',
  'SuccessCondition',
]

export function EdgeEditDialog({
  isOpen,
  onClose,
  edgeData,
  onSave,
}: EdgeEditDialogProps) {
  const initialConditionType = edgeData.condition?.type || 'AlwaysCondition'
  const [conditionType, setConditionType] = useState(initialConditionType)
  const [priority, setPriority] = useState(edgeData.priority ?? 9)
  const [limit, setLimit] = useState(edgeData.condition?.limit ?? 1)
  const [enabled, setEnabled] = useState(edgeData.condition?.enabled ?? true)
  const [outputKey, setOutputKey] = useState(edgeData.output_key ?? 'default')
  const [inputKey, setInputKey] = useState(edgeData.input_key ?? 'default')

  // Update internal state when edgeData changes
  useEffect(() => {
    setConditionType(edgeData.condition?.type || 'AlwaysCondition')
    setPriority(edgeData.priority ?? 9)
    setLimit(edgeData.condition?.limit ?? 1)
    setEnabled(edgeData.condition?.enabled ?? true)
    setOutputKey(edgeData.output_key ?? 'default')
    setInputKey(edgeData.input_key ?? 'default')
  }, [edgeData])

  if (!isOpen) return null

  const handleSave = () => {
    const updatedData: EdgeData = {
      condition: {
        type: conditionType,
        enabled,
        ...(conditionType === 'CountCondition' && { limit }),
      },
      priority,
      output_key: outputKey,
      input_key: inputKey,
    }
    onSave(updatedData)
    onClose()
  }

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl max-w-md w-full mx-4">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b">
          <h2 className="text-lg font-semibold text-gray-900">Edit Edge</h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600 transition-colors"
          >
            <X size={20} />
          </button>
        </div>

        {/* Body */}
        <div className="p-4 space-y-4">
          {/* Condition Type */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Condition Type
            </label>
            <select
              value={conditionType}
              onChange={(e) => setConditionType(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              {CONDITION_TYPES.map((type) => (
                <option key={type} value={type}>
                  {type}
                </option>
              ))}
            </select>
          </div>

          {/* Count Limit (only for CountCondition) */}
          {conditionType === 'CountCondition' && (
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Limit
              </label>
              <input
                type="number"
                min="1"
                value={limit}
                onChange={(e) => setLimit(parseInt(e.target.value) || 1)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
              <p className="text-xs text-gray-500 mt-1">
                Maximum number of times this edge can be traversed
              </p>
            </div>
          )}

          {/* Enabled */}
          <div className="flex items-center">
            <input
              type="checkbox"
              id="enabled"
              checked={enabled}
              onChange={(e) => setEnabled(e.target.checked)}
              className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
            />
            <label
              htmlFor="enabled"
              className="ml-2 block text-sm text-gray-700"
            >
              Enabled
            </label>
          </div>

          {/* Priority */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Priority
            </label>
            <input
              type="number"
              min="1"
              max="10"
              value={priority}
              onChange={(e) => setPriority(parseInt(e.target.value) || 9)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
            <p className="text-xs text-gray-500 mt-1">
              Higher priority edges are evaluated first (1-10)
            </p>
          </div>

          {/* Output Key */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Output Key
            </label>
            <input
              type="text"
              value={outputKey}
              onChange={(e) => setOutputKey(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
            <p className="text-xs text-gray-500 mt-1">
              Which output from the source node to use
            </p>
          </div>

          {/* Input Key */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Input Key
            </label>
            <input
              type="text"
              value={inputKey}
              onChange={(e) => setInputKey(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
            <p className="text-xs text-gray-500 mt-1">
              Which input of the target node to send to
            </p>
          </div>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-end gap-2 p-4 border-t">
          <button
            onClick={onClose}
            className="px-4 py-2 text-sm font-medium text-gray-700 bg-gray-100 rounded-md hover:bg-gray-200 transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={handleSave}
            className="px-4 py-2 text-sm font-medium text-white bg-blue-600 rounded-md hover:bg-blue-700 transition-colors"
          >
            Save
          </button>
        </div>
      </div>
    </div>
  )
}
