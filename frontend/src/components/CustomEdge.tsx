import { useState, useEffect } from 'react'
import { EdgeProps, getBezierPath, EdgeLabelRenderer } from 'reactflow'
import { Save } from 'lucide-react'

// Edge type colors
const EDGE_COLORS: Record<string, string> = {
  default: '#64748b',      // slate-500
  conditional: '#f59e0b',  // amber-500
  loop: '#8b5cf6',         // violet-500
  error: '#ef4444',        // red-500
  success: '#10b981',      // green-500
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

export function CustomEdge({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  data,
}: EdgeProps) {
  const [isHovered, setIsHovered] = useState(false)
  const [isEditing, setIsEditing] = useState(false)
  const [edgeType, setEdgeType] = useState<string>(data?.type || 'default')
  
  // Edit state
  const [conditionType, setConditionType] = useState(data?.condition?.type || 'AlwaysCondition')
  const [priority, setPriority] = useState(data?.priority ?? 9)
  const [limit, setLimit] = useState(data?.condition?.limit ?? 1)
  const [enabled, setEnabled] = useState(data?.condition?.enabled ?? true)
  const [outputKey, setOutputKey] = useState(data?.output_key ?? 'default')
  const [inputKey, setInputKey] = useState(data?.input_key ?? 'default')

  // Update edge type and edit state when data changes
  useEffect(() => {
    const newType = data?.type || 'default'
    if (newType !== edgeType) {
      setEdgeType(newType)
    }
    
    // Sync edit state with data
    setConditionType(data?.condition?.type || 'AlwaysCondition')
    setPriority(data?.priority ?? 9)
    setLimit(data?.condition?.limit ?? 1)
    setEnabled(data?.condition?.enabled ?? true)
    setOutputKey(data?.output_key ?? 'default')
    setInputKey(data?.input_key ?? 'default')
  }, [data, edgeType])

  const [edgePath, labelX, labelY] = getBezierPath({
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
  })

  const edgeColor = EDGE_COLORS[edgeType] || EDGE_COLORS.default
  
  // Create custom marker ID unique to this edge instance
  const markerId = `arrow-${edgeType}-${id}`
  const markerUrl = `url(#${markerId})`

  const handleBoxClick = (e: React.MouseEvent) => {
    e.stopPropagation()
    if (!isEditing) {
      setIsEditing(true)
    }
  }

  const handleSave = () => {
    const updatedData = {
      condition: {
        type: conditionType,
        enabled,
        ...(conditionType === 'CountCondition' && { limit }),
      },
      priority,
      output_key: outputKey,
      input_key: inputKey,
    }
    
    // Update the edge type for visual display
    const newConditionType = updatedData.condition?.type || 'AlwaysCondition'
    let newEdgeType = 'default'
    
    switch (newConditionType) {
      case 'RegexCondition':
      case 'ContextCondition':
      case 'CustomCondition':
        newEdgeType = 'conditional'
        break
      case 'CountCondition':
      case 'LoopCondition':
        newEdgeType = 'loop'
        break
      case 'ErrorCondition':
        newEdgeType = 'error'
        break
      case 'SuccessCondition':
        newEdgeType = 'success'
        break
      case 'AlwaysCondition':
      default:
        newEdgeType = 'default'
        break
    }
    
    setEdgeType(newEdgeType)
    
    // If there's an onEdgeDataChange callback, call it
    if (data?.onEdgeDataChange) {
      data.onEdgeDataChange(id, updatedData)
    }
    
    setIsEditing(false)
  }

  // Format edge info for display
  const getEdgeInfo = () => {
    const condition = data?.condition
    const priority = data?.priority
    
    let info = []
    
    if (condition?.type) {
      info.push(`Type: ${condition.type}`)
    }
    
    if (condition?.type === 'CountCondition' && condition?.limit) {
      info.push(`Limit: ${condition.limit}`)
    }
    
    if (priority !== undefined) {
      info.push(`Priority: ${priority}`)
    }
    
    if (data?.output_key && data.output_key !== 'default') {
      info.push(`Out: ${data.output_key}`)
    }
    
    if (data?.input_key && data.input_key !== 'default') {
      info.push(`In: ${data.input_key}`)
    }
    
    return info.length > 0 ? info : ['Click to edit']
  }

  return (
    <>
      {/* Define custom markers for each edge type */}
      <defs>
        <marker
          id={markerId}
          viewBox="0 0 10 10"
          refX="8"
          refY="5"
          markerWidth="6"
          markerHeight="6"
          orient="auto-start-reverse"
        >
          <path d="M 0 0 L 10 5 L 0 10 z" fill={edgeColor} />
        </marker>
      </defs>
      
      {/* Invisible wider path for easier hovering */}
      <path
        d={edgePath}
        fill="none"
        stroke="transparent"
        strokeWidth={20}
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
        style={{ cursor: 'pointer' }}
        className="react-flow__edge-path"
      />
      
      {/* Visible edge path with color */}
      <path
        d={edgePath}
        fill="none"
        stroke={edgeColor}
        strokeWidth={isHovered ? 3 : 2}
        strokeDasharray="5,5"
        markerEnd={markerUrl}
        className="animated-edge"
        style={{
          pointerEvents: 'none',
          transition: 'stroke-width 0.2s',
        }}
      />
      
      {/* Edge info box - visible on hover or when editing */}
      {(isHovered || isEditing) && (
        <EdgeLabelRenderer>
          <div
            style={{
              position: 'absolute',
              transform: `translate(-50%, -50%) translate(${labelX}px,${labelY}px)`,
              pointerEvents: 'all',
              zIndex: 1000,
            }}
            onMouseEnter={() => setIsHovered(true)}
            onMouseLeave={() => {
              if (!isEditing) setIsHovered(false)
            }}
          >
            {!isEditing ? (
              // Display mode
              <div
                onClick={handleBoxClick}
                className="px-3 py-2 text-xs rounded-lg shadow-lg transition-all cursor-pointer hover:shadow-xl"
                style={{
                  backgroundColor: edgeColor,
                  color: 'white',
                  minWidth: '120px',
                }}
              >
                <div className="font-semibold mb-1">{edgeType.toUpperCase()}</div>
                {getEdgeInfo().map((info, idx) => (
                  <div key={idx} className="opacity-90">
                    {info}
                  </div>
                ))}
              </div>
            ) : (
              // Edit mode
              <div
                className="bg-white rounded-lg shadow-2xl border-2 p-4 text-xs"
                style={{
                  borderColor: edgeColor,
                  minWidth: '280px',
                  maxWidth: '320px',
                }}
                onClick={(e) => e.stopPropagation()}
              >
                <div className="space-y-3">
                  {/* Condition Type */}
                  <div>
                    <label className="block text-gray-700 font-medium mb-1">
                      Condition Type
                    </label>
                    <select
                      value={conditionType}
                      onChange={(e) => setConditionType(e.target.value)}
                      className="w-full px-2 py-1 border border-gray-300 rounded focus:outline-none focus:ring-1 focus:ring-blue-500 text-xs"
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
                      <label className="block text-gray-700 font-medium mb-1">
                        Limit
                      </label>
                      <input
                        type="number"
                        min="1"
                        value={limit}
                        onChange={(e) => setLimit(parseInt(e.target.value) || 1)}
                        className="w-full px-2 py-1 border border-gray-300 rounded focus:outline-none focus:ring-1 focus:ring-blue-500 text-xs"
                      />
                    </div>
                  )}

                  {/* Enabled */}
                  <div className="flex items-center">
                    <input
                      type="checkbox"
                      id={`enabled-${id}`}
                      checked={enabled}
                      onChange={(e) => setEnabled(e.target.checked)}
                      className="h-3 w-3 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                    />
                    <label
                      htmlFor={`enabled-${id}`}
                      className="ml-2 text-gray-700"
                    >
                      Enabled
                    </label>
                  </div>

                  {/* Priority */}
                  <div>
                    <label className="block text-gray-700 font-medium mb-1">
                      Priority (1-10)
                    </label>
                    <input
                      type="number"
                      min="1"
                      max="10"
                      value={priority}
                      onChange={(e) => setPriority(parseInt(e.target.value) || 9)}
                      className="w-full px-2 py-1 border border-gray-300 rounded focus:outline-none focus:ring-1 focus:ring-blue-500 text-xs"
                    />
                  </div>

                  {/* Output Key */}
                  <div>
                    <label className="block text-gray-700 font-medium mb-1">
                      Output Key
                    </label>
                    <input
                      type="text"
                      value={outputKey}
                      onChange={(e) => setOutputKey(e.target.value)}
                      className="w-full px-2 py-1 border border-gray-300 rounded focus:outline-none focus:ring-1 focus:ring-blue-500 text-xs"
                    />
                  </div>

                  {/* Input Key */}
                  <div>
                    <label className="block text-gray-700 font-medium mb-1">
                      Input Key
                    </label>
                    <input
                      type="text"
                      value={inputKey}
                      onChange={(e) => setInputKey(e.target.value)}
                      className="w-full px-2 py-1 border border-gray-300 rounded focus:outline-none focus:ring-1 focus:ring-blue-500 text-xs"
                    />
                  </div>

                  {/* Save Button */}
                  <button
                    onClick={handleSave}
                    className="w-full flex items-center justify-center gap-2 px-3 py-2 text-white rounded hover:opacity-90 transition-colors font-medium"
                    style={{ backgroundColor: edgeColor }}
                  >
                    <Save size={14} />
                    Save
                  </button>
                </div>
              </div>
            )}
          </div>
        </EdgeLabelRenderer>
      )}
    </>
  )
}
