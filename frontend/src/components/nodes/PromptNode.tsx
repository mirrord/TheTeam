import { Handle, Position, useEdges, useReactFlow, NodeProps } from 'reactflow'
import { MessageSquare, Play, Flag, Edit } from 'lucide-react'
import { useMemo, useState, Fragment, useCallback } from 'react'

export interface PromptNodeData {
  label: string
  prompt?: string
  type: 'prompt'
  isStartNode?: boolean
  isEndNode?: boolean
  onEdit?: () => void
  static_inputs?: Record<string, string>
}

export function PromptNode({ id, data }: NodeProps<PromptNodeData>) {
  const [isHovered, setIsHovered] = useState(false)
  const edges = useEdges()
  const { setNodes } = useReactFlow()

  // Extract variables from prompt template (e.g., {var_name})
  const inputVariables = useMemo(() => {
    if (!data.prompt) return []
    const vars = new Set<string>()
    const regex = /\{(\w+)\}/g
    let match
    while ((match = regex.exec(data.prompt)) !== null) {
      vars.add(match[1])
    }
    return Array.from(vars)
  }, [data.prompt])

  const isVariableConnected = useCallback((varName: string) => {
    return edges.some(e => e.target === id && e.targetHandle === varName)
  }, [edges, id])

  const handleStaticInputChange = useCallback((varName: string, value: string) => {
    setNodes(nds => nds.map(node => {
      if (node.id === id) {
        return {
          ...node,
          data: {
            ...node.data,
            static_inputs: { ...(node.data.static_inputs || {}), [varName]: value },
          },
        }
      }
      return node
    }))
  }, [id, setNodes])

  const unconnectedVars = inputVariables.filter(v => !isVariableConnected(v))

  // Calculate minimum height: base handles + extra rows for unconnected var inputs
  const minHeight = Math.max(100, (inputVariables.length + 2) * 40 + unconnectedVars.length * 28)
  
  return (
    <div 
      className="px-4 py-3 shadow-lg rounded-lg border-2 border-blue-500 bg-gray-800 min-w-[260px] transition-all hover:shadow-xl hover:border-blue-400 relative"
      style={{ minHeight: `${minHeight}px` }}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      {/* Edit button */}
      {data.onEdit && (
        <button
          onClick={(e) => {
            e.stopPropagation()
            data.onEdit?.()
          }}
          className="absolute top-2 right-2 p-1 bg-gray-700 hover:bg-gray-600 rounded opacity-0 hover:opacity-100 transition-opacity z-10"
          style={{ opacity: isHovered ? 1 : 0 }}
        >
          <Edit className="w-3 h-3 text-gray-300" />
        </button>
      )}
      
      {/* Generic input handle - always shown for standard data flow */}
      <Handle
        type="target"
        position={Position.Left}
        id="input"
        style={{
          background: '#3b82f6',
          top: inputVariables.length > 0 ? '20%' : '50%',
        }}
        title="input (default)"
      />
      
      {/* Input handles for each template variable */}
      {inputVariables.map((varName, index) => {
        const handleTop = `${((index + 2) * 100) / (inputVariables.length + 2)}%`
        const isConnected = isVariableConnected(varName)
        return (
          <Fragment key={varName}>
            <Handle
              type="target"
              position={Position.Left}
              id={varName}
              style={{
                top: handleTop,
                background: isConnected ? '#60a5fa' : '#93c5fd',
              }}
              title={varName}
            />
            {/* Connected: show label outside on hover */}
            {isHovered && isConnected && (
              <div
                className="absolute left-0 pr-2"
                style={{ top: handleTop, transform: 'translate(-100%, -50%)' }}
              >
                <div className="text-xs text-blue-400 whitespace-nowrap">{varName}</div>
              </div>
            )}
          </Fragment>
        )
      })}

      <div className="flex items-center gap-2 mb-2">
        <MessageSquare className="w-4 h-4 text-blue-400" />
        <div className="font-bold text-sm text-gray-100 flex-1">{data.label}</div>
        {data.isStartNode && (
          <div className="flex items-center gap-1 text-xs px-2 py-0.5 bg-green-600 rounded-full">
            <Play className="w-3 h-3" />
            <span>START</span>
          </div>
        )}
        {data.isEndNode && (
          <div className="flex items-center gap-1 text-xs px-2 py-0.5 bg-red-600 rounded-full">
            <Flag className="w-3 h-3" />
            <span>END</span>
          </div>
        )}
      </div>

      {/* Static inputs for unconnected variable handles */}
      {unconnectedVars.length > 0 && (
        <div className="mt-2 space-y-1.5">
          {unconnectedVars.map(varName => (
            <div key={varName} className="flex items-center gap-1.5">
              <span className="text-xs text-blue-400 shrink-0 font-mono">{varName}:</span>
              <input
                type="text"
                value={(data.static_inputs || {})[varName] || ''}
                placeholder="static value"
                onChange={e => handleStaticInputChange(varName, e.target.value)}
                onMouseDown={e => e.stopPropagation()}
                className="flex-1 min-w-0 text-xs bg-gray-700 border border-gray-600 rounded px-1.5 py-0.5 text-gray-200 focus:outline-none focus:border-blue-500 placeholder-gray-500"
              />
            </div>
          ))}
        </div>
      )}

      {data.prompt && (
        <div className="text-xs text-gray-400 mt-2 max-w-[230px] truncate">
          {data.prompt}
        </div>
      )}

      {/* Show output variable name on hover */}
      {isHovered && (
        <div className="absolute right-0 top-1/2 -translate-y-1/2 translate-x-full pl-2">
          <div className="text-xs text-green-400 whitespace-nowrap">
            response
          </div>
        </div>
      )}

      {/* Output handle for response */}
      <Handle
        type="source"
        position={Position.Right}
        id="response"
        style={{
          background: '#10b981',
        }}
        title="response"
      />
    </div>
  )
}
