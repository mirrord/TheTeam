import { Handle, Position } from 'reactflow'
import { MessageSquare, Play, Flag, Edit, Users } from 'lucide-react'
import { useMemo, useState, Fragment } from 'react'

export interface AgentPromptNodeData {
  label: string
  agent?: string
  prompt?: string
  model?: string
  context_name?: string
  type: 'agentprompt'
  isStartNode?: boolean
  isEndNode?: boolean
  onEdit?: () => void
}

export function AgentPromptNode({ data }: { data: AgentPromptNodeData }) {
  const [isHovered, setIsHovered] = useState(false)
  
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

  // Calculate minimum height based on number of input variables
  const minHeight = Math.max(100, (inputVariables.length + 2) * 40)
  
  return (
    <div 
      className="px-4 py-3 shadow-lg rounded-lg border-2 border-blue-500 bg-gray-800 min-w-[200px] transition-all hover:shadow-xl hover:border-blue-400 relative"
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
      
      {/* Additional input handles for each variable (for advanced routing) */}
      {inputVariables.map((varName, index) => {
        const handleTop = `${((index + 2) * 100) / (inputVariables.length + 2)}%`
        return (
          <Fragment key={`var-${varName}-${index}`}>
            <Handle
              type="target"
              position={Position.Left}
              id={varName}
              style={{
                top: handleTop,
                background: '#60a5fa',
              }}
              title={varName}
            />
            {/* Show variable name aligned with handle on hover */}
            {isHovered && (
              <div 
                className="absolute left-0 -translate-x-full pr-2"
                style={{ top: handleTop, transform: 'translate(-100%, -50%)' }}
              >
                <div className="text-xs text-blue-400 whitespace-nowrap">
                  {varName}
                </div>
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

      {/* Agent badge */}
      {data.agent && (
        <div className="flex items-center gap-1 text-xs px-2 py-1 bg-blue-600/30 border border-blue-500/50 rounded mb-2">
          <Users className="w-3 h-3 text-blue-400" />
          <span className="text-blue-300">{data.agent}</span>
        </div>
      )}

      {data.prompt && (
        <div className="text-xs text-gray-400 mt-2 max-w-[200px] truncate">
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
