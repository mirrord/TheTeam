import { Handle, Position } from 'reactflow'
import { MessageSquareText, Flag, Edit } from 'lucide-react'
import { useState } from 'react'

export interface ChatOutputNodeData {
  label: string
  source?: string
  format_template?: string
  type: 'chatoutput'
  isStartNode?: boolean
  isEndNode?: boolean
  onEdit?: () => void
}

export function ChatOutputNode({ data }: { data: ChatOutputNodeData }) {
  const [isHovered, setIsHovered] = useState(false)
  
  return (
    <div 
      className="px-4 py-3 shadow-lg rounded-lg border-2 border-purple-500 bg-gray-800 min-w-[200px] transition-all hover:shadow-xl hover:border-purple-400 relative"
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

      {/* Input handle */}
      <Handle
        type="target"
        position={Position.Left}
        id="input"
        style={{
          background: '#a855f7',
        }}
        title="input"
      />

      {/* Show input variable name on hover */}
      {isHovered && (
        <div className="absolute left-0 top-1/2 -translate-y-1/2 -translate-x-full pr-2">
          <div className="text-xs text-purple-400 whitespace-nowrap">
            {data.source || 'current_input'}
          </div>
        </div>
      )}

      <div className="flex items-center gap-2 mb-2">
        <MessageSquareText className="w-4 h-4 text-purple-400" />
        <div className="font-bold text-sm text-gray-100 flex-1">{data.label}</div>
        {data.isEndNode && (
          <div className="flex items-center gap-1 text-xs px-2 py-0.5 bg-red-600 rounded-full">
            <Flag className="w-3 h-3" />
            <span>END</span>
          </div>
        )}
      </div>

      <div className="text-xs text-purple-300 font-semibold mb-1">
        OUTPUT NODE
      </div>

      {data.format_template && (
        <div className="text-xs text-gray-400 mt-2 max-w-[200px] truncate">
          Template: {data.format_template}
        </div>
      )}

      {/* Output handle for chaining */}
      <Handle
        type="source"
        position={Position.Right}
        id="output"
        style={{
          background: '#a855f7',
        }}
        title="output"
      />
    </div>
  )
}
