import { Handle, Position } from 'reactflow'
import { MessageCircle, Play, Edit } from 'lucide-react'
import { useState } from 'react'

export interface ChatInputNodeData {
  label: string
  prompt_message?: string
  save_to?: string
  type: 'chatinput'
  isStartNode?: boolean
  isEndNode?: boolean
  onEdit?: () => void
}

export function ChatInputNode({ data }: { data: ChatInputNodeData }) {
  const [isHovered, setIsHovered] = useState(false)
  
  return (
    <div 
      className="px-4 py-3 shadow-lg rounded-lg border-2 border-green-500 bg-gray-800 min-w-[200px] transition-all hover:shadow-xl hover:border-green-400 relative"
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

      <div className="flex items-center gap-2 mb-2">
        <MessageCircle className="w-4 h-4 text-green-400" />
        <div className="font-bold text-sm text-gray-100 flex-1">{data.label}</div>
        {data.isStartNode && (
          <div className="flex items-center gap-1 text-xs px-2 py-0.5 bg-green-600 rounded-full">
            <Play className="w-3 h-3" />
            <span>START</span>
          </div>
        )}
      </div>

      <div className="text-xs text-green-300 font-semibold mb-1">
        INPUT NODE
      </div>

      {data.prompt_message && (
        <div className="text-xs text-gray-400 mt-2 max-w-[200px] truncate">
          {data.prompt_message}
        </div>
      )}

      {/* Show output variable name on hover */}
      {isHovered && (
        <div className="absolute right-0 top-1/2 -translate-y-1/2 translate-x-full pl-2">
          <div className="text-xs text-green-400 whitespace-nowrap">
            {data.save_to || 'user_input'}
          </div>
        </div>
      )}

      {/* Output handle for user input */}
      <Handle
        type="source"
        position={Position.Right}
        id="output"
        style={{
          background: '#10b981',
        }}
        title={data.save_to || 'user_input'}
      />
    </div>
  )
}
