import { Handle, Position } from 'reactflow'
import { FileSearch, Play, Flag, Edit } from 'lucide-react'
import { useState } from 'react'

export interface TextParseNodeData {
  label: string
  extraction?: Record<string, string>
  set?: Record<string, string>
  transform?: string
  type: 'textparse'
  isStartNode?: boolean
  isEndNode?: boolean
  onEdit?: () => void
}

export function TextParseNode({ data }: { data: TextParseNodeData }) {
  const [isHovered, setIsHovered] = useState(false)
  const extractionKeys = data.extraction ? Object.keys(data.extraction) : []
  const setKeys = data.set ? Object.keys(data.set) : []
  const hasOperations = extractionKeys.length > 0 || setKeys.length > 0

  return (
    <div 
      className="px-4 py-3 shadow-lg rounded-lg border-2 border-amber-500 bg-gray-800 min-w-[200px] min-h-[100px] transition-all hover:shadow-xl hover:border-amber-400 relative"
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
          background: '#f59e0b',
        }}
        title="input"
      />

      <div className="flex items-center gap-2 mb-2">
        <FileSearch className="w-4 h-4 text-amber-400" />
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

      {hasOperations && (
        <div className="text-xs text-gray-400 mt-2">
          {extractionKeys.length > 0 && (
            <div className="mb-1">
              <span className="text-amber-400">Extract:</span> {extractionKeys.join(', ')}
            </div>
          )}
          {setKeys.length > 0 && (
            <div>
              <span className="text-amber-400">Set:</span> {setKeys.join(', ')}
            </div>
          )}
        </div>
      )}

      {data.transform && (
        <div className="text-xs text-gray-500 mt-1 italic">
          Transform: {data.transform}
        </div>
      )}
      
      {/* Show input variable name on hover */}
      {isHovered && (
        <div className="absolute left-0 top-1/2 -translate-y-1/2 -translate-x-full pr-2">
          <div className="text-xs text-amber-400 whitespace-nowrap">
            input
          </div>
        </div>
      )}
      
      {/* Show output variable name on hover */}
      {isHovered && (
        <div className="absolute right-0 top-1/2 -translate-y-1/2 translate-x-full pl-2">
          <div className="text-xs text-green-400 whitespace-nowrap">
            output
          </div>
        </div>
      )}

      {/* Output handle */}
      <Handle
        type="source"
        position={Position.Right}
        id="output"
        style={{
          background: '#10b981',
        }}
        title="output"
      />
    </div>
  )
}
