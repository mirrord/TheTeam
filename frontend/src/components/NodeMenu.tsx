import { MessageSquare, Code, Terminal, FileSearch, Users, History, FileEdit, MessageCircle, MessageSquareText, FileInput, FileOutput } from 'lucide-react'

export interface NodeOption {
  type: string
  label: string
  icon: React.ReactNode
  description: string
  defaultData: any
}

const nodeOptions: NodeOption[] = [
  {
    type: 'chatinput',
    label: 'Chat Input',
    icon: <MessageCircle className="w-4 h-4" />,
    description: 'Receive user input from chat',
    defaultData: {
      label: 'Chat Input',
      prompt_message: 'Enter your input:',
      save_to: 'user_input',
      type: 'chatinput',
    },
  },
  {
    type: 'chatoutput',
    label: 'Chat Output',
    icon: <MessageSquareText className="w-4 h-4" />,
    description: 'Display output in chat',
    defaultData: {
      label: 'Chat Output',
      source: 'current_input',
      type: 'chatoutput',
    },
  },
  {
    type: 'fileinput',
    label: 'File Input',
    icon: <FileInput className="w-4 h-4" />,
    description: 'Read data from a file',
    defaultData: {
      label: 'File Input',
      file_path: 'data/input.txt',
      save_to: 'file_content',
      type: 'fileinput',
    },
  },
  {
    type: 'fileoutput',
    label: 'File Output',
    icon: <FileOutput className="w-4 h-4" />,
    description: 'Write data to a file',
    defaultData: {
      label: 'File Output',
      file_path: 'data/output.txt',
      source: 'current_input',
      mode: 'w',
      type: 'fileoutput',
    },
  },
  {
    type: 'prompt',
    label: 'Prompt Node',
    icon: <MessageSquare className="w-4 h-4" />,
    description: 'Format a prompt with variables',
    defaultData: {
      label: 'New Prompt',
      prompt: 'Enter your prompt here with {variables}',
      type: 'prompt',
    },
  },
  {
    type: 'agentprompt',
    label: 'Agent Prompt',
    icon: <Users className="w-4 h-4" />,
    description: 'Execute a prompt using a specific agent',
    defaultData: {
      label: 'New Agent Prompt',
      agent: 'agent_name',
      prompt: 'Enter your prompt here with {variables}',
      type: 'agentprompt',
    },
  },
  {
    type: 'gethistory',
    label: 'Get History',
    icon: <History className="w-4 h-4" />,
    description: 'Get conversation history from an agent',
    defaultData: {
      label: 'Get History',
      agent: 'agent_name',
      save_to: 'agent_history',
      type: 'gethistory',
    },
  },
  {
    type: 'sethistory',
    label: 'Set History',
    icon: <FileEdit className="w-4 h-4" />,
    description: 'Set conversation history for an agent',
    defaultData: {
      label: 'Set History',
      agent: 'agent_name',
      history_from: 'agent_history',
      mode: 'replace',
      type: 'sethistory',
    },
  },
  {
    type: 'textparse',
    label: 'Text Parse',
    icon: <FileSearch className="w-4 h-4" />,
    description: 'Extract and set variables from text',
    defaultData: {
      label: 'New Parser',
      set: {},
      extraction: {},
      type: 'textparse',
    },
  },
  {
    type: 'custom',
    label: 'Custom Code',
    icon: <Code className="w-4 h-4" />,
    description: 'Execute custom Python code',
    defaultData: {
      label: 'New Custom',
      custom_code: '# Your Python code here\n# state is available',
      type: 'custom',
    },
  },
  {
    type: 'toolcall',
    label: 'Tool Call',
    icon: <Terminal className="w-4 h-4" />,
    description: 'Execute a command-line tool',
    defaultData: {
      label: 'New Tool',
      command: 'echo "Hello"',
      save_to: 'tool_result',
      type: 'toolcall',
    },
  },
]

interface NodeMenuProps {
  onAddNode: (type: string, data: any) => void
}

export function NodeMenu({ onAddNode }: NodeMenuProps) {
  return (
    <div className="bg-gray-800 border-r border-gray-700 w-64 p-4">
      <h3 className="text-sm font-semibold text-gray-300 mb-3">Node Palette</h3>
      <div className="space-y-2">
        {nodeOptions.map((option) => (
          <button
            key={option.type}
            onClick={() => onAddNode(option.type, option.defaultData)}
            className="w-full text-left px-3 py-2 rounded-lg bg-gray-700 hover:bg-gray-600 transition-colors"
          >
            <div className="flex items-start gap-2">
              <div className="mt-0.5 text-gray-400">{option.icon}</div>
              <div className="flex-1">
                <div className="text-sm font-medium text-gray-200">{option.label}</div>
                <div className="text-xs text-gray-400 mt-0.5">{option.description}</div>
              </div>
            </div>
          </button>
        ))}
      </div>
      <div className="mt-4 pt-4 border-t border-gray-700">
        <p className="text-xs text-gray-500">
          Click a node type to add it to the canvas. Drag to connect nodes by their handles.
        </p>
      </div>
    </div>
  )
}
