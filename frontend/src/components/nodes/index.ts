import { PromptNode } from './PromptNode'
import { CustomNode } from './CustomNode'
import { ToolCallNode } from './ToolCallNode'
import { TextParseNode } from './TextParseNode'
import { AgentPromptNode } from './AgentPromptNode'
import { GetHistoryNode } from './GetHistoryNode'
import { SetHistoryNode } from './SetHistoryNode'
import { ChatInputNode } from './ChatInputNode'
import { ChatOutputNode } from './ChatOutputNode'
import { FileInputNode } from './FileInputNode'
import { FileOutputNode } from './FileOutputNode'

// Export individual components
export { 
  PromptNode, 
  CustomNode, 
  ToolCallNode, 
  TextParseNode, 
  AgentPromptNode, 
  GetHistoryNode, 
  SetHistoryNode,
  ChatInputNode,
  ChatOutputNode,
  FileInputNode,
  FileOutputNode,
}

// Export node types object for ReactFlow
export const nodeTypes = {
  prompt: PromptNode,
  custom: CustomNode,
  toolcall: ToolCallNode,
  textparse: TextParseNode,
  agentprompt: AgentPromptNode,
  gethistory: GetHistoryNode,
  sethistory: SetHistoryNode,
  chatinput: ChatInputNode,
  chatoutput: ChatOutputNode,
  fileinput: FileInputNode,
  fileoutput: FileOutputNode,
}
