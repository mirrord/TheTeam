import React, { useEffect, useState, useRef } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { useChatStore, Message } from '../store/chatStore'
import { useAgentStore } from '../store/agentStore'
import { useSocketStore } from '../store/socketStore'
import { useConfirmStore } from '../store/confirmStore'
import { Send, Plus, Trash2, Settings, Wrench, MoreHorizontal, X, Eye } from 'lucide-react'
import ToolsSidebar, { ToolInfo } from '../components/ToolsSidebar'
import { ToolConfirmationModal } from '../components/ToolConfirmationModal'

export default function ChatInterface() {
  const { id } = useParams()
  const navigate = useNavigate()
  const {
    conversations,
    currentConversation,
    fetchConversations,
    getConversation,
    createConversation,
    deleteConversation,
    updateAgent,
    updateBaseModel,
    updateEnabledTools,
    renameConversation,
    addMessage,
    sending,
    processing,
    streamingMessages,
    setProcessing,
    startStreaming,
    appendStreamChunk,
    appendToolBlock,
    finalizeStreaming,
  } = useChatStore()
  
  const { agents, fetchAgents } = useAgentStore()
  const { emit, on, off, connected } = useSocketStore()
  const { setPendingConfirmation } = useConfirmStore()
  
  const [inputMessage, setInputMessage] = useState('')
  const [selectedAgent, setSelectedAgent] = useState<string | undefined>()
  const [selectedModel, setSelectedModel] = useState<string>('')
  const [availableModels, setAvailableModels] = useState<string[]>([])
  const [showAgentSelector, setShowAgentSelector] = useState(false)
  const [isBaseModelMode, setIsBaseModelMode] = useState(false)
  // Tools sidebar state
  const [showToolsSidebar, setShowToolsSidebar] = useState(false)
  const [availableTools, setAvailableTools] = useState<ToolInfo[]>([])
  const [enabledTools, setEnabledTools] = useState<string[]>([])
  // Live agent status (drives the "Thinking / Using tool ..." indicator)
  const [agentStatus, setAgentStatus] = useState<
    { status: string; detail?: string | null } | null
  >(null)
  // Options menu and modals for sidebar chat items
  const [openMenuId, setOpenMenuId] = useState<string | null>(null)
  const [renamingId, setRenamingId] = useState<string | null>(null)
  const [renameValue, setRenameValue] = useState('')
  const [deleteConfirmId, setDeleteConfirmId] = useState<string | null>(null)
  const [traceConvId, setTraceConvId] = useState<string | null>(null)
  const [traceData, setTraceData] = useState<{ systemPrompt?: string; convTitle?: string; messages: Array<{ id: string; role: string; content: string; timestamp: string }> } | null>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  // Tracks the last finalized message id to prevent late agent_status events
  // from re-activating the thinking indicator after streaming completes.
  const finalizedMessageIdRef = useRef<string | null>(null)
  const menuRef = useRef<HTMLDivElement>(null)
  
  useEffect(() => {
    fetchConversations()
    fetchAgents()
    fetchModels()
    fetchTools()
  }, [])

  const fetchTools = async () => {
    try {
      const response = await fetch('/api/v1/tools/')
      if (response.ok) {
        const data = await response.json()
        // Backend returns an array of {name, description, platform, source}
        const tools: ToolInfo[] = (data.tools || []).map((t: any) =>
          typeof t === 'string' ? { name: t } : t
        )
        setAvailableTools(tools)
      }
    } catch (error) {
      console.error('Error fetching tools:', error)
    }
  }
  
  const fetchModels = async () => {
    try {
      const response = await fetch('/api/v1/system/models')
      if (response.ok) {
        const data = await response.json()
        setAvailableModels(data.models)
        if (data.models.length > 0 && !selectedModel) {
          setSelectedModel(data.models[0])
        }
      }
    } catch (error) {
      console.error('Error fetching models:', error)
    }
  }
  
  useEffect(() => {
    if (id) {
      getConversation(id)
    }
  }, [id])
  
  useEffect(() => {
    if (currentConversation) {
      setSelectedAgent(currentConversation.agent_id)
      // Detect base-model mode from persisted conversation state
      const baseModel = (currentConversation as any).base_model as string | undefined
      if (baseModel) {
        setIsBaseModelMode(true)
        setSelectedModel(baseModel)
      } else {
        setIsBaseModelMode(false)
      }
      // Initialize enabled tools: persisted override > agent's tools > all available
      const persisted = (currentConversation as any).enabled_tools as
        | string[]
        | null
        | undefined
      if (persisted !== undefined && persisted !== null) {
        setEnabledTools(persisted)
      } else if (currentConversation.agent_id) {
        const agent = agents.find((a) => a.id === currentConversation.agent_id)
        const agentTools = (agent as any)?.tools as string[] | undefined
        if (Array.isArray(agentTools)) {
          setEnabledTools(agentTools)
        } else {
          setEnabledTools(availableTools.map((t) => t.name))
        }
      } else {
        setEnabledTools(availableTools.map((t) => t.name))
      }
    }
  }, [currentConversation, agents, availableTools])
  
  // Auto-scroll to bottom whenever messages or streaming content change
  const activeStreaming = currentConversation ? streamingMessages[currentConversation.id] : null
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [currentConversation?.messages, activeStreaming?.content])
  
  // WebSocket handlers for real-time messages (non-streaming fallback + streaming)
  useEffect(() => {
    console.log('🔧 Setting up socket handlers for conversation:', currentConversation?.id)
    
    const handleMessageResponse = (data: any) => {
      console.log('📨 message_response received:', data)
      if (currentConversation && data.conversation_id === currentConversation.id) {
        addMessage(data.message)
      }
    }
    
    const handleMessageError = (data: any) => {
      console.error('❌ message_error received:', data)
      if (currentConversation && data.conversation_id === currentConversation.id) {
        console.error('Message error:', data.error)
        setProcessing(data.conversation_id, false)
        setAgentStatus(null)
      }
    }

    const handleMessageProcessing = (data: any) => {
      console.log('⏳ message_processing received:', data)
      if (currentConversation && data.conversation_id === currentConversation.id) {
        setProcessing(data.conversation_id, true)
      }
    }

    const handleStreamStart = (data: any) => {
      console.log('🚀 stream_start received:', data)
      if (currentConversation && data.conversation_id === currentConversation.id) {
        startStreaming(data.conversation_id, data.message_id)
      }
    }

    const handleStreamChunk = (data: any) => {
      console.log('📦 stream_chunk received, length:', data.chunk?.length)
      if (currentConversation && data.conversation_id === currentConversation.id) {
        appendStreamChunk(data.conversation_id, data.message_id, data.chunk)
      }
    }

    const handleToolBlock = (data: any) => {
      if (currentConversation && data.conversation_id === currentConversation.id) {
        appendToolBlock(data.conversation_id, data.message_id, data.command || '', data.output || '')
      }
    }

    const handleStreamEnd = (data: any) => {
      console.log('🏁 stream_end received:', data)
      if (currentConversation && data.conversation_id === currentConversation.id) {
        // Record the finalized message id so any late-arriving agent_status
        // events for this message are silently dropped.
        if (data.message?.id) {
          finalizedMessageIdRef.current = data.message.id
        }
        finalizeStreaming(data.conversation_id, data.message)
        setAgentStatus(null)
      }
    }

    const handleAgentStatus = (data: any) => {
      if (currentConversation && data.conversation_id === currentConversation.id) {
        // Drop status updates that belong to an already-finalized message to
        // prevent the thinking indicator from lingering after stream_end.
        if (data.message_id && data.message_id === finalizedMessageIdRef.current) {
          return
        }
        setAgentStatus({ status: data.status, detail: data.detail })
      }
    }

    const handleToolConfirmationRequest = (data: any) => {
      setPendingConfirmation({
        requestId: data.request_id,
        command: data.command,
        conversationId: data.conversation_id,
        messageId: data.message_id,
      })
    }
    
    on('message_response', handleMessageResponse)
    on('message_error', handleMessageError)
    on('message_processing', handleMessageProcessing)
    on('stream_start', handleStreamStart)
    on('stream_chunk', handleStreamChunk)
    on('tool_block', handleToolBlock)
    on('stream_end', handleStreamEnd)
    on('agent_status', handleAgentStatus)
    on('tool_confirmation_request', handleToolConfirmationRequest)
    
    console.log('✅ Socket handlers registered')
    
    return () => {
      console.log('🧹 Cleaning up socket handlers')
      off('message_response', handleMessageResponse)
      off('message_error', handleMessageError)
      off('message_processing', handleMessageProcessing)
      off('stream_start', handleStreamStart)
      off('stream_chunk', handleStreamChunk)
      off('tool_block', handleToolBlock)
      off('stream_end', handleStreamEnd)
      off('agent_status', handleAgentStatus)
      off('tool_confirmation_request', handleToolConfirmationRequest)
    }
  }, [currentConversation, on, off, addMessage, setProcessing, startStreaming, appendStreamChunk, appendToolBlock, finalizeStreaming])

  // Close options menu when clicking outside of it
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setOpenMenuId(null)
      }
    }
    if (openMenuId) {
      document.addEventListener('mousedown', handler)
    }
    return () => document.removeEventListener('mousedown', handler)
  }, [openMenuId])

  const handleSendMessage = async () => {
    if (!inputMessage.trim() || !currentConversation || !connected) return
    
    // Optimistically add user message
    const userMessage: Message = {
      id: `temp-${Date.now()}`,
      role: 'user',
      content: inputMessage,
      timestamp: new Date().toISOString(),
    }
    addMessage(userMessage)
    
    // Set processing state immediately
    setProcessing(currentConversation.id, true)
    setAgentStatus({ status: 'thinking' })
    
    // Send via WebSocket — streaming is enabled by default. Forward the
    // current enabled-tools selection so the backend can filter accordingly.
    emit('chat_message', {
      conversation_id: currentConversation.id,
      message: inputMessage,
      stream: true,
      enabled_tools: enabledTools,
    })
    
    setInputMessage('')
  }
  
  const handleCreateConversation = async () => {
    const convId = await createConversation(selectedAgent, 'New Conversation')
    navigate(`/chat/${convId}`)
  }
  
  const handleDeleteConversation = async (convId: string) => {
    await deleteConversation(convId)
    if (currentConversation?.id === convId) {
      navigate('/chat')
    }
  }

  const handleStartRename = (conv: { id: string; title: string }) => {
    setRenamingId(conv.id)
    setRenameValue(conv.title)
    setOpenMenuId(null)
  }

  const handleConfirmRename = async () => {
    if (!renamingId) return
    const trimmed = renameValue.trim()
    if (trimmed) {
      await renameConversation(renamingId, trimmed)
    }
    setRenamingId(null)
  }

  const handleShowTrace = async (convId: string, agentId?: string | null, convTitle?: string) => {
    setOpenMenuId(null)
    try {
      const convResp = await fetch(`/api/v1/chat/conversations/${convId}`)
      if (!convResp.ok) return
      const convData = await convResp.json()
      let systemPrompt: string | undefined
      if (agentId) {
        try {
          const agentResp = await fetch(`/api/v1/agents/${agentId}`)
          if (agentResp.ok) {
            const agentData = await agentResp.json()
            systemPrompt = agentData?.agent?.config?.system_prompt
          }
        } catch {
          // system prompt unavailable — trace still shows
        }
      }
      setTraceData({ systemPrompt, convTitle, messages: convData.conversation.messages })
      setTraceConvId(convId)
    } catch {
      // silently ignore — trace unavailable
    }
  }


  const handleAgentChange = async (agentId: string) => {
    if (!currentConversation) return
    
    if (agentId === 'base_model') {
      // Switch to base model mode. Pick a model immediately and persist.
      setIsBaseModelMode(true)
      setSelectedAgent(undefined)
      const modelToUse = selectedModel || availableModels[0] || ''
      if (modelToUse) {
        setSelectedModel(modelToUse)
        try {
          await updateBaseModel(currentConversation.id, modelToUse)
        } catch (e) {
          console.error('Failed to set base model:', e)
        }
      }
    } else {
      // Regular agent mode
      setIsBaseModelMode(false)
      setSelectedAgent(agentId)
      await updateAgent(currentConversation.id, agentId)
    }
  }
  
  const handleModelChange = async (model: string) => {
    if (!currentConversation) return
    setSelectedModel(model)
    if (isBaseModelMode) {
      try {
        await updateBaseModel(currentConversation.id, model)
      } catch (e) {
        console.error('Failed to update base model:', e)
      }
    }
  }

  const handleToggleTool = (name: string) => {
    if (!currentConversation) return
    const next = enabledTools.includes(name)
      ? enabledTools.filter((n) => n !== name)
      : [...enabledTools, name]
    setEnabledTools(next)
    updateEnabledTools(currentConversation.id, next).catch((e) =>
      console.error('Failed to update tools:', e)
    )
  }

  const handleEnableAllTools = () => {
    if (!currentConversation) return
    const all = availableTools.map((t) => t.name)
    setEnabledTools(all)
    updateEnabledTools(currentConversation.id, all).catch(console.error)
  }

  const handleDisableAllTools = () => {
    if (!currentConversation) return
    setEnabledTools([])
    updateEnabledTools(currentConversation.id, []).catch(console.error)
  }
  
  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSendMessage()
    }
  }
  
  return (
    <div className="flex h-full">
      {/* Conversation list */}
      <div className="w-64 bg-gray-800 border-r border-gray-700 flex flex-col">
        <div className="p-4 border-b border-gray-700">
          <h2 className="text-lg font-semibold mb-2">Conversations</h2>
          <button
            onClick={handleCreateConversation}
            className="w-full flex items-center justify-center gap-2 px-3 py-2 bg-primary-600 hover:bg-primary-700 rounded-lg text-sm"
          >
            <Plus className="w-4 h-4" />
            New Chat
          </button>
        </div>
        
        <div className="flex-1 overflow-y-auto p-2">
          {conversations.map((conv) => (
            <div
              key={conv.id}
              className={`group relative mb-1 rounded-lg ${
                currentConversation?.id === conv.id
                  ? 'bg-primary-600'
                  : 'hover:bg-gray-700'
              }`}
            >
              {renamingId === conv.id ? (
                <input
                  value={renameValue}
                  onChange={(e) => setRenameValue(e.target.value)}
                  onBlur={handleConfirmRename}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') { e.preventDefault(); handleConfirmRename() }
                    if (e.key === 'Escape') { setRenamingId(null) }
                  }}
                  autoFocus
                  className="w-full px-3 py-2 bg-gray-700 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary-500"
                />
              ) : (
                <button
                  onClick={() => navigate(`/chat/${conv.id}`)}
                  className="w-full text-left px-3 py-2 pr-8"
                >
                  <div className="font-medium truncate">{conv.title}</div>
                  <div className="text-xs text-gray-400">
                    {conv.message_count} messages
                  </div>
                </button>
              )}
              {/* Options kebab button */}
              <button
                onClick={(e) => {
                  e.stopPropagation()
                  setOpenMenuId(openMenuId === conv.id ? null : conv.id)
                }}
                className="absolute top-2 right-1 p-1 opacity-0 group-hover:opacity-100 hover:bg-gray-600 rounded transition-opacity"
                title="Options"
              >
                <MoreHorizontal className="w-3 h-3" />
              </button>
              {/* Dropdown options menu */}
              {openMenuId === conv.id && (
                <div
                  ref={menuRef}
                  className="absolute right-0 top-8 bg-gray-700 border border-gray-600 rounded-lg shadow-xl z-50 min-w-[160px] py-1"
                >
                  <button
                    onClick={() => handleStartRename(conv)}
                    className="w-full text-left px-3 py-2 text-sm hover:bg-gray-600 flex items-center gap-2"
                  >
                    <span>Rename</span>
                  </button>
                  <button
                    onClick={() => handleShowTrace(conv.id, conv.agent_id, conv.title)}
                    className="w-full text-left px-3 py-2 text-sm hover:bg-gray-600 flex items-center gap-2"
                  >
                    <Eye className="w-3 h-3" />
                    <span>Show Full Trace</span>
                  </button>
                  <div className="border-t border-gray-600 my-1" />
                  <button
                    onClick={() => { setDeleteConfirmId(conv.id); setOpenMenuId(null) }}
                    className="w-full text-left px-3 py-2 text-sm hover:bg-red-600 text-red-300 hover:text-white flex items-center gap-2"
                  >
                    <Trash2 className="w-3 h-3" />
                    <span>Delete</span>
                  </button>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
      
      {/* Chat area */}
      <div className="flex-1 flex flex-col">
        {currentConversation ? (
          <>
            {/* Chat header */}
            <div className="bg-gray-800 border-b border-gray-700 p-4 flex items-center justify-between">
              <div>
                <h3 className="font-semibold">{currentConversation.title}</h3>
                <p className="text-sm text-gray-400">
                  {isBaseModelMode 
                    ? `Base Model - ${selectedModel || 'No model selected'}`
                    : selectedAgent 
                      ? agents.find(a => a.id === selectedAgent)?.name 
                      : 'No agent selected'
                  }
                </p>
              </div>
              <div className="flex items-center gap-1">
                <button
                  onClick={() => setShowToolsSidebar(true)}
                  className="p-2 hover:bg-gray-700 rounded-lg relative"
                  title="Manage tools"
                >
                  <Wrench className="w-5 h-5" />
                  {enabledTools.length > 0 && (
                    <span className="absolute -top-1 -right-1 text-[10px] bg-primary-600 rounded-full px-1.5 min-w-[18px] text-center">
                      {enabledTools.length}
                    </span>
                  )}
                </button>
                <button
                  onClick={() => setShowAgentSelector(!showAgentSelector)}
                  className="p-2 hover:bg-gray-700 rounded-lg"
                >
                  <Settings className="w-5 h-5" />
                </button>
              </div>
            </div>
            
            {/* Agent selector dropdown */}
            {showAgentSelector && (
              <div className="bg-gray-800 border-b border-gray-700 p-4 space-y-4">
                <div>
                  <label className="block text-sm font-medium mb-2">Select Agent:</label>
                  <select
                    value={isBaseModelMode ? 'base_model' : (selectedAgent || '')}
                    onChange={(e) => handleAgentChange(e.target.value)}
                    className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500"
                  >
                    <option value="base_model">Base Model</option>
                    {agents.map((agent) => (
                      <option key={agent.id} value={agent.id}>
                        {agent.name} ({agent.model})
                      </option>
                    ))}
                  </select>
                </div>
                
                {/* Show model selector when Base Model is selected (explicitly or by default) */}
                {(isBaseModelMode || !selectedAgent) && (
                  <div>
                    <label className="block text-sm font-medium mb-2">Select Model:</label>
                    <select
                      value={selectedModel}
                      onChange={(e) => handleModelChange(e.target.value)}
                      className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500"
                    >
                      {availableModels.map((model) => (
                        <option key={model} value={model}>
                          {model}
                        </option>
                      ))}
                    </select>
                  </div>
                )}
              </div>
            )}
            
            {/* Messages */}
            <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-gray-900">
              {currentConversation.messages.map((msg) => (
                <div
                  key={msg.id}
                  className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  <div
                    className={`max-w-[70%] rounded-lg px-4 py-2 ${
                      msg.role === 'user'
                        ? 'bg-primary-600 text-white'
                        : msg.role === 'system'
                        ? 'bg-gray-700 text-gray-300 italic'
                        : 'bg-gray-800 text-gray-100'
                    }`}
                  >
                    <div className="whitespace-pre-wrap">{msg.content}</div>
                    <div className="text-xs opacity-70 mt-1">
                      {new Date(msg.timestamp).toLocaleTimeString()}
                    </div>
                  </div>
                </div>
              ))}

              {/* Processing/thinking indicator - shows immediately when message is being processed */}
              {processing && !activeStreaming && (
                <div className="flex justify-start">
                  <div className="max-w-[70%] rounded-lg px-4 py-2 bg-gray-800 text-gray-100">
                    <div className="flex items-center gap-2">
                      <div className="flex gap-1">
                        <span className="inline-block w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                        <span className="inline-block w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                        <span className="inline-block w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                      </div>
                      <span className="text-sm text-gray-400">
                        {statusLabel(agentStatus)}
                      </span>
                    </div>
                  </div>
                </div>
              )}

              {/* In-progress streaming message */}
              {activeStreaming && (
                <div className="flex justify-start">
                  <div className="max-w-[70%] rounded-lg px-4 py-2 bg-gray-800 text-gray-100">
                    <div className="whitespace-pre-wrap">
                      {activeStreaming.content}
                      <span className="inline-block w-2 h-4 ml-0.5 bg-gray-400 animate-pulse align-middle" />
                    </div>
                    {activeStreaming.toolBlocks.map((tb, i) => (
                      <ToolBlockView key={i} command={tb.command} output={tb.output} />
                    ))}
                    {agentStatus &&
                      (agentStatus.status === 'tool_call' ||
                        agentStatus.status === 'tool_result') && (
                        <div className="text-xs text-gray-400 mt-2 italic">
                          {statusLabel(agentStatus)}
                        </div>
                      )}
                  </div>
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>
            
            {/* Input area */}
            <div className="bg-gray-800 border-t border-gray-700 p-4">
              <div className="flex gap-2">
                <textarea
                  value={inputMessage}
                  onChange={(e) => setInputMessage(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="Type a message..."
                  disabled={!connected || sending}
                  className="flex-1 px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500 resize-none disabled:opacity-50"
                  rows={3}
                />
                <button
                  onClick={handleSendMessage}
                  disabled={!inputMessage.trim() || !connected || sending}
                  className="px-4 py-2 bg-primary-600 hover:bg-primary-700 disabled:bg-gray-600 disabled:cursor-not-allowed rounded-lg"
                >
                  <Send className="w-5 h-5" />
                </button>
              </div>
              {!connected && (
                <p className="text-sm text-red-400 mt-2">
                  Not connected to server. Messages cannot be sent.
                </p>
              )}
            </div>
          </>
        ) : (
          <div className="flex-1 flex items-center justify-center text-gray-400">
            <div className="text-center">
              <p className="text-lg mb-2">No conversation selected</p>
              <p className="text-sm">Start a new conversation to begin chatting</p>
            </div>
          </div>
        )}
      </div>

      {/* Tool enable/disable sliding sidebar */}
      <ToolsSidebar
        isOpen={showToolsSidebar}
        onClose={() => setShowToolsSidebar(false)}
        tools={availableTools}
        enabledTools={enabledTools}
        onToggle={handleToggleTool}
        onEnableAll={handleEnableAllTools}
        onDisableAll={handleDisableAllTools}
      />

      {/* Tool call confirmation modal */}
      <ToolConfirmationModal />

      {/* Delete confirmation modal */}
      {deleteConfirmId && (
        <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50">
          <div className="bg-gray-800 rounded-xl p-6 max-w-sm w-full mx-4 space-y-4 border border-gray-700 shadow-2xl">
            <h3 className="text-lg font-semibold flex items-center gap-2 text-red-300">
              <Trash2 className="w-5 h-5" />
              Delete Conversation
            </h3>
            <p className="text-sm text-gray-300">
              Delete &ldquo;{conversations.find((c) => c.id === deleteConfirmId)?.title}&rdquo;?
              This action cannot be undone.
            </p>
            <div className="flex justify-end gap-2">
              <button
                onClick={() => setDeleteConfirmId(null)}
                className="px-4 py-2 rounded-lg bg-gray-700 hover:bg-gray-600 text-sm"
              >
                Cancel
              </button>
              <button
                onClick={() => {
                  const id = deleteConfirmId
                  setDeleteConfirmId(null)
                  handleDeleteConversation(id)
                }}
                className="px-4 py-2 rounded-lg bg-red-600 hover:bg-red-500 text-sm"
              >
                Delete
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Full trace modal */}
      {traceConvId && traceData && (
        <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50">
          <div className="bg-gray-900 border border-gray-700 rounded-xl w-full max-w-3xl max-h-[85vh] flex flex-col mx-4 shadow-2xl">
            {/* Header */}
            <div className="flex items-center justify-between p-4 border-b border-gray-700 flex-shrink-0">
              <div>
                <h3 className="font-semibold flex items-center gap-2">
                  <Eye className="w-4 h-4 text-primary-400" />
                  Full Conversation Trace
                </h3>
                {traceData.convTitle && (
                  <p className="text-xs text-gray-400 mt-0.5">{traceData.convTitle}</p>
                )}
              </div>
              <button
                onClick={() => { setTraceConvId(null); setTraceData(null) }}
                className="p-2 hover:bg-gray-700 rounded-lg"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
            {/* Scrollable body */}
            <div className="flex-1 overflow-y-auto p-4 space-y-3">
              {/* System prompt block */}
              {traceData.systemPrompt && (
                <div className="rounded-lg border border-amber-800/50 bg-amber-950/30 px-4 py-3">
                  <div className="text-xs font-semibold text-amber-400 mb-1 uppercase tracking-wide">
                    System Prompt
                  </div>
                  <div className="text-sm text-amber-200/80 whitespace-pre-wrap">
                    {traceData.systemPrompt}
                  </div>
                </div>
              )}
              {/* Messages */}
              {traceData.messages.map((msg) => (
                <div key={msg.id} className="space-y-1">
                  <div className="flex items-center gap-2">
                    <span className={`text-xs font-semibold uppercase tracking-wide px-2 py-0.5 rounded ${
                      msg.role === 'user'
                        ? 'bg-primary-900/60 text-primary-300'
                        : msg.role === 'system'
                        ? 'bg-amber-900/60 text-amber-300'
                        : 'bg-gray-700 text-gray-300'
                    }`}>
                      {msg.role}
                    </span>
                    <span className="text-xs text-gray-500">
                      {new Date(msg.timestamp).toLocaleTimeString()}
                    </span>
                  </div>
                  {/* Render assistant messages: split think blocks from response */}
                  {msg.role === 'assistant'
                    ? renderTraceAssistantContent(msg.content)
                    : (
                      <div className="text-sm text-gray-200 whitespace-pre-wrap pl-1">
                        {msg.content}
                      </div>
                    )
                  }
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

/** Styled display block for a single tool call + output. */
function ToolBlockView({ command, output }: { command: string; output: string }) {
  const lines = output ? output.trimEnd().split('\n') : []
  return (
    <div className="my-2 rounded border border-gray-600 bg-gray-900 text-sm font-mono overflow-x-auto">
      <div className="flex items-center gap-2 px-3 py-1.5 bg-gray-700/60 border-b border-gray-600">
        <span className="text-green-400 select-none">$</span>
        <span className="text-gray-100">{command}</span>
      </div>
      <div className="px-3 py-2 text-gray-300">
        {lines.length > 0
          ? lines.map((line, i) => <div key={i}>{line || '\u00a0'}</div>)
          : <span className="text-gray-500 italic">no output</span>
        }
      </div>
    </div>
  )
}

/** Split an assistant message into think blocks and regular content for the trace view. */
function renderTraceAssistantContent(content: string): React.ReactNode {
  // Match <think>...</think> blocks (possibly multiline)
  const parts: React.ReactNode[] = []
  const regex = /<think>([\s\S]*?)<\/think>/g
  let lastIndex = 0
  let match: RegExpExecArray | null

  while ((match = regex.exec(content)) !== null) {
    // Text before the think block
    if (match.index > lastIndex) {
      const before = content.slice(lastIndex, match.index)
      if (before.trim()) {
        parts.push(
          <div key={`text-${lastIndex}`} className="text-sm text-gray-200 whitespace-pre-wrap pl-1">
            {before}
          </div>
        )
      }
    }
    // Think block
    parts.push(
      <div
        key={`think-${match.index}`}
        className="text-sm text-gray-400 whitespace-pre-wrap pl-3 border-l-2 border-amber-700/60 bg-gray-800/60 rounded-r py-1 pr-2 italic"
      >
        <span className="text-xs text-amber-500/80 font-semibold not-italic uppercase tracking-wide mr-1">thinking:</span>
        {match[1]}
      </div>
    )
    lastIndex = match.index + match[0].length
  }

  // Remaining text after last think block
  if (lastIndex < content.length) {
    const remaining = content.slice(lastIndex)
    if (remaining.trim()) {
      parts.push(
        <div key={`text-end`} className="text-sm text-gray-200 whitespace-pre-wrap pl-1">
          {remaining}
        </div>
      )
    }
  }

  // If no think blocks found, render as plain text
  if (parts.length === 0) {
    return <div className="text-sm text-gray-200 whitespace-pre-wrap pl-1">{content}</div>
  }

  return <div className="space-y-1">{parts}</div>
}

function statusLabel(
  status: { status: string; detail?: string | null } | null
): string {
  if (!status) return 'Thinking...'
  switch (status.status) {
    case 'thinking':
      return 'Thinking...'
    case 'tool_call':
      return status.detail ? `Using tool: ${status.detail}...` : 'Using tool...'
    case 'tool_result':
      return 'Processing tool result...'
    case 'generating':
      return 'Generating...'
    default:
      return status.detail ? `${status.status}: ${status.detail}` : status.status
  }
}
