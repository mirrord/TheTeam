import { useEffect, useState, useRef } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { useChatStore, Message } from '../store/chatStore'
import { useAgentStore } from '../store/agentStore'
import { useSocketStore } from '../store/socketStore'
import { Send, Plus, Trash2, Settings } from 'lucide-react'

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
    addMessage,
    sending,
    streamingMessages,
    startStreaming,
    appendStreamChunk,
    finalizeStreaming,
  } = useChatStore()
  
  const { agents, fetchAgents } = useAgentStore()
  const { emit, on, off, connected } = useSocketStore()
  
  const [inputMessage, setInputMessage] = useState('')
  const [selectedAgent, setSelectedAgent] = useState<string | undefined>()
  const [selectedModel, setSelectedModel] = useState<string>('')
  const [availableModels, setAvailableModels] = useState<string[]>([])
  const [showAgentSelector, setShowAgentSelector] = useState(false)
  const [isBaseModelMode, setIsBaseModelMode] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  
  useEffect(() => {
    fetchConversations()
    fetchAgents()
    fetchModels()
  }, [])
  
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
    }
  }, [currentConversation])
  
  // Auto-scroll to bottom whenever messages or streaming content change
  const activeStreaming = currentConversation ? streamingMessages[currentConversation.id] : null
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [currentConversation?.messages, activeStreaming?.content])
  
  // WebSocket handlers for real-time messages (non-streaming fallback + streaming)
  useEffect(() => {
    const handleMessageResponse = (data: any) => {
      if (currentConversation && data.conversation_id === currentConversation.id) {
        addMessage(data.message)
      }
    }
    
    const handleMessageError = (data: any) => {
      if (currentConversation && data.conversation_id === currentConversation.id) {
        console.error('Message error:', data.error)
      }
    }

    const handleStreamStart = (data: any) => {
      if (currentConversation && data.conversation_id === currentConversation.id) {
        startStreaming(data.conversation_id, data.message_id)
      }
    }

    const handleStreamChunk = (data: any) => {
      if (currentConversation && data.conversation_id === currentConversation.id) {
        appendStreamChunk(data.conversation_id, data.message_id, data.chunk)
      }
    }

    const handleStreamEnd = (data: any) => {
      if (currentConversation && data.conversation_id === currentConversation.id) {
        finalizeStreaming(data.conversation_id, data.message)
      }
    }
    
    on('message_response', handleMessageResponse)
    on('message_error', handleMessageError)
    on('stream_start', handleStreamStart)
    on('stream_chunk', handleStreamChunk)
    on('stream_end', handleStreamEnd)
    
    return () => {
      off('message_response', handleMessageResponse)
      off('message_error', handleMessageError)
      off('stream_start', handleStreamStart)
      off('stream_chunk', handleStreamChunk)
      off('stream_end', handleStreamEnd)
    }
  }, [currentConversation, on, off, addMessage, startStreaming, appendStreamChunk, finalizeStreaming])
  
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
    
    // Send via WebSocket — streaming is enabled by default
    emit('chat_message', {
      conversation_id: currentConversation.id,
      message: inputMessage,
      stream: true,
    })
    
    setInputMessage('')
  }
  
  const handleCreateConversation = async () => {
    const convId = await createConversation(selectedAgent, 'New Conversation')
    navigate(`/chat/${convId}`)
  }
  
  const handleDeleteConversation = async (convId: string) => {
    if (confirm('Are you sure you want to delete this conversation?')) {
      await deleteConversation(convId)
      if (currentConversation?.id === convId) {
        navigate('/chat')
      }
    }
  }
  
  const handleAgentChange = async (agentId: string) => {
    if (!currentConversation) return
    
    if (agentId === 'base_model') {
      // Switch to base model mode
      setIsBaseModelMode(true)
      setSelectedAgent(undefined)
      // Use the first available model if not already selected
      if (!selectedModel && availableModels.length > 0) {
        setSelectedModel(availableModels[0])
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
    
    // In base model mode, create a temporary agent with default parameters
    if (isBaseModelMode) {
      // Create a temporary agent config
      const tempAgentConfig = {
        id: `temp_${Date.now()}`,
        name: `Base Model - ${model}`,
        model: model,
        system_prompt: '',
        temperature: 0.7,
        max_tokens: -1,
        tools: [],
      }
      
      // Send config to use this model
      try {
        const response = await fetch(`/api/v1/chat/conversations/${currentConversation.id}/agent`, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ agent_config: tempAgentConfig }),
        })
        if (!response.ok) {
          console.error('Failed to set base model')
        }
      } catch (error) {
        console.error('Error setting base model:', error)
      }
    }
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
              <button
                onClick={() => navigate(`/chat/${conv.id}`)}
                className="w-full text-left px-3 py-2"
              >
                <div className="font-medium truncate">{conv.title}</div>
                <div className="text-xs text-gray-400">
                  {conv.message_count} messages
                </div>
              </button>
              <button
                onClick={() => handleDeleteConversation(conv.id)}
                className="absolute top-2 right-2 p-1 opacity-0 group-hover:opacity-100 hover:bg-red-600 rounded transition-opacity"
              >
                <Trash2 className="w-3 h-3" />
              </button>
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
              <button
                onClick={() => setShowAgentSelector(!showAgentSelector)}
                className="p-2 hover:bg-gray-700 rounded-lg"
              >
                <Settings className="w-5 h-5" />
              </button>
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
                
                {/* Show model selector only when Base Model is selected */}
                {isBaseModelMode && (
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

              {/* In-progress streaming message */}
              {activeStreaming && (
                <div className="flex justify-start">
                  <div className="max-w-[70%] rounded-lg px-4 py-2 bg-gray-800 text-gray-100">
                    <div className="whitespace-pre-wrap">
                      {activeStreaming.content}
                      <span className="inline-block w-2 h-4 ml-0.5 bg-gray-400 animate-pulse align-middle" />
                    </div>
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
    </div>
  )
}
