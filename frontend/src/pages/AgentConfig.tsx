import { useEffect, useState } from 'react'
import { useAgentStore, Agent, AgentConfig as AgentConfigType } from '../store/agentStore'
import { useFlowchartStore } from '../store/flowchartStore'
import { Plus, Edit2, Trash2, Play, Save, X } from 'lucide-react'

export default function AgentConfig() {
  const { agents, fetchAgents, createAgent, updateAgent, deleteAgent, testAgent, loading } = useAgentStore()
  const { flowcharts, fetchFlowcharts } = useFlowchartStore()
  
  const [selectedAgent, setSelectedAgent] = useState<Agent | null>(null)
  const [isEditing, setIsEditing] = useState(false)
  const [isCreating, setIsCreating] = useState(false)
  const [editingConfig, setEditingConfig] = useState<AgentConfigType | null>(null)
  const [testPrompt, setTestPrompt] = useState('')
  const [testResult, setTestResult] = useState<any>(null)
  const [testLoading, setTestLoading] = useState(false)
  const [availableTools, setAvailableTools] = useState<string[]>([])
  const [availableModels, setAvailableModels] = useState<string[]>([])
  const [agentType, setAgentType] = useState<'base_model' | 'custom'>('base_model')
  
  useEffect(() => {
    fetchAgents()
    fetchFlowcharts()
    fetchAvailableTools()
    fetchAvailableModels()
  }, [])
  
  const fetchAvailableTools = async () => {
    try {
      const response = await fetch('/api/v1/tools/')
      if (response.ok) {
        const data = await response.json()
        setAvailableTools(data.tools || [])
      }
    } catch (error) {
      console.error('Error fetching tools:', error)
    }
  }
  
  const fetchAvailableModels = async () => {
    try {
      const response = await fetch('/api/v1/system/models')
      if (response.ok) {
        const data = await response.json()
        setAvailableModels(data.models || [])
      }
    } catch (error) {
      console.error('Error fetching models:', error)
    }
  }
  
  const handleCreate = () => {
    setIsCreating(true)
    setIsEditing(true)
    setAgentType('base_model')
    setEditingConfig({
      id: '',
      name: 'New Agent',
      model: 'glm-4.7-flash:latest',
      system_prompt: '',
      temperature: 0.7,
      max_tokens: 2048,
      tools: [],
      flowchart: '',
      tool_auto_loop: false,
      tool_max_iterations: 5,
      memory_enabled: false,
      current_context: 'default',
      compaction: {
        enabled: false,
        threshold: 20,
        keep_last: 6,
        summary_model: null,
        memory_category: 'context_summaries',
        summary_max_tokens: 512,
      },
      recall: {
        enabled: false,
        sources: ['memory', 'history'],
        n_results: 5,
        recall_model: null,
        categories: [],
        min_relevance: 0.5,
      },
    })
  }
  
  const handleEdit = (agent: Agent) => {
    setSelectedAgent(agent)
    setIsEditing(true)
    setAgentType('base_model') // Default to base_model type
    // Fetch full config
    fetch(`/api/v1/agents/${agent.id}`)
      .then(res => res.json())
      .then(data => setEditingConfig(data.agent.config))
  }
  
  const handleSave = async () => {
    if (!editingConfig) return
    
    try {
      if (isCreating) {
        const newId = await createAgent(editingConfig)
        setSelectedAgent({ ...selectedAgent!, id: newId })
        setIsCreating(false)
      } else if (selectedAgent) {
        await updateAgent(selectedAgent.id, editingConfig)
      }
      setIsEditing(false)
      setEditingConfig(null)
    } catch (error) {
      console.error('Error saving agent:', error)
    }
  }
  
  const handleCancel = () => {
    setIsEditing(false)
    setIsCreating(false)
    setEditingConfig(null)
  }
  
  const handleDelete = async (agent: Agent) => {
    if (confirm(`Are you sure you want to delete ${agent.name}?`)) {
      await deleteAgent(agent.id)
      if (selectedAgent?.id === agent.id) {
        setSelectedAgent(null)
      }
    }
  }
  
  const handleTest = async () => {
    if (!selectedAgent || !testPrompt.trim()) return
    
    setTestLoading(true)
    try {
      const result = await testAgent(selectedAgent.id, testPrompt)
      setTestResult(result)
    } catch (error) {
      console.error('Error testing agent:', error)
    } finally {
      setTestLoading(false)
    }
  }
  
  const updateConfig = (field: string, value: any) => {
    if (editingConfig) {
      setEditingConfig({ ...editingConfig, [field]: value })
    }
  }
  
  return (
    <div className="flex h-full">
      {/* Agent list */}
      <div className="w-64 bg-gray-800 border-r border-gray-700 flex flex-col">
        <div className="p-4 border-b border-gray-700">
          <h2 className="text-lg font-semibold mb-2">Agents</h2>
          <button
            onClick={handleCreate}
            className="w-full flex items-center justify-center gap-2 px-3 py-2 bg-primary-600 hover:bg-primary-700 rounded-lg text-sm"
          >
            <Plus className="w-4 h-4" />
            New Agent
          </button>
        </div>
        
        <div className="flex-1 overflow-y-auto p-2">
          {agents.map((agent) => (
            <div
              key={agent.id}
              className={`group relative mb-1 rounded-lg ${
                selectedAgent?.id === agent.id
                  ? 'bg-primary-600'
                  : 'hover:bg-gray-700'
              }`}
            >
              <button
                onClick={() => setSelectedAgent(agent)}
                className="w-full text-left px-3 py-2"
              >
                <div className="font-medium truncate">{agent.name}</div>
                <div className="text-xs text-gray-400">{agent.model}</div>
                <div className="text-xs text-gray-500">{agent.source}</div>
              </button>
              <div className="absolute top-2 right-2 flex gap-1 opacity-0 group-hover:opacity-100">
                <button
                  onClick={(e) => {
                    e.stopPropagation()
                    handleEdit(agent)
                  }}
                  className="p-1 hover:bg-blue-600 rounded transition-opacity"
                >
                  <Edit2 className="w-3 h-3" />
                </button>
                <button
                  onClick={(e) => {
                    e.stopPropagation()
                    handleDelete(agent)
                  }}
                  className="p-1 hover:bg-red-600 rounded transition-opacity"
                >
                  <Trash2 className="w-3 h-3" />
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>
      
      {/* Agent details/editor */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {selectedAgent || isCreating ? (
          <>
            {/* Header */}
            <div className="bg-gray-800 border-b border-gray-700 p-4 flex items-center justify-between">
              <h3 className="text-lg font-semibold">
                {isCreating ? 'New Agent' : isEditing ? 'Edit Agent' : 'Agent Configuration'}
              </h3>
              <div className="flex gap-2">
                {isEditing ? (
                  <>
                    <button
                      onClick={handleSave}
                      disabled={loading}
                      className="flex items-center gap-2 px-3 py-2 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 rounded-lg text-sm"
                    >
                      <Save className="w-4 h-4" />
                      Save
                    </button>
                    <button
                      onClick={handleCancel}
                      className="flex items-center gap-2 px-3 py-2 bg-gray-600 hover:bg-gray-700 rounded-lg text-sm"
                    >
                      <X className="w-4 h-4" />
                      Cancel
                    </button>
                  </>
                ) : (
                  <button
                    onClick={() => handleEdit(selectedAgent!)}
                    className="flex items-center gap-2 px-3 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg text-sm"
                  >
                    <Edit2 className="w-4 h-4" />
                    Edit
                  </button>
                )}
              </div>
            </div>
            
            {/* Configuration form */}
            <div className="flex-1 overflow-y-auto p-6 bg-gray-900">
              {isEditing && editingConfig ? (
                <div className="max-w-2xl space-y-6">
                  {/* Basic Info Section */}
                  <div className="space-y-4">
                    <h4 className="text-md font-semibold border-b border-gray-700 pb-2">Basic Information</h4>
                    
                    <div>
                      <label className="block text-sm font-medium mb-2">ID</label>
                      <input
                        type="text"
                        value={editingConfig.id || ''}
                        onChange={(e) => updateConfig('id', e.target.value)}
                        disabled={!isCreating}
                        className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500 disabled:opacity-50"
                      />
                    </div>
                    
                    <div>
                      <label className="block text-sm font-medium mb-2">Name</label>
                      <input
                        type="text"
                        value={editingConfig.name || ''}
                        onChange={(e) => updateConfig('name', e.target.value)}
                        className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500"
                      />
                    </div>
                    
                    <div>
                      <label className="block text-sm font-medium mb-2">Agent Type</label>
                      <select
                        value={agentType}
                        onChange={(e) => setAgentType(e.target.value as 'base_model' | 'custom')}
                        className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500"
                      >
                        <option value="base_model">Base Model</option>
                        <option value="custom">Custom</option>
                      </select>
                    </div>
                    
                    {agentType === 'base_model' && (
                      <div>
                        <label className="block text-sm font-medium mb-2">Model</label>
                        <select
                          value={editingConfig.model}
                          onChange={(e) => updateConfig('model', e.target.value)}
                          className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500"
                        >
                          {availableModels.map((model) => (
                            <option key={model} value={model}>{model}</option>
                          ))}
                        </select>
                      </div>
                    )}
                    
                    <div>
                      <label className="block text-sm font-medium mb-2">System Prompt</label>
                      <textarea
                        value={editingConfig.system_prompt || ''}
                        onChange={(e) => updateConfig('system_prompt', e.target.value)}
                        rows={6}
                        className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500"
                      />
                    </div>
                    
                    <div>
                      <label className="block text-sm font-medium mb-2">Current Context</label>
                      <input
                        type="text"
                        value={editingConfig.current_context || 'default'}
                        onChange={(e) => updateConfig('current_context', e.target.value)}
                        className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500"
                      />
                      <p className="text-xs text-gray-500 mt-1">
                        Name of the active conversation context
                      </p>
                    </div>
                  </div>
                  
                  {/* Generation Parameters Section */}
                  <div className="space-y-4">
                    <h4 className="text-md font-semibold border-b border-gray-700 pb-2">Generation Parameters</h4>
                    
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <label className="block text-sm font-medium mb-2">Temperature</label>
                        <input
                          type="number"
                          step="0.1"
                          min="0"
                          max="2"
                          value={editingConfig.temperature || 0.7}
                          onChange={(e) => updateConfig('temperature', parseFloat(e.target.value))}
                          className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500"
                        />
                      </div>
                      
                      <div>
                        <label className="block text-sm font-medium mb-2">Max Tokens</label>
                        <input
                          type="number"
                          value={editingConfig.max_tokens || 2048}
                          onChange={(e) => updateConfig('max_tokens', parseInt(e.target.value))}
                          className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500"
                        />
                        <p className="text-xs text-gray-500 mt-1">Use -1 for unlimited</p>
                      </div>
                    </div>
                  </div>
                  
                  {/* Tools Section */}
                  <div className="space-y-4">
                    <h4 className="text-md font-semibold border-b border-gray-700 pb-2">Tools & Capabilities</h4>
                    
                    <div>
                      <label className="block text-sm font-medium mb-2">Available Tools</label>
                      <div className="bg-gray-800 border border-gray-700 rounded-lg p-3 max-h-40 overflow-y-auto">
                        {availableTools.length > 0 ? (
                          availableTools.map((tool) => (
                            <label key={tool} className="flex items-center gap-2 mb-2">
                              <input
                                type="checkbox"
                                checked={(editingConfig.tools || []).includes(tool)}
                                onChange={(e) => {
                                  const currentTools = editingConfig.tools || []
                                  if (e.target.checked) {
                                    updateConfig('tools', [...currentTools, tool])
                                  } else {
                                    updateConfig('tools', currentTools.filter(t => t !== tool))
                                  }
                                }}
                                className="w-4 h-4"
                              />
                              <span className="text-sm">{tool}</span>
                            </label>
                          ))
                        ) : (
                          <p className="text-sm text-gray-500">No tools available</p>
                        )}
                      </div>
                    </div>
                    
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <label className="flex items-center gap-2">
                          <input
                            type="checkbox"
                            checked={editingConfig.tool_auto_loop || false}
                            onChange={(e) => updateConfig('tool_auto_loop', e.target.checked)}
                            className="w-4 h-4"
                          />
                          <span className="text-sm">Tool Auto Loop</span>
                        </label>
                      </div>
                      
                      <div>
                        <label className="block text-sm font-medium mb-2">Max Tool Iterations</label>
                        <input
                          type="number"
                          min="1"
                          max="20"
                          value={editingConfig.tool_max_iterations || 5}
                          onChange={(e) => updateConfig('tool_max_iterations', parseInt(e.target.value))}
                          className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500"
                        />
                      </div>
                    </div>
                    
                    <div>
                      <label className="flex items-center gap-2">
                        <input
                          type="checkbox"
                          checked={editingConfig.memory_enabled || false}
                          onChange={(e) => updateConfig('memory_enabled', e.target.checked)}
                          className="w-4 h-4"
                        />
                        <span className="text-sm">Enable Memory Tools</span>
                      </label>
                    </div>
                  </div>
                  
                  {/* Flowchart Section */}
                  <div className="space-y-4">
                    <h4 className="text-md font-semibold border-b border-gray-700 pb-2">Chain-of-Thought Flowchart</h4>
                    
                    <div>
                      <label className="block text-sm font-medium mb-2">Select Flowchart (Optional)</label>
                      <select
                        value={editingConfig.flowchart || ''}
                        onChange={(e) => updateConfig('flowchart', e.target.value)}
                        className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500"
                      >
                        <option value="">None</option>
                        {flowcharts.map((fc) => (
                          <option key={fc.id} value={fc.id}>
                            {fc.name}
                          </option>
                        ))}
                      </select>
                      <p className="text-xs text-gray-500 mt-1">
                        Chain-of-thought flowcharts guide the agent's reasoning process
                      </p>
                    </div>
                  </div>
                  
                  {/* Compaction Section */}
                  <div className="space-y-4">
                    <h4 className="text-md font-semibold border-b border-gray-700 pb-2">Context Compaction</h4>
                    
                    <div>
                      <label className="flex items-center gap-2">
                        <input
                          type="checkbox"
                          checked={editingConfig.compaction?.enabled || false}
                          onChange={(e) => updateConfig('compaction', {
                            ...editingConfig.compaction,
                            enabled: e.target.checked
                          })}
                          className="w-4 h-4"
                        />
                        <span className="text-sm">Enable Context Compaction</span>
                      </label>
                      <p className="text-xs text-gray-500 mt-1">
                        Automatically summarize old messages when context grows too large
                      </p>
                    </div>
                    
                    {editingConfig.compaction?.enabled && (
                      <div className="grid grid-cols-2 gap-4 ml-6">
                        <div>
                          <label className="block text-sm font-medium mb-2">Threshold</label>
                          <input
                            type="number"
                            min="1"
                            value={editingConfig.compaction?.threshold || 20}
                            onChange={(e) => updateConfig('compaction', {
                              ...editingConfig.compaction,
                              threshold: parseInt(e.target.value)
                            })}
                            className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500"
                          />
                          <p className="text-xs text-gray-500 mt-1">Messages before compacting</p>
                        </div>
                        
                        <div>
                          <label className="block text-sm font-medium mb-2">Keep Last</label>
                          <input
                            type="number"
                            min="1"
                            value={editingConfig.compaction?.keep_last || 6}
                            onChange={(e) => updateConfig('compaction', {
                              ...editingConfig.compaction,
                              keep_last: parseInt(e.target.value)
                            })}
                            className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500"
                          />
                          <p className="text-xs text-gray-500 mt-1">Recent messages to preserve</p>
                        </div>
                        
                        <div>
                          <label className="block text-sm font-medium mb-2">Memory Category</label>
                          <input
                            type="text"
                            value={editingConfig.compaction?.memory_category || 'context_summaries'}
                            onChange={(e) => updateConfig('compaction', {
                              ...editingConfig.compaction,
                              memory_category: e.target.value
                            })}
                            className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500"
                          />
                        </div>
                        
                        <div>
                          <label className="block text-sm font-medium mb-2">Summary Max Tokens</label>
                          <input
                            type="number"
                            value={editingConfig.compaction?.summary_max_tokens || 512}
                            onChange={(e) => updateConfig('compaction', {
                              ...editingConfig.compaction,
                              summary_max_tokens: parseInt(e.target.value)
                            })}
                            className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500"
                          />
                        </div>
                      </div>
                    )}
                  </div>
                  
                  {/* Recall Section */}
                  <div className="space-y-4">
                    <h4 className="text-md font-semibold border-b border-gray-700 pb-2">Memory Recall</h4>
                    
                    <div>
                      <label className="flex items-center gap-2">
                        <input
                          type="checkbox"
                          checked={editingConfig.recall?.enabled || false}
                          onChange={(e) => updateConfig('recall', {
                            ...editingConfig.recall,
                            enabled: e.target.checked
                          })}
                          className="w-4 h-4"
                        />
                        <span className="text-sm">Enable Memory Recall</span>
                      </label>
                      <p className="text-xs text-gray-500 mt-1">
                        Automatically retrieve relevant memories before each response
                      </p>
                    </div>
                    
                    {editingConfig.recall?.enabled && (
                      <div className="grid grid-cols-2 gap-4 ml-6">
                        <div>
                          <label className="block text-sm font-medium mb-2">Max Results</label>
                          <input
                            type="number"
                            min="1"
                            value={editingConfig.recall?.n_results || 5}
                            onChange={(e) => updateConfig('recall', {
                              ...editingConfig.recall,
                              n_results: parseInt(e.target.value)
                            })}
                            className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500"
                          />
                          <p className="text-xs text-gray-500 mt-1">Max memories to retrieve</p>
                        </div>
                        
                        <div>
                          <label className="block text-sm font-medium mb-2">Min Relevance</label>
                          <input
                            type="number"
                            step="0.1"
                            min="0"
                            max="1"
                            value={editingConfig.recall?.min_relevance || 0.5}
                            onChange={(e) => updateConfig('recall', {
                              ...editingConfig.recall,
                              min_relevance: parseFloat(e.target.value)
                            })}
                            className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500"
                          />
                          <p className="text-xs text-gray-500 mt-1">Minimum relevance score (0-1)</p>
                        </div>
                        
                        <div className="col-span-2">
                          <label className="block text-sm font-medium mb-2">Memory Categories</label>
                          <input
                            type="text"
                            value={editingConfig.recall?.categories?.join(', ') || ''}
                            onChange={(e) => updateConfig('recall', {
                              ...editingConfig.recall,
                              categories: e.target.value ? e.target.value.split(',').map(s => s.trim()) : []
                            })}
                            placeholder="Leave empty to search all categories"
                            className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500"
                          />
                          <p className="text-xs text-gray-500 mt-1">Comma-separated list of categories to search</p>
                        </div>
                        
                        <div className="col-span-2">
                          <label className="block text-sm font-medium mb-2">Sources</label>
                          <div className="flex gap-4">
                            <label className="flex items-center gap-2">
                              <input
                                type="checkbox"
                                checked={editingConfig.recall?.sources?.includes('memory') || false}
                                onChange={(e) => {
                                  const sources = editingConfig.recall?.sources || [];
                                  updateConfig('recall', {
                                    ...editingConfig.recall,
                                    sources: e.target.checked 
                                      ? [...sources.filter(s => s !== 'memory'), 'memory']
                                      : sources.filter(s => s !== 'memory')
                                  });
                                }}
                                className="w-4 h-4"
                              />
                              <span className="text-sm">Memory Store</span>
                            </label>
                            <label className="flex items-center gap-2">
                              <input
                                type="checkbox"
                                checked={editingConfig.recall?.sources?.includes('history') || false}
                                onChange={(e) => {
                                  const sources = editingConfig.recall?.sources || [];
                                  updateConfig('recall', {
                                    ...editingConfig.recall,
                                    sources: e.target.checked 
                                      ? [...sources.filter(s => s !== 'history'), 'history']
                                      : sources.filter(s => s !== 'history')
                                  });
                                }}
                                className="w-4 h-4"
                              />
                              <span className="text-sm">Conversation History</span>
                            </label>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                  
                  {/* Persistence Section */}
                  <div className="space-y-4">
                    <h4 className="text-md font-semibold border-b border-gray-700 pb-2">Persistence</h4>
                    
                    <div>
                      <label className="flex items-center gap-2">
                        <input
                          type="checkbox"
                          checked={editingConfig.save_to_file || false}
                          onChange={(e) => updateConfig('save_to_file', e.target.checked)}
                          className="w-4 h-4"
                        />
                        <span className="text-sm">Save to file (persistent across sessions)</span>
                      </label>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="max-w-2xl">
                  <h4 className="text-lg font-semibold mb-4">Test Agent</h4>
                  <div className="space-y-4">
                    <div>
                      <label className="block text-sm font-medium mb-2">Test Prompt</label>
                      <textarea
                        value={testPrompt}
                        onChange={(e) => setTestPrompt(e.target.value)}
                        placeholder="Enter a prompt to test this agent..."
                        rows={4}
                        className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500"
                      />
                    </div>
                    
                    <button
                      onClick={handleTest}
                      disabled={testLoading || !testPrompt.trim()}
                      className="flex items-center gap-2 px-4 py-2 bg-primary-600 hover:bg-primary-700 disabled:bg-gray-600 disabled:cursor-not-allowed rounded-lg"
                    >
                      <Play className="w-4 h-4" />
                      {testLoading ? 'Testing...' : 'Test'}
                    </button>
                    
                    {testResult && (
                      <div className="mt-4">
                        <h5 className="text-sm font-medium mb-2">Response:</h5>
                        <div className="p-4 bg-gray-800 border border-gray-700 rounded-lg whitespace-pre-wrap">
                          {testResult.response}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          </>
        ) : (
          <div className="flex-1 flex items-center justify-center text-gray-400">
            <div className="text-center">
              <p className="text-lg mb-2">No agent selected</p>
              <p className="text-sm">Select an agent or create a new one</p>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
