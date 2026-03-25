import { useEffect, useState, useCallback } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import ReactFlow, {
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
  Connection,
  useReactFlow,
  Edge,
} from 'reactflow'
import 'reactflow/dist/style.css'
import { useFlowchartStore } from '../store/flowchartStore'
import { useSocketStore } from '../store/socketStore'
import { Play, Save, Download, Upload, Trash2, Plus, CheckCircle, XCircle } from 'lucide-react'
import { nodeTypes } from '../components/nodes'
import { NodeMenu } from '../components/NodeMenu'
import { CustomEdge } from '../components/CustomEdge'

const edgeTypes = {
  custom: CustomEdge,
}

export default function FlowchartEditor() {
  const { id } = useParams()
  const navigate = useNavigate()
  const {
    flowcharts,
    currentFlowchart,
    fetchFlowcharts,
    getFlowchart,
    updateFlowchart,
    deleteFlowchart,
    importFromYaml,
    exportToYaml,
    executeFlowchart,
    nodes: storeNodes,
    edges: storeEdges,
    updateEdgeData,
    loading,
  } = useFlowchartStore()
  
  const { emit, on, off } = useSocketStore()
  const { fitView } = useReactFlow()
  
  const [nodes, setNodes, onNodesChange] = useNodesState(storeNodes)
  const [edges, setEdges, onEdgesChange] = useEdgesState(storeEdges)
  const [selectedFlowchart, setSelectedFlowchart] = useState<string | null>(id || null)
  const [executionId, setExecutionId] = useState<string | null>(null)
  const [executionStatus, setExecutionStatus] = useState<any>(null)
  
  useEffect(() => {
    fetchFlowcharts()
  }, [])
  
  useEffect(() => {
    if (id && id !== 'new') {
      setSelectedFlowchart(id)
      // Clear existing data before loading new flowchart
      setNodes([])
      setEdges([])
      getFlowchart(id)
    } else if (id === 'new') {
      // Initialize a new flowchart
      setSelectedFlowchart(null)
      setNodes([])
      setEdges([])
    }
  }, [id])
  
  // Sync nodes first, then edges (nodes must exist before edges can connect them)
  useEffect(() => {
    setNodes(storeNodes)
  }, [storeNodes, setNodes])
  
  const handleEdgeTypeChange = useCallback((edgeId: string, newType: string) => {
    setEdges((eds) =>
      eds.map((edge) => {
        if (edge.id === edgeId) {
          return {
            ...edge,
            data: {
              ...edge.data,
              type: newType,
            },
          }
        }
        return edge
      })
    )
  }, [setEdges])
  
  const handleEdgeDataChange = useCallback((edgeId: string, newData: any) => {
    updateEdgeData(edgeId, newData)
  }, [updateEdgeData])
  
  useEffect(() => {
    // Only set edges after nodes are ready
    if (storeEdges.length > 0 || nodes.length > 0) {
      // Ensure all edges have the custom type and callback
      const customEdges = storeEdges.map(edge => ({
        ...edge,
        type: edge.type || 'custom',
        data: {
          ...edge.data,
          onEdgeTypeChange: handleEdgeTypeChange,
          onEdgeDataChange: handleEdgeDataChange,
        },
      }))
      setEdges(customEdges)
    }
  }, [storeEdges, nodes.length, setEdges, handleEdgeTypeChange, handleEdgeDataChange])
  
  // Fit view after edges are loaded
  useEffect(() => {
    if (edges.length > 0 && nodes.length > 0) {
      // Delay fitView slightly to ensure edges are rendered
      setTimeout(() => {
        fitView({ padding: 0.2, duration: 200 })
      }, 100)
    }
  }, [edges.length, nodes.length, fitView])
  
  // WebSocket handlers for execution updates
  useEffect(() => {
    const handleExecutionUpdate = (data: any) => {
      if (data.execution_id === executionId) {
        setExecutionStatus(data)
      }
    }
    
    const handleNodeExecution = (data: any) => {
      if (data.execution_id === executionId) {
        // Highlight current node
        setNodes((nds) =>
          nds.map((node) => ({
            ...node,
            style: {
              ...node.style,
              border: node.id === data.node_id ? '2px solid #0ea5e9' : undefined,
            },
          }))
        )
      }
    }
    
    const handleExecutionComplete = (data: any) => {
      if (data.execution_id === executionId) {
        setExecutionStatus({ ...data, status: 'completed' })
        // Clear node highlights
        setNodes((nds) =>
          nds.map((node) => ({
            ...node,
            style: { ...node.style, border: undefined },
          }))
        )
      }
    }
    
    on('execution_update', handleExecutionUpdate)
    on('node_execution', handleNodeExecution)
    on('execution_complete', handleExecutionComplete)
    
    return () => {
      off('execution_update', handleExecutionUpdate)
      off('node_execution', handleNodeExecution)
      off('execution_complete', handleExecutionComplete)
    }
  }, [executionId, on, off])
  
  const onConnect = useCallback(
    (connection: Connection) => {
      if (!connection.source || !connection.target) return
      
      const newEdge: Edge = {
        id: `${connection.source}-${connection.target}-${Date.now()}`,
        source: connection.source,
        target: connection.target,
        sourceHandle: connection.sourceHandle,
        targetHandle: connection.targetHandle,
        type: 'custom',
        data: {
          type: 'default',
          onEdgeTypeChange: handleEdgeTypeChange,
        },
      }
      setEdges((eds) => [...eds, newEdge])
    },
    [setEdges, handleEdgeTypeChange]
  )
  
  const handleAddNode = useCallback((type: string, data: any) => {
    const newNode = {
      id: `${type}-${Date.now()}`,
      type,
      position: { x: Math.random() * 400 + 100, y: Math.random() * 300 + 100 },
      data,
    }
    setNodes((nds) => [...nds, newNode])
  }, [setNodes])
  
  const handleSave = async () => {
    if (!selectedFlowchart || !currentFlowchart) return
    
    // Convert ReactFlow nodes/edges back to flowchart config
    const nodesConfig: Record<string, any> = {}
    nodes.forEach((node) => {
      nodesConfig[node.id] = {
        type: node.type,
        position: node.position,
        ...node.data,
      }
    })
    
    // Add edges as 'next' connections
    edges.forEach((edge) => {
      if (!nodesConfig[edge.source].next) {
        nodesConfig[edge.source].next = []
      }
      nodesConfig[edge.source].next.push(edge.target)
    })
    
    const config = {
      ...currentFlowchart,
      nodes: nodesConfig,
    }
    
    await updateFlowchart(selectedFlowchart, config)
  }
  
  const handleExport = async () => {
    if (!selectedFlowchart) return
    const yamlData = await exportToYaml(selectedFlowchart)
    
    // Download file
    const blob = new Blob([yamlData], { type: 'text/yaml' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${selectedFlowchart}.yaml`
    a.click()
    URL.revokeObjectURL(url)
  }
  
  const handleImport = () => {
    const input = document.createElement('input')
    input.type = 'file'
    input.accept = '.yaml,.yml'
    input.onchange = async (e: any) => {
      const file = e.target.files[0]
      if (file) {
        const text = await file.text()
        const flowchartId = await importFromYaml(text, file.name.replace(/\.ya?ml$/, ''))
        navigate(`/flowcharts/${flowchartId}`)
      }
    }
    input.click()
  }
  
  const handleExecute = async () => {
    if (!selectedFlowchart) return
    const execId = await executeFlowchart(selectedFlowchart, {})
    setExecutionId(execId)
    emit('subscribe_execution', { execution_id: execId })
  }
  
  const handleDelete = async () => {
    if (!selectedFlowchart) return
    if (confirm('Are you sure you want to delete this flowchart?')) {
      await deleteFlowchart(selectedFlowchart)
      navigate('/flowcharts')
    }
  }
  
  return (
    <div className="flex h-full">
      {/* Flowchart list sidebar */}
      <div className="w-64 bg-gray-800 border-r border-gray-700 flex flex-col">
        <div className="p-4 border-b border-gray-700">
          <h2 className="text-lg font-semibold">Flowcharts</h2>
          <button
            onClick={() => navigate('/flowcharts/new')}
            className="mt-2 w-full flex items-center justify-center gap-2 px-3 py-2 bg-primary-600 hover:bg-primary-700 rounded-lg text-sm"
          >
            <Plus className="w-4 h-4" />
            New Flowchart
          </button>
        </div>
        
        <div className="flex-1 overflow-y-auto p-2">
          {flowcharts.map((fc) => (
            <button
              key={fc.id}
              onClick={() => navigate(`/flowcharts/${fc.id}`)}
              className={`w-full text-left px-3 py-2 rounded-lg mb-1 transition-colors ${
                selectedFlowchart === fc.id
                  ? 'bg-primary-600 text-white'
                  : 'text-gray-300 hover:bg-gray-700'
              }`}
            >
              <div className="font-medium">{fc.name}</div>
              <div className="text-xs text-gray-400">{fc.node_count} nodes</div>
            </button>
          ))}
        </div>
      </div>
      
      {/* Main editor */}
      <div className="flex-1 flex flex-col">
        {/* Toolbar */}
        <div className="bg-gray-800 border-b border-gray-700 p-3 flex items-center gap-2">
          <button
            onClick={handleSave}
            disabled={!selectedFlowchart || loading}
            className="flex items-center gap-2 px-3 py-2 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 disabled:cursor-not-allowed rounded-lg text-sm"
          >
            <Save className="w-4 h-4" />
            Save
          </button>
          
          <button
            onClick={handleExecute}
            disabled={!selectedFlowchart || loading || executionStatus?.status === 'running'}
            className="flex items-center gap-2 px-3 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed rounded-lg text-sm"
          >
            <Play className="w-4 h-4" />
            Execute
          </button>
          
          <div className="flex-1" />
          
          <button
            onClick={handleImport}
            className="flex items-center gap-2 px-3 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg text-sm"
          >
            <Upload className="w-4 h-4" />
            Import
          </button>
          
          <button
            onClick={handleExport}
            disabled={!selectedFlowchart}
            className="flex items-center gap-2 px-3 py-2 bg-gray-700 hover:bg-gray-600 disabled:bg-gray-800 disabled:cursor-not-allowed rounded-lg text-sm"
          >
            <Download className="w-4 h-4" />
            Export
          </button>
          
          <button
            onClick={handleDelete}
            disabled={!selectedFlowchart}
            className="flex items-center gap-2 px-3 py-2 bg-red-600 hover:bg-red-700 disabled:bg-gray-600 disabled:cursor-not-allowed rounded-lg text-sm"
          >
            <Trash2 className="w-4 h-4" />
            Delete
          </button>
          
          {executionStatus && (
            <div className="flex items-center gap-2 px-3 py-2 bg-gray-700 rounded-lg text-sm">
              {executionStatus.status === 'completed' ? (
                <CheckCircle className="w-4 h-4 text-green-400" />
              ) : executionStatus.status === 'failed' ? (
                <XCircle className="w-4 h-4 text-red-400" />
              ) : (
                <div className="w-4 h-4 border-2 border-blue-400 border-t-transparent rounded-full animate-spin" />
              )}
              <span className="capitalize">{executionStatus.status}</span>
            </div>
          )}
        </div>
        
        {/* ReactFlow canvas with Node Menu */}
        <div className="flex-1 flex bg-gray-900">
          {/* Node Palette */}
          <NodeMenu onAddNode={handleAddNode} />
          
          {/* Canvas */}
          <div className="flex-1">
            {selectedFlowchart || id === 'new' ? (
              <ReactFlow
                key={selectedFlowchart || 'new-flowchart'}
                nodes={nodes}
                edges={edges}
                onNodesChange={onNodesChange}
                onEdgesChange={onEdgesChange}
                onConnect={onConnect}
                nodeTypes={nodeTypes}
                edgeTypes={edgeTypes}
                defaultEdgeOptions={{
                  type: 'custom',
                  animated: true,
                }}
                connectionLineStyle={{ stroke: '#64748b', strokeWidth: 2 }}
                className="bg-gray-900"
              >
                <Background color="#374151" gap={16} />
                <Controls />
                <MiniMap />
              </ReactFlow>
            ) : (
              <div className="flex items-center justify-center h-full text-gray-400">
                <div className="text-center">
                  <p className="text-lg mb-2">No flowchart selected</p>
                  <p className="text-sm">Create a new flowchart or select one from the list</p>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
