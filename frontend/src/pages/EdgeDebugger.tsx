/**
 * Edge Debugger - Test page to investigate edge rendering issues
 */

import { useEffect, useState } from 'react'
import ReactFlow, {
  Background,
  Controls,
  MiniMap,
  Node,
  Edge,
  useNodesState,
  useEdgesState,
  addEdge,
  Connection,
  useReactFlow,
} from 'reactflow'
import 'reactflow/dist/style.css'
import { nodeTypes } from '../components/nodes'
import { CustomEdge } from '../components/CustomEdge'

const edgeTypes = {
  custom: CustomEdge,
}

export default function EdgeDebugger() {
  const [debugInfo, setDebugInfo] = useState<string[]>([])
  const { fitView } = useReactFlow()

  const addLog = (message: string) => {
    setDebugInfo((prev) => [...prev, `[${new Date().toLocaleTimeString()}] ${message}`])
    console.log('[EdgeDebugger]', message)
  }

  // Define test nodes
  const initialNodes: Node[] = [
    {
      id: 'node-1',
      type: 'prompt',
      position: { x: 100, y: 100 },
      data: {
        label: 'Node 1',
        prompt: 'Test prompt',
        type: 'prompt',
      },
    },
    {
      id: 'node-2',
      type: 'textparse',
      position: { x: 400, y: 100 },
      data: {
        label: 'Node 2',
        type: 'textparse',
        set: { output: '{input}' },
      },
    },
    {
      id: 'node-3',
      type: 'custom',
      position: { x: 700, y: 100 },
      data: {
        label: 'Node 3',
        type: 'custom',
        custom_code: 'return {}',
      },
    },
  ]

  // Define test edges with correct handle IDs
  const initialEdges: Edge[] = [
    {
      id: 'edge-1-2',
      source: 'node-1',
      target: 'node-2',
      sourceHandle: 'response',
      targetHandle: 'input',
      type: 'custom',
      animated: true,
      data: {
        type: 'default',
      },
    },
    {
      id: 'edge-2-3',
      source: 'node-2',
      target: 'node-3',
      sourceHandle: 'output',
      targetHandle: 'input',
      type: 'custom',
      animated: true,
      data: {
        type: 'conditional',
      },
    },
  ]

  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes)
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges)

  useEffect(() => {
    addLog(`Initial render - Nodes: ${initialNodes.length}, Edges: ${initialEdges.length}`)
    addLog(`Node IDs: ${initialNodes.map((n) => n.id).join(', ')}`)
    addLog(`Edges: ${initialEdges.map((e) => `${e.source}->${e.target}`).join(', ')}`)

    // Log edge details
    initialEdges.forEach((edge, i) => {
      addLog(
        `Edge ${i}: ${edge.source}[${edge.sourceHandle}] -> ${edge.target}[${edge.targetHandle}]`
      )
    })

    // Fit view after a short delay
    setTimeout(() => {
      fitView({ padding: 0.2 })
      addLog('Fitted view')
    }, 100)
  }, [])

  useEffect(() => {
    addLog(`Nodes state updated: ${nodes.length} nodes`)
  }, [nodes])

  useEffect(() => {
    addLog(`Edges state updated: ${edges.length} edges`)
  }, [edges])

  const onConnect = (connection: Connection) => {
    addLog(`Connection created: ${connection.source} -> ${connection.target}`)
    setEdges((eds) => addEdge(connection, eds))
  }

  const testAddEdge = () => {
    const newEdge: Edge = {
      id: `edge-test-${Date.now()}`,
      source: 'node-3',
      target: 'node-1',
      sourceHandle: 'output',
      targetHandle: 'input',
      type: 'custom',
      animated: true,
      data: {
        type: 'error',
      },
    }
    addLog(`Adding test edge: ${newEdge.source} -> ${newEdge.target}`)
    setEdges((eds) => [...eds, newEdge])
  }

  const clearAndReload = () => {
    addLog('Clearing all nodes and edges')
    setNodes([])
    setEdges([])

    setTimeout(() => {
      addLog('Reloading nodes')
      setNodes(initialNodes)

      setTimeout(() => {
        addLog('Reloading edges')
        setEdges(initialEdges)
      }, 100)
    }, 100)
  }

  const logCurrentState = () => {
    addLog('=== Current State ===')
    addLog(`Nodes: ${nodes.length}`)
    nodes.forEach((node) => {
      addLog(`  - ${node.id} (${node.type})`)
    })
    addLog(`Edges: ${edges.length}`)
    edges.forEach((edge) => {
      addLog(`  - ${edge.id}: ${edge.source}[${edge.sourceHandle}] -> ${edge.target}[${edge.targetHandle}]`)
    })
  }

  return (
    <div className="flex h-screen bg-gray-900">
      {/* Debug panel */}
      <div className="w-80 bg-gray-800 border-r border-gray-700 flex flex-col">
        <div className="p-4 border-b border-gray-700">
          <h1 className="text-xl font-bold mb-4">Edge Debugger</h1>

          <div className="space-y-2">
            <button
              onClick={testAddEdge}
              className="w-full px-3 py-2 bg-blue-600 hover:bg-blue-700 rounded text-sm"
            >
              Add Test Edge
            </button>
            <button
              onClick={clearAndReload}
              className="w-full px-3 py-2 bg-amber-600 hover:bg-amber-700 rounded text-sm"
            >
              Clear & Reload
            </button>
            <button
              onClick={logCurrentState}
              className="w-full px-3 py-2 bg-green-600 hover:bg-green-700 rounded text-sm"
            >
              Log State
            </button>
          </div>

          <div className="mt-4 text-xs">
            <div className="bg-gray-700 px-2 py-1 rounded">
              Nodes: {nodes.length} | Edges: {edges.length}
            </div>
          </div>
        </div>

        {/* Log output */}
        <div className="flex-1 overflow-y-auto p-4">
          <h2 className="text-sm font-semibold mb-2 text-gray-400">Debug Log</h2>
          <div className="space-y-1 text-xs font-mono">
            {debugInfo.map((log, i) => (
              <div key={i} className="text-gray-300">
                {log}
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* ReactFlow canvas */}
      <div className="flex-1">
        <ReactFlow
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
            data: {
              type: 'default',
            },
          }}
          connectionLineStyle={{ stroke: '#64748b', strokeWidth: 2 }}
          fitView
          className="bg-gray-900"
        >
          <Background color="#374151" gap={16} />
          <Controls />
          <MiniMap />
        </ReactFlow>
      </div>
    </div>
  )
}
