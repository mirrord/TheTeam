/**
 * Flowchart store for managing flowcharts and executions.
 */

import { create } from 'zustand'
import { Node, Edge } from 'reactflow'

export interface Flowchart {
  id: string
  name: string
  description?: string
  node_count: number
  source: 'file' | 'runtime'
  path?: string
}

export interface FlowchartConfig {
  id?: string
  name: string
  description?: string
  nodes: Record<string, any>
  edges?: Array<{ source: string; target: string }>
  save_to_file?: boolean
}

export interface ExecutionStatus {
  id: string
  flowchart_id: string
  status: 'starting' | 'running' | 'completed' | 'failed' | 'stopped'
  started_at: string
  completed_at?: string
  current_node?: string
  outputs: Record<string, any>
  error?: string
}

interface FlowchartState {
  flowcharts: Flowchart[]
  currentFlowchart: FlowchartConfig | null
  currentExecution: ExecutionStatus | null
  loading: boolean
  error: string | null

  // ReactFlow state
  nodes: Node[]
  edges: Edge[]

  // Actions
  fetchFlowcharts: () => Promise<void>
  getFlowchart: (id: string) => Promise<void>
  createFlowchart: (config: FlowchartConfig) => Promise<string>
  updateFlowchart: (id: string, config: FlowchartConfig) => Promise<void>
  deleteFlowchart: (id: string) => Promise<void>
  importFromYaml: (yaml: string, name?: string) => Promise<string>
  exportToYaml: (id: string) => Promise<string>
  validateFlowchart: (id: string) => Promise<any>
  executeFlowchart: (id: string, context: any) => Promise<string>
  stopExecution: (executionId: string) => Promise<void>
  subscribeToExecution: (executionId: string) => void
  
  setNodes: (nodes: Node[]) => void
  setEdges: (edges: Edge[]) => void
  setCurrentFlowchart: (flowchart: FlowchartConfig | null) => void
  updateNodeData: (nodeId: string, data: any) => void
  updateEdgeData: (edgeId: string, data: any) => void
}

export const useFlowchartStore = create<FlowchartState>((set, get) => ({
  flowcharts: [],
  currentFlowchart: null,
  currentExecution: null,
  loading: false,
  error: null,
  nodes: [],
  edges: [],

  fetchFlowcharts: async () => {
    set({ loading: true, error: null })
    try {
      const response = await fetch('/api/v1/flowcharts/')
      if (!response.ok) throw new Error('Failed to fetch flowcharts')
      const data = await response.json()
      set({ flowcharts: data.flowcharts, loading: false })
    } catch (error: any) {
      set({ error: error.message, loading: false })
    }
  },

  getFlowchart: async (id: string) => {
    set({ loading: true, error: null })
    try {
      const response = await fetch(`/api/v1/flowcharts/${id}`)
      if (!response.ok) throw new Error('Failed to fetch flowchart')
      const data = await response.json()
      set({ currentFlowchart: data.flowchart.config, loading: false })
      
      // Convert to ReactFlow format
      const config = data.flowchart.config
      
      // Helper function to get the correct handle IDs based on node type
      const getHandleIds = (nodeType: string) => {
        switch (nodeType) {
          case 'prompt':
          case 'agentprompt':
            return { source: 'response', target: 'input' }
          case 'chatinput':
          case 'fileinput':
            // Input nodes only have source handles (they're start nodes)
            return { source: 'output', target: 'input' }
          case 'chatoutput':
          case 'fileoutput':
            // Output nodes only have target handles (they're end nodes)
            return { source: 'output', target: 'input' }
          case 'custom':
          case 'toolcall':
          case 'textparse':
            return { source: 'output', target: 'input' }
          default:
            return { source: 'output', target: 'input' }
        }
      }
      
      // Mark start node
      const startNode = config.start_node || null
      
      const nodes: Node[] = Object.entries(config.nodes || {}).map(([id, node]: [string, any], index) => {
        const isStartNode = id === startNode
        return {
          id,
          type: node.type || 'prompt',
          position: node.position || { x: 100 + (index % 3) * 250, y: 100 + Math.floor(index / 3) * 150 },
          data: {
            label: id,
            prompt: node.prompt,
            custom_code: node.custom_code,
            command: node.command,
            save_to: node.save_to,
            extraction: node.extraction,
            set: node.set,
            transform: node.transform,
            type: node.type || 'prompt',
            isStartNode,
            static_inputs: node.static_inputs,
          },
          style: isStartNode ? {
            border: '3px solid #22c55e',
            boxShadow: '0 0 10px rgba(34, 197, 94, 0.5)',
          } : undefined,
        }
      })
      
      const edges: Edge[] = []
      
      // Map condition types to edge display types
      const getEdgeType = (conditionType?: string): string => {
        if (!conditionType) return 'default'
        
        switch (conditionType) {
          case 'RegexCondition':
          case 'ContextCondition':
          case 'CustomCondition':
            return 'conditional'
          case 'CountCondition':
          case 'LoopCondition':
            return 'loop'
          case 'ErrorCondition':
            return 'error'
          case 'SuccessCondition':
            return 'success'
          case 'AlwaysCondition':
          default:
            return 'default'
        }
      }
      
      // Parse edges from the top-level edges array (YAML format)
      if (config.edges && Array.isArray(config.edges)) {
        config.edges.forEach((edge: any, idx: number) => {
          const sourceNode = config.nodes[edge.from]
          const targetNode = config.nodes[edge.to]
          
          if (sourceNode && targetNode && edge.to !== 'END') {
            const sourceType = sourceNode.type || 'prompt'
            const targetType = targetNode.type || 'prompt'
            const handles = getHandleIds(sourceType)
            const targetHandles = getHandleIds(targetType)
            const edgeType = getEdgeType(edge.condition?.type)
            
            edges.push({
              id: `edge-${edge.from}-${edge.to}-${idx}`,
              source: edge.from,
              target: edge.to,
              sourceHandle: handles.source,
              targetHandle: targetHandles.target,
              type: 'custom',
              animated: true,
              data: {
                type: edgeType,
                condition: edge.condition,
                priority: edge.priority,
                output_key: edge.output_key,
                input_key: edge.input_key,
              },
              label: edge.condition?.type || undefined,
            })
          }
        })
      }
      
      // Fall back to node.next format if no edges array
      if (edges.length === 0) {
        Object.entries(config.nodes || {}).forEach(([source, node]: [string, any]) => {
          if (node.next) {
            const targets = Array.isArray(node.next) ? node.next : [node.next]
            targets.forEach((target: string, idx: number) => {
              if (target !== 'END' && config.nodes[target]) {
                const sourceType = node.type || 'prompt'
                const targetNode = config.nodes[target]
                const targetType = targetNode?.type || 'prompt'
                const handles = getHandleIds(sourceType)
                const targetHandles = getHandleIds(targetType)
                
                edges.push({
                  id: `${source}-${target}-${idx}`,
                  source,
                  target,
                  sourceHandle: handles.source,
                  targetHandle: targetHandles.target,
                  type: 'custom',
                  animated: true,
                  data: {
                    type: 'default',
                  },
                })
              }
            })
          }
        })
      }
      
      // Mark end nodes (nodes with no outgoing edges)
      const nodesWithOutgoing = new Set(edges.map(e => e.source))
      const updatedNodes = nodes.map(node => {
        const isEndNode = !nodesWithOutgoing.has(node.id) && node.id !== startNode
        if (isEndNode) {
          return {
            ...node,
            data: { ...node.data, isEndNode: true },
            style: {
              ...node.style,
              border: node.data.isStartNode ? node.style?.border : '3px solid #ef4444',
              boxShadow: node.data.isStartNode ? node.style?.boxShadow : '0 0 10px rgba(239, 68, 68, 0.5)',
            },
          }
        }
        return node
      })
      
      console.log(`Loaded flowchart ${id}:`, { 
        nodeCount: updatedNodes.length, 
        edgeCount: edges.length,
        nodes: updatedNodes.map(n => n.id),
        edges: edges.map(e => `${e.source}->${e.target}`)
      })
      
      set({ nodes: updatedNodes, edges })
    } catch (error: any) {
      set({ error: error.message, loading: false })
    }
  },

  createFlowchart: async (config: FlowchartConfig) => {
    set({ loading: true, error: null })
    try {
      const response = await fetch('/api/v1/flowcharts/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config),
      })
      if (!response.ok) throw new Error('Failed to create flowchart')
      const data = await response.json()
      await get().fetchFlowcharts()
      set({ loading: false })
      return data.flowchart_id
    } catch (error: any) {
      set({ error: error.message, loading: false })
      throw error
    }
  },

  updateFlowchart: async (id: string, config: FlowchartConfig) => {
    set({ loading: true, error: null })
    try {
      const response = await fetch(`/api/v1/flowcharts/${id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config),
      })
      if (!response.ok) throw new Error('Failed to update flowchart')
      await get().fetchFlowcharts()
      set({ loading: false })
    } catch (error: any) {
      set({ error: error.message, loading: false })
      throw error
    }
  },

  deleteFlowchart: async (id: string) => {
    set({ loading: true, error: null })
    try {
      const response = await fetch(`/api/v1/flowcharts/${id}`, {
        method: 'DELETE',
      })
      if (!response.ok) throw new Error('Failed to delete flowchart')
      await get().fetchFlowcharts()
      set({ loading: false })
    } catch (error: any) {
      set({ error: error.message, loading: false })
      throw error
    }
  },

  importFromYaml: async (yaml: string, name?: string) => {
    set({ loading: true, error: null })
    try {
      const response = await fetch('/api/v1/flowcharts/import', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ yaml, name }),
      })
      if (!response.ok) throw new Error('Failed to import flowchart')
      const data = await response.json()
      await get().fetchFlowcharts()
      set({ loading: false })
      return data.flowchart_id
    } catch (error: any) {
      set({ error: error.message, loading: false })
      throw error
    }
  },

  exportToYaml: async (id: string) => {
    set({ loading: true, error: null })
    try {
      const response = await fetch(`/api/v1/flowcharts/${id}/export`)
      if (!response.ok) throw new Error('Failed to export flowchart')
      const data = await response.json()
      set({ loading: false })
      return data.yaml
    } catch (error: any) {
      set({ error: error.message, loading: false })
      throw error
    }
  },

  validateFlowchart: async (id: string) => {
    try {
      const response = await fetch(`/api/v1/flowcharts/${id}/validate`, {
        method: 'POST',
      })
      if (!response.ok) throw new Error('Failed to validate flowchart')
      const data = await response.json()
      return data
    } catch (error: any) {
      set({ error: error.message })
      throw error
    }
  },

  executeFlowchart: async (id: string, context: any) => {
    set({ loading: true, error: null })
    try {
      const response = await fetch(`/api/v1/flowcharts/${id}/execute`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ context }),
      })
      if (!response.ok) throw new Error('Failed to execute flowchart')
      const data = await response.json()
      set({ loading: false })
      return data.execution_id
    } catch (error: any) {
      set({ error: error.message, loading: false })
      throw error
    }
  },

  stopExecution: async (_executionId: string) => {
    try {
      // This will be handled via WebSocket
      // Future: implement stop execution via socket
    } catch (error: any) {
      set({ error: error.message })
      throw error
    }
  },

  subscribeToExecution: (_executionId: string) => {
    // This will be handled via WebSocket in the component
  },

  setNodes: (nodes: Node[]) => set({ nodes }),
  setEdges: (edges: Edge[]) => set({ edges }),
  setCurrentFlowchart: (flowchart: FlowchartConfig | null) => set({ currentFlowchart: flowchart }),
  
  updateNodeData: (nodeId: string, data: any) => {
    set((state) => ({
      nodes: state.nodes.map((node) =>
        node.id === nodeId ? { ...node, data: { ...node.data, ...data } } : node
      ),
    }))
  },
  
  updateEdgeData: (edgeId: string, data: any) => {
    set((state) => ({
      edges: state.edges.map((edge) =>
        edge.id === edgeId ? { ...edge, data: { ...edge.data, ...data } } : edge
      ),
    }))
  },
}))
