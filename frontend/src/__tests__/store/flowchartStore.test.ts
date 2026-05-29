import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { Node, Edge } from 'reactflow'
import { useFlowchartStore } from '../../store/flowchartStore'

const initialState = {
  flowcharts: [],
  currentFlowchart: null,
  currentExecution: null,
  loading: false,
  error: null,
  nodes: [],
  edges: [],
}

beforeEach(() => {
  useFlowchartStore.setState(initialState)
  vi.restoreAllMocks()
})

afterEach(() => {
  vi.restoreAllMocks()
})

// ─── Pure state mutations ────────────────────────────────────────────────────

describe('setNodes', () => {
  it('replaces the nodes array', () => {
    const nodes: Node[] = [{ id: 'n1', type: 'prompt', position: { x: 0, y: 0 }, data: {} }]
    useFlowchartStore.getState().setNodes(nodes)
    expect(useFlowchartStore.getState().nodes).toEqual(nodes)
  })

  it('clears nodes when passed an empty array', () => {
    useFlowchartStore.setState({ nodes: [{ id: 'n1', position: { x: 0, y: 0 }, data: {} }] })
    useFlowchartStore.getState().setNodes([])
    expect(useFlowchartStore.getState().nodes).toHaveLength(0)
  })
})

describe('setEdges', () => {
  it('replaces the edges array', () => {
    const edges: Edge[] = [{ id: 'e1', source: 'n1', target: 'n2' }]
    useFlowchartStore.getState().setEdges(edges)
    expect(useFlowchartStore.getState().edges).toEqual(edges)
  })
})

describe('setCurrentFlowchart', () => {
  it('sets the current flowchart', () => {
    const fc = { name: 'test', nodes: {}, id: 'fc-1' }
    useFlowchartStore.getState().setCurrentFlowchart(fc)
    expect(useFlowchartStore.getState().currentFlowchart).toEqual(fc)
  })

  it('clears when passed null', () => {
    useFlowchartStore.setState({ currentFlowchart: { name: 'x', nodes: {} } })
    useFlowchartStore.getState().setCurrentFlowchart(null)
    expect(useFlowchartStore.getState().currentFlowchart).toBeNull()
  })
})

describe('updateNodeData', () => {
  it('merges new data into the target node', () => {
    useFlowchartStore.setState({
      nodes: [
        { id: 'n1', position: { x: 0, y: 0 }, data: { label: 'original', prompt: 'old' } },
        { id: 'n2', position: { x: 100, y: 0 }, data: { label: 'other' } },
      ],
    })

    useFlowchartStore.getState().updateNodeData('n1', { prompt: 'updated' })

    const nodes = useFlowchartStore.getState().nodes
    expect(nodes.find(n => n.id === 'n1')?.data.prompt).toBe('updated')
    // Existing keys on n1 are preserved
    expect(nodes.find(n => n.id === 'n1')?.data.label).toBe('original')
    // Other nodes are untouched
    expect(nodes.find(n => n.id === 'n2')?.data.label).toBe('other')
  })

  it('is a no-op when the node id does not exist', () => {
    useFlowchartStore.setState({
      nodes: [{ id: 'n1', position: { x: 0, y: 0 }, data: { label: 'x' } }],
    })
    useFlowchartStore.getState().updateNodeData('nonexistent', { label: 'y' })
    expect(useFlowchartStore.getState().nodes[0].data.label).toBe('x')
  })
})

describe('updateEdgeData', () => {
  it('merges new data into the target edge', () => {
    useFlowchartStore.setState({
      edges: [
        { id: 'e1', source: 'a', target: 'b', data: { type: 'default', priority: 9 } },
        { id: 'e2', source: 'b', target: 'c', data: { type: 'loop' } },
      ],
    })

    useFlowchartStore.getState().updateEdgeData('e1', { priority: 1 })

    const edges = useFlowchartStore.getState().edges
    expect(edges.find(e => e.id === 'e1')?.data.priority).toBe(1)
    // Other data preserved
    expect(edges.find(e => e.id === 'e1')?.data.type).toBe('default')
    // Other edges untouched
    expect(edges.find(e => e.id === 'e2')?.data.type).toBe('loop')
  })
})

// ─── getFlowchart — YAML-to-ReactFlow conversion ────────────────────────────
//
// This is the most complex function; the tests verify the conversion logic
// independently of the REST API by mocking fetch.

function mockGetFlowchartFetch(config: Record<string, any>) {
  vi.spyOn(globalThis, 'fetch').mockResolvedValue(
    new Response(
      JSON.stringify({ flowchart: { config } }),
      { status: 200 }
    )
  )
}

describe('getFlowchart — node conversion', () => {
  it('creates one ReactFlow node per config node with the correct type', async () => {
    mockGetFlowchartFetch({
      start_node: 'node_a',
      nodes: {
        node_a: { type: 'prompt', prompt: 'hello' },
        node_b: { type: 'toolcall', command: 'ls' },
      },
      edges: [],
    })

    await useFlowchartStore.getState().getFlowchart('fc-1')

    const { nodes } = useFlowchartStore.getState()
    expect(nodes).toHaveLength(2)
    expect(nodes.find(n => n.id === 'node_a')?.type).toBe('prompt')
    expect(nodes.find(n => n.id === 'node_b')?.type).toBe('toolcall')
  })

  it('marks the start_node with isStartNode=true', async () => {
    mockGetFlowchartFetch({
      start_node: 'start',
      nodes: {
        start: { type: 'prompt', prompt: 'begin' },
        end: { type: 'chatoutput' },
      },
      edges: [{ from: 'start', to: 'end' }],
    })

    await useFlowchartStore.getState().getFlowchart('fc-1')

    const { nodes } = useFlowchartStore.getState()
    expect(nodes.find(n => n.id === 'start')?.data.isStartNode).toBe(true)
    expect(nodes.find(n => n.id === 'end')?.data.isStartNode).toBeFalsy()
  })

  it('marks nodes with no outgoing edges as isEndNode (when not start)', async () => {
    mockGetFlowchartFetch({
      start_node: 'a',
      nodes: {
        a: { type: 'prompt' },
        b: { type: 'chatoutput' },
      },
      edges: [{ from: 'a', to: 'b' }],
    })

    await useFlowchartStore.getState().getFlowchart('fc-1')

    const { nodes } = useFlowchartStore.getState()
    expect(nodes.find(n => n.id === 'b')?.data.isEndNode).toBe(true)
    expect(nodes.find(n => n.id === 'a')?.data.isEndNode).toBeFalsy()
  })

  it('uses provided node position when available', async () => {
    mockGetFlowchartFetch({
      start_node: 'only',
      nodes: {
        only: { type: 'prompt', position: { x: 42, y: 99 } },
      },
      edges: [],
    })

    await useFlowchartStore.getState().getFlowchart('fc-1')
    const node = useFlowchartStore.getState().nodes[0]
    expect(node.position).toEqual({ x: 42, y: 99 })
  })

  it('assigns a default position when none is provided', async () => {
    mockGetFlowchartFetch({
      start_node: 'only',
      nodes: { only: { type: 'prompt' } },
      edges: [],
    })

    await useFlowchartStore.getState().getFlowchart('fc-1')
    const node = useFlowchartStore.getState().nodes[0]
    expect(typeof node.position.x).toBe('number')
    expect(typeof node.position.y).toBe('number')
  })
})

describe('getFlowchart — edge conversion (top-level edges array)', () => {
  it('creates one ReactFlow edge per YAML edge (excluding END targets)', async () => {
    mockGetFlowchartFetch({
      start_node: 'a',
      nodes: {
        a: { type: 'prompt' },
        b: { type: 'chatoutput' },
      },
      edges: [
        { from: 'a', to: 'b' },
        { from: 'b', to: 'END' },  // should be excluded
      ],
    })

    await useFlowchartStore.getState().getFlowchart('fc-1')
    expect(useFlowchartStore.getState().edges).toHaveLength(1)
    expect(useFlowchartStore.getState().edges[0].source).toBe('a')
    expect(useFlowchartStore.getState().edges[0].target).toBe('b')
  })

  it('maps CountCondition to edge type "loop"', async () => {
    mockGetFlowchartFetch({
      start_node: 'a',
      nodes: { a: { type: 'prompt' }, b: { type: 'prompt' } },
      edges: [{ from: 'a', to: 'b', condition: { type: 'CountCondition', limit: 3 } }],
    })

    await useFlowchartStore.getState().getFlowchart('fc-1')
    expect(useFlowchartStore.getState().edges[0].data.type).toBe('loop')
  })

  it('maps RegexCondition to edge type "conditional"', async () => {
    mockGetFlowchartFetch({
      start_node: 'a',
      nodes: { a: { type: 'prompt' }, b: { type: 'prompt' } },
      edges: [{ from: 'a', to: 'b', condition: { type: 'RegexCondition' } }],
    })

    await useFlowchartStore.getState().getFlowchart('fc-1')
    expect(useFlowchartStore.getState().edges[0].data.type).toBe('conditional')
  })

  it('maps ErrorCondition to edge type "error"', async () => {
    mockGetFlowchartFetch({
      start_node: 'a',
      nodes: { a: { type: 'prompt' }, b: { type: 'prompt' } },
      edges: [{ from: 'a', to: 'b', condition: { type: 'ErrorCondition' } }],
    })

    await useFlowchartStore.getState().getFlowchart('fc-1')
    expect(useFlowchartStore.getState().edges[0].data.type).toBe('error')
  })

  it('maps SuccessCondition to edge type "success"', async () => {
    mockGetFlowchartFetch({
      start_node: 'a',
      nodes: { a: { type: 'prompt' }, b: { type: 'prompt' } },
      edges: [{ from: 'a', to: 'b', condition: { type: 'SuccessCondition' } }],
    })

    await useFlowchartStore.getState().getFlowchart('fc-1')
    expect(useFlowchartStore.getState().edges[0].data.type).toBe('success')
  })

  it('uses "default" edge type when there is no condition', async () => {
    mockGetFlowchartFetch({
      start_node: 'a',
      nodes: { a: { type: 'prompt' }, b: { type: 'prompt' } },
      edges: [{ from: 'a', to: 'b' }],
    })

    await useFlowchartStore.getState().getFlowchart('fc-1')
    expect(useFlowchartStore.getState().edges[0].data.type).toBe('default')
  })

  it('stores condition, priority, output_key, and input_key in edge data', async () => {
    const condition = { type: 'ContextCondition' }
    mockGetFlowchartFetch({
      start_node: 'a',
      nodes: { a: { type: 'prompt' }, b: { type: 'prompt' } },
      edges: [{ from: 'a', to: 'b', condition, priority: 2, output_key: 'out', input_key: 'in' }],
    })

    await useFlowchartStore.getState().getFlowchart('fc-1')
    const edgeData = useFlowchartStore.getState().edges[0].data
    expect(edgeData.condition).toEqual(condition)
    expect(edgeData.priority).toBe(2)
    expect(edgeData.output_key).toBe('out')
    expect(edgeData.input_key).toBe('in')
  })
})

describe('getFlowchart — node.next fallback (legacy format)', () => {
  it('creates edges from node.next when no top-level edges array exists', async () => {
    mockGetFlowchartFetch({
      start_node: 'a',
      nodes: {
        a: { type: 'prompt', next: ['b'] },
        b: { type: 'chatoutput' },
      },
    })

    await useFlowchartStore.getState().getFlowchart('fc-1')
    expect(useFlowchartStore.getState().edges).toHaveLength(1)
    expect(useFlowchartStore.getState().edges[0].source).toBe('a')
    expect(useFlowchartStore.getState().edges[0].target).toBe('b')
  })

  it('skips "END" targets in the legacy format', async () => {
    mockGetFlowchartFetch({
      start_node: 'a',
      nodes: {
        a: { type: 'prompt', next: ['END'] },
      },
    })

    await useFlowchartStore.getState().getFlowchart('fc-1')
    expect(useFlowchartStore.getState().edges).toHaveLength(0)
  })
})

describe('getFlowchart — handle ID selection by node type', () => {
  async function edgeFor(sourceType: string, targetType: string) {
    mockGetFlowchartFetch({
      start_node: 'src',
      nodes: { src: { type: sourceType }, tgt: { type: targetType } },
      edges: [{ from: 'src', to: 'tgt' }],
    })
    await useFlowchartStore.getState().getFlowchart('fc-1')
    return useFlowchartStore.getState().edges[0]
  }

  it('prompt source node uses sourceHandle "response"', async () => {
    const edge = await edgeFor('prompt', 'chatoutput')
    expect(edge.sourceHandle).toBe('response')
  })

  it('agentprompt source node uses sourceHandle "response"', async () => {
    const edge = await edgeFor('agentprompt', 'chatoutput')
    expect(edge.sourceHandle).toBe('response')
  })

  it('chatinput source node uses sourceHandle "output"', async () => {
    const edge = await edgeFor('chatinput', 'prompt')
    expect(edge.sourceHandle).toBe('output')
  })

  it('toolcall source node uses sourceHandle "output"', async () => {
    const edge = await edgeFor('toolcall', 'prompt')
    expect(edge.sourceHandle).toBe('output')
  })
})

describe('getFlowchart — fetch errors', () => {
  it('sets error and loading=false when fetch fails', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response('', { status: 404 })
    )

    await useFlowchartStore.getState().getFlowchart('missing')
    expect(useFlowchartStore.getState().error).toBeTruthy()
  })

  it('sets error and loading=false on network error', async () => {
    vi.spyOn(globalThis, 'fetch').mockRejectedValue(new Error('Network down'))

    await useFlowchartStore.getState().getFlowchart('missing')
    expect(useFlowchartStore.getState().error).toBeTruthy()
    expect(useFlowchartStore.getState().loading).toBe(false)
  })
})
