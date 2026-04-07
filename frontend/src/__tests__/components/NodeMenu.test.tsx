import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { NodeMenu } from '../../components/NodeMenu'

const ALL_NODE_TYPES = [
  'chatinput',
  'chatoutput',
  'fileinput',
  'fileoutput',
  'prompt',
  'agentprompt',
  'gethistory',
  'sethistory',
  'textparse',
  'custom',
  'toolcall',
]

const ALL_NODE_LABELS = [
  'Chat Input',
  'Chat Output',
  'File Input',
  'File Output',
  'Prompt Node',
  'Agent Prompt',
  'Get History',
  'Set History',
  'Text Parse',
  'Custom Code',
  'Tool Call',
]

describe('NodeMenu', () => {
  it('renders all 11 node type buttons', () => {
    render(<NodeMenu onAddNode={vi.fn()} />)
    const buttons = screen.getAllByRole('button')
    // 11 node type buttons (the instruction text at the bottom is not a button)
    expect(buttons).toHaveLength(11)
  })

  it.each(ALL_NODE_LABELS)('renders a button labelled "%s"', (label) => {
    render(<NodeMenu onAddNode={vi.fn()} />)
    expect(screen.getByText(label)).toBeInTheDocument()
  })

  it.each(ALL_NODE_TYPES)('calls onAddNode with type "%s" when its button is clicked', (type) => {
    const onAddNode = vi.fn()
    render(<NodeMenu onAddNode={onAddNode} />)

    // Find button by its containing label text
    const label = ALL_NODE_LABELS[ALL_NODE_TYPES.indexOf(type)]
    const button = screen.getByText(label).closest('button')!
    fireEvent.click(button)

    expect(onAddNode).toHaveBeenCalledTimes(1)
    expect(onAddNode.mock.calls[0][0]).toBe(type)
  })

  it('passes the correct defaultData for the chatinput node', () => {
    const onAddNode = vi.fn()
    render(<NodeMenu onAddNode={onAddNode} />)

    fireEvent.click(screen.getByText('Chat Input').closest('button')!)

    const [, data] = onAddNode.mock.calls[0]
    expect(data.type).toBe('chatinput')
    expect(data.save_to).toBe('user_input')
  })

  it('passes the correct defaultData for the toolcall node', () => {
    const onAddNode = vi.fn()
    render(<NodeMenu onAddNode={onAddNode} />)

    fireEvent.click(screen.getByText('Tool Call').closest('button')!)

    const [, data] = onAddNode.mock.calls[0]
    expect(data.type).toBe('toolcall')
    expect(data.command).toBe('echo "Hello"')
  })

  it('passes the correct defaultData for the agentprompt node', () => {
    const onAddNode = vi.fn()
    render(<NodeMenu onAddNode={onAddNode} />)

    fireEvent.click(screen.getByText('Agent Prompt').closest('button')!)

    const [, data] = onAddNode.mock.calls[0]
    expect(data.type).toBe('agentprompt')
    expect(data.agent).toBe('agent_name')
  })

  it('renders the "Node Palette" heading', () => {
    render(<NodeMenu onAddNode={vi.fn()} />)
    expect(screen.getByText('Node Palette')).toBeInTheDocument()
  })

  it('each button click uses a fresh call (not accumulated)', () => {
    const onAddNode = vi.fn()
    render(<NodeMenu onAddNode={onAddNode} />)

    fireEvent.click(screen.getByText('Prompt Node').closest('button')!)
    fireEvent.click(screen.getByText('Custom Code').closest('button')!)

    expect(onAddNode).toHaveBeenCalledTimes(2)
    expect(onAddNode.mock.calls[0][0]).toBe('prompt')
    expect(onAddNode.mock.calls[1][0]).toBe('custom')
  })
})
