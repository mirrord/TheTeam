import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { ToolConfirmationModal } from '../../components/ToolConfirmationModal'
import { useConfirmStore } from '../../store/confirmStore'
import { useSocketStore } from '../../store/socketStore'

// Prevent real Socket.IO connection during tests
vi.mock('socket.io-client', () => ({
  io: vi.fn(() => ({
    on: vi.fn(),
    off: vi.fn(),
    emit: vi.fn(),
    connect: vi.fn(),
    disconnect: vi.fn(),
    connected: false,
  })),
}))

const socketInitialState = {
  socket: null,
  connected: false,
  connecting: false,
  error: null,
  reconnectAttempts: 0,
  clientId: null,
}

const sampleConfirmation = {
  requestId: 'req-42',
  command: 'echo "hello world"',
  conversationId: 'conv-1',
  messageId: 'msg-1',
}

beforeEach(() => {
  useConfirmStore.setState({ pendingConfirmation: null })
  useSocketStore.setState(socketInitialState)
})

describe('ToolConfirmationModal — no pending confirmation', () => {
  it('renders nothing when pendingConfirmation is null', () => {
    const { container } = render(<ToolConfirmationModal />)
    expect(container.firstChild).toBeNull()
  })
})

describe('ToolConfirmationModal — with pending confirmation', () => {
  beforeEach(() => {
    useConfirmStore.setState({ pendingConfirmation: sampleConfirmation })
  })

  it('renders the command text', () => {
    render(<ToolConfirmationModal />)
    expect(screen.getByText('echo "hello world"')).toBeInTheDocument()
  })

  it('renders Allow and Deny buttons', () => {
    render(<ToolConfirmationModal />)
    expect(screen.getByRole('button', { name: /allow/i })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /deny/i })).toBeInTheDocument()
  })

  it('shows the approval prompt heading', () => {
    render(<ToolConfirmationModal />)
    expect(screen.getByText(/Tool Call Requires Approval/i)).toBeInTheDocument()
  })
})

describe('ToolConfirmationModal — Allow button', () => {
  it('emits tool_confirmation_response with approved:true and clears confirmation', () => {
    const mockEmit = vi.fn()
    useSocketStore.setState({ emit: mockEmit } as any)
    useConfirmStore.setState({ pendingConfirmation: sampleConfirmation })

    render(<ToolConfirmationModal />)
    fireEvent.click(screen.getByRole('button', { name: /allow/i }))

    expect(mockEmit).toHaveBeenCalledWith('tool_confirmation_response', {
      request_id: 'req-42',
      approved: true,
    })
    expect(useConfirmStore.getState().pendingConfirmation).toBeNull()
  })
})

describe('ToolConfirmationModal — Deny button', () => {
  it('emits tool_confirmation_response with approved:false and clears confirmation', () => {
    const mockEmit = vi.fn()
    useSocketStore.setState({ emit: mockEmit } as any)
    useConfirmStore.setState({ pendingConfirmation: sampleConfirmation })

    render(<ToolConfirmationModal />)
    fireEvent.click(screen.getByRole('button', { name: /deny/i }))

    expect(mockEmit).toHaveBeenCalledWith('tool_confirmation_response', {
      request_id: 'req-42',
      approved: false,
    })
    expect(useConfirmStore.getState().pendingConfirmation).toBeNull()
  })
})
