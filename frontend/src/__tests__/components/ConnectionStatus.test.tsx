import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import ConnectionStatus from '../../components/ConnectionStatus'
import { useSocketStore } from '../../store/socketStore'

// Prevent the real Socket.IO connection from initialising during tests
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

beforeEach(() => {
  useSocketStore.setState(socketInitialState)
})

describe('ConnectionStatus — connected', () => {
  it('shows "Connected" text', () => {
    useSocketStore.setState({ connected: true })
    render(<ConnectionStatus />)
    expect(screen.getByText('Connected')).toBeInTheDocument()
  })

  it('does not show any error or reconnect text', () => {
    useSocketStore.setState({ connected: true })
    render(<ConnectionStatus />)
    expect(screen.queryByText(/reconnect/i)).toBeNull()
    expect(screen.queryByText(/Disconnected/i)).toBeNull()
  })
})

describe('ConnectionStatus — connecting (first attempt)', () => {
  it('shows "Connecting…" when reconnectAttempts is 0', () => {
    useSocketStore.setState({ connecting: true, reconnectAttempts: 0 })
    render(<ConnectionStatus />)
    expect(screen.getByText(/Connecting\.\.\./)).toBeInTheDocument()
  })
})

describe('ConnectionStatus — connecting (retry)', () => {
  it('shows attempt count when reconnectAttempts > 0', () => {
    useSocketStore.setState({ connecting: true, reconnectAttempts: 3 })
    render(<ConnectionStatus />)
    expect(screen.getByText(/Reconnecting \(3\)\.\.\./)).toBeInTheDocument()
  })
})

describe('ConnectionStatus — error', () => {
  it('shows the error message', () => {
    useSocketStore.setState({ error: 'Connection refused' })
    render(<ConnectionStatus />)
    expect(screen.getByText('Connection refused')).toBeInTheDocument()
  })

  it('shows a "Dismiss" button', () => {
    useSocketStore.setState({ error: 'Some error' })
    render(<ConnectionStatus />)
    expect(screen.getByText('Dismiss')).toBeInTheDocument()
  })

  it('calls resetError when Dismiss is clicked', () => {
    const resetError = vi.fn()
    useSocketStore.setState({ error: 'err', resetError })
    render(<ConnectionStatus />)
    fireEvent.click(screen.getByText('Dismiss'))
    expect(resetError).toHaveBeenCalledTimes(1)
  })
})

describe('ConnectionStatus — disconnected (no error)', () => {
  it('shows "Disconnected" text', () => {
    // Default state: connected=false, connecting=false, error=null
    render(<ConnectionStatus />)
    expect(screen.getByText('Disconnected')).toBeInTheDocument()
  })
})
