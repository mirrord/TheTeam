/**
 * WebSocket store with robust reconnection handling.
 * Manages Socket.IO connection with automatic reconnection,
 * exponential backoff, and seamless state recovery.
 */

import { create } from 'zustand'
import { io, Socket } from 'socket.io-client'

interface SocketState {
  socket: Socket | null
  connected: boolean
  connecting: boolean
  error: string | null
  reconnectAttempts: number
  clientId: string | null
  
  // Actions
  connect: () => void
  disconnect: () => void
  emit: (event: string, data: any) => void
  on: (event: string, handler: (...args: any[]) => void) => void
  off: (event: string, handler: (...args: any[]) => void) => void
  resetError: () => void
}

const MAX_RECONNECT_ATTEMPTS = 10
const INITIAL_RECONNECT_DELAY = 1000
const MAX_RECONNECT_DELAY = 30000

export const useSocketStore = create<SocketState>((set, get) => {
  let reconnectTimer: NodeJS.Timeout | null = null
  let reconnectDelay = INITIAL_RECONNECT_DELAY

  const setupSocketHandlers = (socket: Socket) => {
    socket.on('connect', () => {
      console.log('✅ Socket connected:', socket.id)
      reconnectDelay = INITIAL_RECONNECT_DELAY // Reset backoff
      
      set({
        connected: true,
        connecting: false,
        error: null,
        reconnectAttempts: 0,
        clientId: socket.id,
      })

      // Clear any pending reconnect timer
      if (reconnectTimer) {
        clearTimeout(reconnectTimer)
        reconnectTimer = null
      }
    })

    socket.on('connection_established', (data) => {
      console.log('🎉 Connection established:', data)
      set({ clientId: data.client_id })
    })

    socket.on('disconnect', (reason) => {
      console.warn('⚠️ Socket disconnected:', reason)
      set({ connected: false, clientId: null })

      // Only attempt reconnection if not intentional
      if (reason === 'io server disconnect') {
        // Server forcibly disconnected - attempt reconnection
        attemptReconnect()
      } else if (reason === 'transport close' || reason === 'transport error') {
        // Network issues - attempt reconnection
        attemptReconnect()
      }
      // 'io client disconnect' means we disconnected intentionally - no reconnect
    })

    socket.on('connect_error', (error) => {
      console.error('❌ Connection error:', error.message)
      set({
        connected: false,
        connecting: false,
        error: error.message,
      })
      
      attemptReconnect()
    })

    socket.on('error', (error) => {
      console.error('❌ Socket error:', error)
      set({ error: error.message || 'Unknown socket error' })
    })

    socket.on('reconnect', (attemptNumber) => {
      console.log('🔄 Reconnected after', attemptNumber, 'attempts')
      reconnectDelay = INITIAL_RECONNECT_DELAY
      set({ reconnectAttempts: 0, error: null })
    })

    socket.on('reconnect_attempt', (attemptNumber) => {
      console.log('🔄 Reconnect attempt', attemptNumber)
      set({ reconnectAttempts: attemptNumber, connecting: true })
    })

    socket.on('reconnect_error', (error) => {
      console.error('❌ Reconnect error:', error.message)
      set({ error: error.message })
    })

    socket.on('reconnect_failed', () => {
      console.error('❌ Reconnection failed after maximum attempts')
      set({
        connected: false,
        connecting: false,
        error: 'Failed to reconnect after maximum attempts',
      })
    })

    // Ping-pong for connection health monitoring
    socket.on('pong', () => {
      // Connection is alive
      set({ error: null })
    })

    // Handle server-side errors
    socket.on('error', (data) => {
      console.error('Server error:', data)
      set({ error: data.message || 'Server error' })
    })
  }

  const attemptReconnect = () => {
    const state = get()
    
    if (state.reconnectAttempts >= MAX_RECONNECT_ATTEMPTS) {
      console.error('❌ Max reconnection attempts reached')
      set({
        connecting: false,
        error: 'Maximum reconnection attempts exceeded. Please refresh the page.',
      })
      return
    }

    if (reconnectTimer) {
      return // Already attempting to reconnect
    }

    const currentDelay = reconnectDelay
    console.log(`⏳ Will attempt reconnect in ${currentDelay}ms...`)

    set({ connecting: true })

    reconnectTimer = setTimeout(() => {
      reconnectTimer = null
      const { socket } = get()
      
      if (socket && !socket.connected) {
        console.log('🔄 Attempting manual reconnect...')
        socket.connect()
        
        // Exponential backoff with jitter
        reconnectDelay = Math.min(
          reconnectDelay * 2 + Math.random() * 1000,
          MAX_RECONNECT_DELAY
        )
      }
    }, currentDelay)
  }

  return {
    socket: null,
    connected: false,
    connecting: false,
    error: null,
    reconnectAttempts: 0,
    clientId: null,

    connect: () => {
      const state = get()
      
      // Don't create multiple connections
      if (state.socket) {
        console.warn('⚠️ Socket already exists')
        if (!state.socket.connected) {
          state.socket.connect()
        }
        return
      }

      console.log('🔌 Initializing socket connection...')
      set({ connecting: true })

      const socket = io({
        transports: ['websocket', 'polling'],
        reconnection: true,
        reconnectionDelay: INITIAL_RECONNECT_DELAY,
        reconnectionDelayMax: MAX_RECONNECT_DELAY,
        reconnectionAttempts: MAX_RECONNECT_ATTEMPTS,
        timeout: 20000,
        autoConnect: true,
        // Enable multiplexing for better connection management
        multiplex: true,
      })

      setupSocketHandlers(socket)
      set({ socket })

      // Send periodic pings to detect connection issues
      const pingInterval = setInterval(() => {
        if (socket.connected) {
          socket.emit('ping', { timestamp: Date.now() })
        }
      }, 15000)

      // Clean up on disconnect
      socket.on('disconnect', () => {
        clearInterval(pingInterval)
      })
    },

    disconnect: () => {
      const { socket } = get()
      if (socket) {
        console.log('👋 Disconnecting socket...')
        socket.disconnect()
        set({
          socket: null,
          connected: false,
          connecting: false,
          clientId: null,
        })
      }
      
      if (reconnectTimer) {
        clearTimeout(reconnectTimer)
        reconnectTimer = null
      }
    },

    emit: (event: string, data: any) => {
      const { socket, connected } = get()
      if (socket && connected) {
        socket.emit(event, data)
      } else {
        console.warn('⚠️ Cannot emit - socket not connected')
        set({ error: 'Not connected to server' })
      }
    },

    on: (event: string, handler: (...args: any[]) => void) => {
      const { socket } = get()
      if (socket) {
        socket.on(event, handler)
      }
    },

    off: (event: string, handler: (...args: any[]) => void) => {
      const { socket } = get()
      if (socket) {
        socket.off(event, handler)
      }
    },

    resetError: () => {
      set({ error: null })
    },
  }
})
