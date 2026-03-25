/**
 * Chat store for managing conversations and messages.
 */

import { create } from 'zustand'

export interface Message {
  id: string
  role: 'user' | 'assistant' | 'system'
  content: string
  timestamp: string
}

export interface Conversation {
  id: string
  title: string
  agent_id?: string
  created_at: string
  updated_at: string
  message_count: number
}

export interface ConversationDetail extends Conversation {
  messages: Message[]
}

interface ChatState {
  conversations: Conversation[]
  currentConversation: ConversationDetail | null
  loading: boolean
  sending: boolean
  error: string | null
  /** Active streaming messages keyed by conversation id. */
  streamingMessages: Record<string, { id: string; content: string }>

  // Actions
  fetchConversations: () => Promise<void>
  getConversation: (id: string) => Promise<void>
  createConversation: (agent_id?: string, title?: string) => Promise<string>
  deleteConversation: (id: string) => Promise<void>
  sendMessage: (conversationId: string, message: string) => Promise<void>
  updateAgent: (conversationId: string, agent_id: string) => Promise<void>
  setCurrentConversation: (conversation: ConversationDetail | null) => void
  addMessage: (message: Message) => void
  startStreaming: (conversationId: string, messageId: string) => void
  appendStreamChunk: (conversationId: string, messageId: string, chunk: string) => void
  finalizeStreaming: (conversationId: string, message: Message) => void
}

export const useChatStore = create<ChatState>((set, get) => ({
  conversations: [],
  currentConversation: null,
  loading: false,
  sending: false,
  error: null,
  streamingMessages: {},

  fetchConversations: async () => {
    set({ loading: true, error: null })
    try {
      const response = await fetch('/api/v1/chat/conversations')
      if (!response.ok) throw new Error('Failed to fetch conversations')
      const data = await response.json()
      set({ conversations: data.conversations, loading: false })
    } catch (error: any) {
      set({ error: error.message, loading: false })
    }
  },

  getConversation: async (id: string) => {
    set({ loading: true, error: null })
    try {
      const response = await fetch(`/api/v1/chat/conversations/${id}`)
      if (!response.ok) throw new Error('Failed to fetch conversation')
      const data = await response.json()
      set({ currentConversation: data.conversation, loading: false })
    } catch (error: any) {
      set({ error: error.message, loading: false })
    }
  },

  createConversation: async (agent_id?: string, title?: string) => {
    set({ loading: true, error: null })
    try {
      const response = await fetch('/api/v1/chat/conversations', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ agent_id, title }),
      })
      if (!response.ok) throw new Error('Failed to create conversation')
      const data = await response.json()
      await get().fetchConversations()
      set({ loading: false })
      return data.conversation_id
    } catch (error: any) {
      set({ error: error.message, loading: false })
      throw error
    }
  },

  deleteConversation: async (id: string) => {
    set({ loading: true, error: null })
    try {
      const response = await fetch(`/api/v1/chat/conversations/${id}`, {
        method: 'DELETE',
      })
      if (!response.ok) throw new Error('Failed to delete conversation')
      await get().fetchConversations()
      if (get().currentConversation?.id === id) {
        set({ currentConversation: null })
      }
      set({ loading: false })
    } catch (error: any) {
      set({ error: error.message, loading: false })
      throw error
    }
  },

  sendMessage: async (conversationId: string, message: string) => {
    set({ sending: true, error: null })
    try {
      // Message will be sent via WebSocket for real-time response
      // But we also provide REST endpoint as fallback
      const response = await fetch(`/api/v1/chat/conversations/${conversationId}/messages`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message }),
      })
      if (!response.ok) throw new Error('Failed to send message')
      set({ sending: false })
    } catch (error: any) {
      set({ error: error.message, sending: false })
      throw error
    }
  },

  updateAgent: async (conversationId: string, agent_id: string) => {
    set({ loading: true, error: null })
    try {
      const response = await fetch(`/api/v1/chat/conversations/${conversationId}/agent`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ agent_id }),
      })
      if (!response.ok) throw new Error('Failed to update agent')
      
      // Update current conversation if it's the one being modified
      const current = get().currentConversation
      if (current && current.id === conversationId) {
        set({
          currentConversation: { ...current, agent_id },
          loading: false,
        })
      } else {
        set({ loading: false })
      }
    } catch (error: any) {
      set({ error: error.message, loading: false })
      throw error
    }
  },

  setCurrentConversation: (conversation: ConversationDetail | null) => {
    set({ currentConversation: conversation })
  },

  addMessage: (message: Message) => {
    set((state) => {
      if (!state.currentConversation) return state
      return {
        currentConversation: {
          ...state.currentConversation,
          messages: [...state.currentConversation.messages, message],
        },
      }
    })
  },

  startStreaming: (conversationId: string, messageId: string) => {
    set((state) => ({
      streamingMessages: {
        ...state.streamingMessages,
        [conversationId]: { id: messageId, content: '' },
      },
      sending: true,
    }))
  },

  appendStreamChunk: (conversationId: string, _messageId: string, chunk: string) => {
    set((state) => {
      const current = state.streamingMessages[conversationId]
      if (!current) return state
      return {
        streamingMessages: {
          ...state.streamingMessages,
          [conversationId]: { ...current, content: current.content + chunk },
        },
      }
    })
  },

  finalizeStreaming: (conversationId: string, message: Message) => {
    set((state) => {
      const { [conversationId]: _removed, ...rest } = state.streamingMessages
      const conv = state.currentConversation
      if (!conv || conv.id !== conversationId) {
        return { streamingMessages: rest, sending: false }
      }
      return {
        streamingMessages: rest,
        sending: false,
        currentConversation: {
          ...conv,
          messages: [...conv.messages, message],
        },
      }
    })
  },
}))
