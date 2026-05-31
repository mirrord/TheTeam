/**
 * Store for managing pending tool confirmation requests.
 * When the server emits a tool_confirmation_request event, the pending
 * confirmation is stored here and displayed via ToolConfirmationModal.
 */

import { create } from 'zustand'

export interface PendingConfirmation {
  requestId: string
  command: string
  conversationId: string
  messageId: string
}

interface ConfirmState {
  pendingConfirmation: PendingConfirmation | null
  setPendingConfirmation: (confirmation: PendingConfirmation) => void
  clearPendingConfirmation: () => void
}

export const useConfirmStore = create<ConfirmState>((set) => ({
  pendingConfirmation: null,

  setPendingConfirmation: (confirmation) => {
    set({ pendingConfirmation: confirmation })
  },

  clearPendingConfirmation: () => {
    set({ pendingConfirmation: null })
  },
}))
