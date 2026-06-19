import { describe, it, expect, beforeEach } from 'vitest'
import { useConfirmStore, PendingConfirmation } from '../../store/confirmStore'

const sample: PendingConfirmation = {
  requestId: 'req-1',
  command: 'echo hello',
  conversationId: 'conv-1',
  messageId: 'msg-1',
}

beforeEach(() => {
  useConfirmStore.setState({ pendingConfirmation: null })
})

describe('confirmStore — initial state', () => {
  it('starts with no pending confirmation', () => {
    expect(useConfirmStore.getState().pendingConfirmation).toBeNull()
  })
})

describe('confirmStore — setPendingConfirmation', () => {
  it('stores the confirmation object', () => {
    useConfirmStore.getState().setPendingConfirmation(sample)
    expect(useConfirmStore.getState().pendingConfirmation).toEqual(sample)
  })

  it('overwrites an existing confirmation', () => {
    useConfirmStore.getState().setPendingConfirmation(sample)
    const newer: PendingConfirmation = {
      requestId: 'req-2',
      command: 'ls -la',
      conversationId: 'conv-2',
      messageId: 'msg-2',
    }
    useConfirmStore.getState().setPendingConfirmation(newer)
    expect(useConfirmStore.getState().pendingConfirmation?.requestId).toBe('req-2')
  })
})

describe('confirmStore — clearPendingConfirmation', () => {
  it('sets pendingConfirmation back to null', () => {
    useConfirmStore.getState().setPendingConfirmation(sample)
    useConfirmStore.getState().clearPendingConfirmation()
    expect(useConfirmStore.getState().pendingConfirmation).toBeNull()
  })

  it('is safe to call when already null', () => {
    expect(() => useConfirmStore.getState().clearPendingConfirmation()).not.toThrow()
    expect(useConfirmStore.getState().pendingConfirmation).toBeNull()
  })
})
