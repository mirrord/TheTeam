import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { useChatStore, Message, ConversationDetail } from '../../store/chatStore'

// ─── helpers ────────────────────────────────────────────────────────────────

function makeConversation(overrides: Partial<ConversationDetail> = {}): ConversationDetail {
  return {
    id: 'conv-1',
    title: 'Test Conversation',
    created_at: '2024-01-01T00:00:00Z',
    updated_at: '2024-01-01T00:00:00Z',
    message_count: 0,
    messages: [],
    ...overrides,
  }
}

function makeMessage(overrides: Partial<Message> = {}): Message {
  return {
    id: 'msg-1',
    role: 'assistant',
    content: 'Hello',
    timestamp: '2024-01-01T00:00:00Z',
    ...overrides,
  }
}

const initialState = {
  conversations: [],
  currentConversation: null,
  loading: false,
  sending: false,
  processing: false,
  error: null,
  streamingMessages: {},
}

beforeEach(() => {
  useChatStore.setState(initialState)
  vi.restoreAllMocks()
})

afterEach(() => {
  vi.restoreAllMocks()
})

// ─── Pure state mutations ────────────────────────────────────────────────────

describe('setCurrentConversation', () => {
  it('sets the current conversation', () => {
    const conv = makeConversation()
    useChatStore.getState().setCurrentConversation(conv)
    expect(useChatStore.getState().currentConversation).toEqual(conv)
  })

  it('clears the current conversation when passed null', () => {
    useChatStore.setState({ currentConversation: makeConversation() })
    useChatStore.getState().setCurrentConversation(null)
    expect(useChatStore.getState().currentConversation).toBeNull()
  })
})

describe('addMessage', () => {
  it('appends a message to the current conversation', () => {
    useChatStore.setState({ currentConversation: makeConversation() })
    const msg = makeMessage()
    useChatStore.getState().addMessage(msg)

    const { currentConversation } = useChatStore.getState()
    expect(currentConversation?.messages).toHaveLength(1)
    expect(currentConversation?.messages[0]).toEqual(msg)
  })

  it('does nothing when there is no current conversation', () => {
    useChatStore.getState().addMessage(makeMessage())
    expect(useChatStore.getState().currentConversation).toBeNull()
  })

  it('preserves existing messages when appending', () => {
    const existing = makeMessage({ id: 'msg-0', content: 'first' })
    useChatStore.setState({ currentConversation: makeConversation({ messages: [existing] }) })

    const newMsg = makeMessage({ id: 'msg-1', content: 'second' })
    useChatStore.getState().addMessage(newMsg)

    expect(useChatStore.getState().currentConversation?.messages).toHaveLength(2)
    expect(useChatStore.getState().currentConversation?.messages[1].content).toBe('second')
  })
})

describe('setProcessing', () => {
  it('sets processing to true', () => {
    useChatStore.getState().setProcessing('conv-1', true)
    expect(useChatStore.getState().processing).toBe(true)
  })

  it('sets processing to false', () => {
    useChatStore.setState({ processing: true })
    useChatStore.getState().setProcessing('conv-1', false)
    expect(useChatStore.getState().processing).toBe(false)
  })
})

// ─── Streaming pipeline ──────────────────────────────────────────────────────

describe('startStreaming', () => {
  it('creates an entry in streamingMessages', () => {
    useChatStore.getState().startStreaming('conv-1', 'stream-msg-1')

    const { streamingMessages } = useChatStore.getState()
    expect(streamingMessages['conv-1']).toEqual({ id: 'stream-msg-1', content: '' })
  })

  it('sets sending=true and processing=false', () => {
    useChatStore.setState({ sending: false, processing: true })
    useChatStore.getState().startStreaming('conv-1', 'id-1')

    expect(useChatStore.getState().sending).toBe(true)
    expect(useChatStore.getState().processing).toBe(false)
  })

  it('does not overwrite entries for other conversations', () => {
    useChatStore.setState({
      streamingMessages: { 'conv-99': { id: 'x', content: 'hi' } },
    })
    useChatStore.getState().startStreaming('conv-1', 'id-1')

    const { streamingMessages } = useChatStore.getState()
    expect(streamingMessages['conv-99']).toEqual({ id: 'x', content: 'hi' })
  })
})

describe('appendStreamChunk', () => {
  it('appends a chunk to the in-progress streaming content', () => {
    useChatStore.setState({
      streamingMessages: { 'conv-1': { id: 'id-1', content: 'Hello' } },
    })

    useChatStore.getState().appendStreamChunk('conv-1', 'id-1', ', World')

    expect(useChatStore.getState().streamingMessages['conv-1'].content).toBe('Hello, World')
  })

  it('is a no-op when there is no active stream for the conversation', () => {
    useChatStore.getState().appendStreamChunk('conv-1', 'id-1', ' chunk')
    expect(useChatStore.getState().streamingMessages['conv-1']).toBeUndefined()
  })

  it('accumulates multiple chunks in order', () => {
    useChatStore.setState({
      streamingMessages: { 'conv-1': { id: 'id-1', content: '' } },
    })

    useChatStore.getState().appendStreamChunk('conv-1', 'id-1', 'A')
    useChatStore.getState().appendStreamChunk('conv-1', 'id-1', 'B')
    useChatStore.getState().appendStreamChunk('conv-1', 'id-1', 'C')

    expect(useChatStore.getState().streamingMessages['conv-1'].content).toBe('ABC')
  })
})

describe('finalizeStreaming', () => {
  it('removes the streaming entry and appends the final message to the conversation', () => {
    const conv = makeConversation({ id: 'conv-1' })
    useChatStore.setState({
      currentConversation: conv,
      streamingMessages: { 'conv-1': { id: 'stream-id', content: 'streamed content' } },
      sending: true,
    })

    const finalMsg = makeMessage({ id: 'final-msg', content: 'streamed content', role: 'assistant' })
    useChatStore.getState().finalizeStreaming('conv-1', finalMsg)

    const state = useChatStore.getState()
    expect(state.streamingMessages['conv-1']).toBeUndefined()
    expect(state.sending).toBe(false)
    expect(state.processing).toBe(false)
    expect(state.currentConversation?.messages).toHaveLength(1)
    expect(state.currentConversation?.messages[0]).toEqual(finalMsg)
  })

  it('removes the streaming entry even when there is no matching current conversation', () => {
    useChatStore.setState({
      currentConversation: makeConversation({ id: 'other-conv' }),
      streamingMessages: { 'conv-1': { id: 'id-1', content: '' } },
    })

    useChatStore.getState().finalizeStreaming('conv-1', makeMessage())

    expect(useChatStore.getState().streamingMessages['conv-1']).toBeUndefined()
    // Other conversation is untouched
    expect(useChatStore.getState().currentConversation?.id).toBe('other-conv')
  })

  it('sets sending=false and processing=false in all cases', () => {
    useChatStore.setState({ sending: true, processing: true })
    useChatStore.getState().finalizeStreaming('conv-1', makeMessage())

    expect(useChatStore.getState().sending).toBe(false)
    expect(useChatStore.getState().processing).toBe(false)
  })
})

// ─── Fetch-backed actions (mocked) ──────────────────────────────────────────

describe('fetchConversations', () => {
  it('sets conversations on success', async () => {
    const convData = { conversations: [{ id: '1', title: 'Chat' }] }
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(JSON.stringify(convData), { status: 200 })
    )

    await useChatStore.getState().fetchConversations()

    expect(useChatStore.getState().conversations).toEqual(convData.conversations)
    expect(useChatStore.getState().loading).toBe(false)
  })

  it('sets error on failure', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response('', { status: 500 })
    )

    await useChatStore.getState().fetchConversations()

    expect(useChatStore.getState().error).toBeTruthy()
    expect(useChatStore.getState().loading).toBe(false)
  })
})

describe('createConversation', () => {
  it('returns the conversation_id and refreshes the list', async () => {
    vi.spyOn(globalThis, 'fetch')
      .mockResolvedValueOnce(
        new Response(JSON.stringify({ conversation_id: 'new-conv-1' }), { status: 200 })
      )
      .mockResolvedValueOnce(
        new Response(JSON.stringify({ conversations: [] }), { status: 200 })
      )

    const id = await useChatStore.getState().createConversation('agent-1', 'My Chat')
    expect(id).toBe('new-conv-1')
  })

  it('sets error and rethrows on failure', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response('', { status: 500 })
    )

    await expect(useChatStore.getState().createConversation()).rejects.toThrow()
    expect(useChatStore.getState().error).toBeTruthy()
  })
})

describe('deleteConversation', () => {
  it('clears currentConversation when deleting the active one', async () => {
    const conv = makeConversation({ id: 'conv-to-delete' })
    useChatStore.setState({ currentConversation: conv })

    vi.spyOn(globalThis, 'fetch')
      .mockResolvedValueOnce(new Response('{}', { status: 200 }))
      .mockResolvedValueOnce(new Response(JSON.stringify({ conversations: [] }), { status: 200 }))

    await useChatStore.getState().deleteConversation('conv-to-delete')
    expect(useChatStore.getState().currentConversation).toBeNull()
  })

  it('leaves currentConversation intact when deleting a different one', async () => {
    const conv = makeConversation({ id: 'keep-me' })
    useChatStore.setState({ currentConversation: conv })

    vi.spyOn(globalThis, 'fetch')
      .mockResolvedValueOnce(new Response('{}', { status: 200 }))
      .mockResolvedValueOnce(new Response(JSON.stringify({ conversations: [] }), { status: 200 }))

    await useChatStore.getState().deleteConversation('other-conv')
    expect(useChatStore.getState().currentConversation?.id).toBe('keep-me')
  })
})

describe('updateAgent', () => {
  it('updates agent_id in the current conversation when it matches', async () => {
    const conv = makeConversation({ id: 'conv-1', agent_id: 'old-agent' })
    useChatStore.setState({ currentConversation: conv })

    vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response('{}', { status: 200 })
    )

    await useChatStore.getState().updateAgent('conv-1', 'new-agent')
    expect(useChatStore.getState().currentConversation?.agent_id).toBe('new-agent')
  })

  it('does not mutate the conversation when ids do not match', async () => {
    const conv = makeConversation({ id: 'other', agent_id: 'unchanged' })
    useChatStore.setState({ currentConversation: conv })

    vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response('{}', { status: 200 })
    )

    await useChatStore.getState().updateAgent('conv-1', 'new-agent')
    expect(useChatStore.getState().currentConversation?.agent_id).toBe('unchanged')
  })
})
