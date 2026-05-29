import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { useToastStore } from '../../store/toastStore'

beforeEach(() => {
  useToastStore.setState({ toasts: [] })
  vi.useFakeTimers()
})

afterEach(() => {
  vi.useRealTimers()
})

describe('toastStore — addToast', () => {
  it('adds a toast with the provided message and type', () => {
    useToastStore.getState().addToast('Hello', 'success')
    const { toasts } = useToastStore.getState()

    expect(toasts).toHaveLength(1)
    expect(toasts[0].message).toBe('Hello')
    expect(toasts[0].type).toBe('success')
  })

  it('defaults type to "info" when not provided', () => {
    useToastStore.getState().addToast('Info message')
    expect(useToastStore.getState().toasts[0].type).toBe('info')
  })

  it('assigns a unique id to every toast', () => {
    useToastStore.getState().addToast('A', 'info')
    useToastStore.getState().addToast('B', 'info')
    const { toasts } = useToastStore.getState()
    expect(toasts[0].id).not.toBe(toasts[1].id)
  })

  it('multiple toasts accumulate in order', () => {
    useToastStore.getState().addToast('first', 'info')
    useToastStore.getState().addToast('second', 'error')
    const { toasts } = useToastStore.getState()
    expect(toasts).toHaveLength(2)
    expect(toasts[0].message).toBe('first')
    expect(toasts[1].message).toBe('second')
  })

  it('stores the duration on the toast', () => {
    useToastStore.getState().addToast('msg', 'info', 3000)
    expect(useToastStore.getState().toasts[0].duration).toBe(3000)
  })

  it('auto-removes the toast after its duration', () => {
    useToastStore.getState().addToast('temp', 'info', 2000)
    expect(useToastStore.getState().toasts).toHaveLength(1)

    vi.advanceTimersByTime(2000)
    expect(useToastStore.getState().toasts).toHaveLength(0)
  })

  it('does NOT auto-remove when duration is 0', () => {
    useToastStore.getState().addToast('sticky', 'warning', 0)
    vi.advanceTimersByTime(30_000)
    expect(useToastStore.getState().toasts).toHaveLength(1)
  })

  it('only removes the toast whose timer fires, not others', () => {
    useToastStore.getState().addToast('short', 'info', 1000)
    useToastStore.getState().addToast('long', 'info', 5000)

    vi.advanceTimersByTime(1000)

    const { toasts } = useToastStore.getState()
    expect(toasts).toHaveLength(1)
    expect(toasts[0].message).toBe('long')
  })
})

describe('toastStore — removeToast', () => {
  it('removes the toast with the given id', () => {
    useToastStore.getState().addToast('remove me', 'error')
    const id = useToastStore.getState().toasts[0].id

    useToastStore.getState().removeToast(id)
    expect(useToastStore.getState().toasts).toHaveLength(0)
  })

  it('leaves other toasts intact', () => {
    useToastStore.getState().addToast('keep', 'success')
    useToastStore.getState().addToast('remove', 'error')

    const removeId = useToastStore.getState().toasts[1].id
    useToastStore.getState().removeToast(removeId)

    const { toasts } = useToastStore.getState()
    expect(toasts).toHaveLength(1)
    expect(toasts[0].message).toBe('keep')
  })

  it('is a no-op for an unknown id', () => {
    useToastStore.getState().addToast('stay', 'info')
    useToastStore.getState().removeToast('nonexistent-id')
    expect(useToastStore.getState().toasts).toHaveLength(1)
  })
})
