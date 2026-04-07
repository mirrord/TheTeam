import { describe, it, expect, beforeEach } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { ToastContainer } from '../../components/ToastContainer'
import { useToastStore } from '../../store/toastStore'

beforeEach(() => {
  useToastStore.setState({ toasts: [] })
})

describe('ToastContainer', () => {
  it('renders nothing when there are no toasts', () => {
    const { container } = render(<ToastContainer />)
    // The outer wrapper div exists but has no children
    expect(container.querySelectorAll('[class*="rounded-lg"]')).toHaveLength(0)
  })

  it('renders one item per toast', () => {
    useToastStore.setState({
      toasts: [
        { id: '1', message: 'First', type: 'info' },
        { id: '2', message: 'Second', type: 'error' },
      ],
    })

    render(<ToastContainer />)
    expect(screen.getByText('First')).toBeInTheDocument()
    expect(screen.getByText('Second')).toBeInTheDocument()
  })

  it('displays the toast message text', () => {
    useToastStore.setState({
      toasts: [{ id: '1', message: 'Something went wrong', type: 'error' }],
    })

    render(<ToastContainer />)
    expect(screen.getByText('Something went wrong')).toBeInTheDocument()
  })

  it('removes a toast when the X button is clicked', () => {
    useToastStore.setState({
      toasts: [{ id: 'toast-abc', message: 'Dismiss me', type: 'success' }],
    })

    render(<ToastContainer />)

    // Click the dismiss (X) button
    const buttons = screen.getAllByRole('button')
    fireEvent.click(buttons[0])

    expect(useToastStore.getState().toasts).toHaveLength(0)
  })

  it('preserves other toasts when one is dismissed', () => {
    useToastStore.setState({
      toasts: [
        { id: 'keep', message: 'Keep me', type: 'info' },
        { id: 'remove', message: 'Remove me', type: 'warning' },
      ],
    })

    render(<ToastContainer />)

    // The buttons are rendered in order; click the second one
    const buttons = screen.getAllByRole('button')
    fireEvent.click(buttons[1])

    expect(useToastStore.getState().toasts).toHaveLength(1)
    expect(useToastStore.getState().toasts[0].id).toBe('keep')
  })

  it('renders success toasts with green styling', () => {
    useToastStore.setState({
      toasts: [{ id: '1', message: 'Done!', type: 'success' }],
    })

    const { container } = render(<ToastContainer />)
    // The toast item div contains green classes
    const toastDiv = container.querySelector('.bg-green-900')
    expect(toastDiv).not.toBeNull()
  })

  it('renders error toasts with red styling', () => {
    useToastStore.setState({
      toasts: [{ id: '1', message: 'Error!', type: 'error' }],
    })

    const { container } = render(<ToastContainer />)
    const toastDiv = container.querySelector('.bg-red-900')
    expect(toastDiv).not.toBeNull()
  })

  it('renders warning toasts with yellow styling', () => {
    useToastStore.setState({
      toasts: [{ id: '1', message: 'Warn!', type: 'warning' }],
    })

    const { container } = render(<ToastContainer />)
    const toastDiv = container.querySelector('.bg-yellow-900')
    expect(toastDiv).not.toBeNull()
  })

  it('renders info toasts with blue styling', () => {
    useToastStore.setState({
      toasts: [{ id: '1', message: 'Info', type: 'info' }],
    })

    const { container } = render(<ToastContainer />)
    const toastDiv = container.querySelector('.bg-blue-900')
    expect(toastDiv).not.toBeNull()
  })
})
