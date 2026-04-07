import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { EdgeEditDialog } from '../../components/EdgeEditDialog'

const defaultEdgeData = {
  condition: { type: 'AlwaysCondition', enabled: true },
  priority: 9,
  output_key: 'default',
  input_key: 'default',
}

describe('EdgeEditDialog — visibility', () => {
  it('renders nothing when isOpen is false', () => {
    const { container } = render(
      <EdgeEditDialog
        isOpen={false}
        onClose={vi.fn()}
        edgeData={defaultEdgeData}
        onSave={vi.fn()}
      />
    )
    expect(container).toBeEmptyDOMElement()
  })

  it('renders the dialog when isOpen is true', () => {
    render(
      <EdgeEditDialog
        isOpen={true}
        onClose={vi.fn()}
        edgeData={defaultEdgeData}
        onSave={vi.fn()}
      />
    )
    expect(screen.getByText('Edit Edge')).toBeInTheDocument()
  })
})

describe('EdgeEditDialog — form initialisation', () => {
  it('populates fields from edgeData', () => {
    render(
      <EdgeEditDialog
        isOpen={true}
        onClose={vi.fn()}
        edgeData={{
          condition: { type: 'RegexCondition', enabled: false },
          priority: 3,
          output_key: 'myOutput',
          input_key: 'myInput',
        }}
        onSave={vi.fn()}
      />
    )

    const conditionSelect = screen.getByRole('combobox') as HTMLSelectElement
    expect(conditionSelect.value).toBe('RegexCondition')

    const priorityInput = screen.getByDisplayValue('3') as HTMLInputElement
    expect(priorityInput.value).toBe('3')

    expect((screen.getByDisplayValue('myOutput') as HTMLInputElement).value).toBe('myOutput')
    expect((screen.getByDisplayValue('myInput') as HTMLInputElement).value).toBe('myInput')
  })

  it('defaults to priority 9, output_key "default", input_key "default"', () => {
    render(
      <EdgeEditDialog
        isOpen={true}
        onClose={vi.fn()}
        edgeData={{}}
        onSave={vi.fn()}
      />
    )

    expect((screen.getByDisplayValue('9') as HTMLInputElement).value).toBe('9')
    const defaultValues = screen.getAllByDisplayValue('default')
    expect(defaultValues).toHaveLength(2)
  })
})

describe('EdgeEditDialog — CountCondition limit field', () => {
  it('hides the limit field for non-CountCondition types', () => {
    render(
      <EdgeEditDialog
        isOpen={true}
        onClose={vi.fn()}
        edgeData={defaultEdgeData}
        onSave={vi.fn()}
      />
    )
    expect(screen.queryByText(/Maximum number of times/i)).toBeNull()
  })

  it('shows the limit field when CountCondition is selected', async () => {
    const user = userEvent.setup()
    render(
      <EdgeEditDialog
        isOpen={true}
        onClose={vi.fn()}
        edgeData={defaultEdgeData}
        onSave={vi.fn()}
      />
    )

    await user.selectOptions(screen.getByRole('combobox'), 'CountCondition')
    expect(screen.getByText(/Maximum number of times/i)).toBeInTheDocument()
  })

  it('preserves the limit value when editing it', () => {
    render(
      <EdgeEditDialog
        isOpen={true}
        onClose={vi.fn()}
        edgeData={{ condition: { type: 'CountCondition', limit: 1 } }}
        onSave={vi.fn()}
      />
    )

    // The limit number input is the one with value 1
    const limitInput = screen.getByDisplayValue('1') as HTMLInputElement
    fireEvent.change(limitInput, { target: { value: '5' } })
    expect(limitInput.value).toBe('5')
  })
})

describe('EdgeEditDialog — close actions', () => {
  it('calls onClose when the X button is clicked', () => {
    const onClose = vi.fn()
    render(
      <EdgeEditDialog isOpen={true} onClose={onClose} edgeData={defaultEdgeData} onSave={vi.fn()} />
    )

    // The X (close) button in the header
    const closeButton = screen.getByRole('button', { name: '' })
    fireEvent.click(closeButton)
    expect(onClose).toHaveBeenCalledTimes(1)
  })

  it('calls onClose when the Cancel button is clicked', () => {
    const onClose = vi.fn()
    render(
      <EdgeEditDialog isOpen={true} onClose={onClose} edgeData={defaultEdgeData} onSave={vi.fn()} />
    )

    fireEvent.click(screen.getByText('Cancel'))
    expect(onClose).toHaveBeenCalledTimes(1)
  })
})

describe('EdgeEditDialog — save action', () => {
  it('calls onSave with the current form values', () => {
    const onSave = vi.fn()

    render(
      <EdgeEditDialog
        isOpen={true}
        onClose={vi.fn()}
        edgeData={defaultEdgeData}
        onSave={onSave}
      />
    )

    // There are two 'default' inputs (output_key and input_key) — change the first (output_key)
    const [outputKeyInput] = screen.getAllByDisplayValue('default') as HTMLInputElement[]
    fireEvent.change(outputKeyInput, { target: { value: 'result' } })

    fireEvent.click(screen.getByText('Save'))

    expect(onSave).toHaveBeenCalledTimes(1)
    const saved = onSave.mock.calls[0][0]
    expect(saved.output_key).toBe('result')
    expect(saved.condition.type).toBe('AlwaysCondition')
  })

  it('calls onClose after saving', () => {
    const onClose = vi.fn()
    render(
      <EdgeEditDialog
        isOpen={true}
        onClose={onClose}
        edgeData={defaultEdgeData}
        onSave={vi.fn()}
      />
    )

    fireEvent.click(screen.getByText('Save'))
    expect(onClose).toHaveBeenCalledTimes(1)
  })

  it('includes the limit in the saved data for CountCondition', async () => {
    const onSave = vi.fn()
    const user = userEvent.setup()

    render(
      <EdgeEditDialog
        isOpen={true}
        onClose={vi.fn()}
        edgeData={{ condition: { type: 'CountCondition', limit: 3 } }}
        onSave={onSave}
      />
    )

    fireEvent.click(screen.getByText('Save'))

    const saved = onSave.mock.calls[0][0]
    expect(saved.condition.limit).toBe(3)
  })

  it('omits the limit key for non-CountCondition types', () => {
    const onSave = vi.fn()
    render(
      <EdgeEditDialog
        isOpen={true}
        onClose={vi.fn()}
        edgeData={defaultEdgeData}
        onSave={onSave}
      />
    )

    fireEvent.click(screen.getByText('Save'))

    const saved = onSave.mock.calls[0][0]
    expect(saved.condition.limit).toBeUndefined()
  })
})

describe('EdgeEditDialog — edgeData prop update', () => {
  it('re-initialises form fields when edgeData changes', () => {
    const { rerender } = render(
      <EdgeEditDialog
        isOpen={true}
        onClose={vi.fn()}
        edgeData={{ condition: { type: 'AlwaysCondition' }, priority: 5 }}
        onSave={vi.fn()}
      />
    )

    rerender(
      <EdgeEditDialog
        isOpen={true}
        onClose={vi.fn()}
        edgeData={{ condition: { type: 'ErrorCondition' }, priority: 2 }}
        onSave={vi.fn()}
      />
    )

    const conditionSelect = screen.getByRole('combobox') as HTMLSelectElement
    expect(conditionSelect.value).toBe('ErrorCondition')
    expect((screen.getByDisplayValue('2') as HTMLInputElement).value).toBe('2')
  })
})
