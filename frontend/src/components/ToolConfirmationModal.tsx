import { ShieldQuestion } from 'lucide-react'
import { useConfirmStore } from '../store/confirmStore'
import { useSocketStore } from '../store/socketStore'

/**
 * Modal that blocks UI and asks the user to approve or deny a pending tool call.
 * It is shown whenever the server emits a `tool_confirmation_request` event.
 * The user's response is sent back via `tool_confirmation_response`.
 */
export function ToolConfirmationModal() {
  const { pendingConfirmation, clearPendingConfirmation } = useConfirmStore()
  const { emit } = useSocketStore()

  if (!pendingConfirmation) return null

  const respond = (approved: boolean) => {
    emit('tool_confirmation_response', {
      request_id: pendingConfirmation.requestId,
      approved,
    })
    clearPendingConfirmation()
  }

  return (
    <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50">
      <div className="bg-gray-800 rounded-xl p-6 max-w-lg w-full mx-4 space-y-4 border border-gray-700 shadow-2xl">
        <h3 className="text-lg font-semibold flex items-center gap-2 text-yellow-300">
          <ShieldQuestion className="w-5 h-5 flex-shrink-0" />
          Tool Call Requires Approval
        </h3>
        <p className="text-sm text-gray-300">
          The agent wants to run the following command. Allow it to proceed?
        </p>
        <pre className="bg-gray-900 rounded-lg p-3 text-xs text-green-300 font-mono whitespace-pre-wrap break-all border border-gray-600">
          {pendingConfirmation.command}
        </pre>
        <div className="flex justify-end gap-2">
          <button
            onClick={() => respond(false)}
            className="px-4 py-2 rounded-lg bg-gray-700 hover:bg-gray-600 text-sm text-gray-200 transition-colors"
          >
            Deny
          </button>
          <button
            onClick={() => respond(true)}
            className="px-4 py-2 rounded-lg bg-green-700 hover:bg-green-600 text-sm text-white transition-colors"
          >
            Allow
          </button>
        </div>
      </div>
    </div>
  )
}
