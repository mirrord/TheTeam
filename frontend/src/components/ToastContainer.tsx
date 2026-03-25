import { X, CheckCircle, XCircle, AlertTriangle, Info } from 'lucide-react'
import { useToastStore, Toast as ToastType } from '../store/toastStore'

function ToastItem({ toast }: { toast: ToastType }) {
  const { removeToast } = useToastStore()

  const icons = {
    success: <CheckCircle className="w-5 h-5 text-green-400" />,
    error: <XCircle className="w-5 h-5 text-red-400" />,
    warning: <AlertTriangle className="w-5 h-5 text-yellow-400" />,
    info: <Info className="w-5 h-5 text-blue-400" />,
  }

  const colors = {
    success: 'bg-green-900 border-green-600',
    error: 'bg-red-900 border-red-600',
    warning: 'bg-yellow-900 border-yellow-600',
    info: 'bg-blue-900 border-blue-600',
  }

  return (
    <div
      className={`flex items-start gap-3 p-4 rounded-lg border shadow-lg ${colors[toast.type]} animate-slide-in`}
    >
      {icons[toast.type]}
      <div className="flex-1 text-sm text-gray-100">
        {toast.message}
      </div>
      <button
        onClick={() => removeToast(toast.id)}
        className="text-gray-400 hover:text-gray-200 transition-colors"
      >
        <X className="w-4 h-4" />
      </button>
    </div>
  )
}

export function ToastContainer() {
  const { toasts } = useToastStore()

  return (
    <div className="fixed top-4 right-4 z-50 flex flex-col gap-2 max-w-md">
      {toasts.map((toast) => (
        <ToastItem key={toast.id} toast={toast} />
      ))}
    </div>
  )
}
