import { useSocketStore } from '../store/socketStore'
import { Wifi, WifiOff, Loader2, AlertTriangle } from 'lucide-react'

export default function ConnectionStatus() {
  const { connected, connecting, error, reconnectAttempts, resetError } = useSocketStore()
  
  if (connected) {
    return (
      <div className="flex items-center gap-2 text-green-400 text-sm">
        <Wifi className="w-4 h-4" />
        <span>Connected</span>
      </div>
    )
  }
  
  if (connecting) {
    return (
      <div className="flex items-center gap-2 text-yellow-400 text-sm">
        <Loader2 className="w-4 h-4 animate-spin" />
        <span>
          {reconnectAttempts > 0 ? `Reconnecting (${reconnectAttempts})...` : 'Connecting...'}
        </span>
      </div>
    )
  }
  
  if (error) {
    return (
      <div className="space-y-2">
        <div className="flex items-center gap-2 text-red-400 text-sm">
          <AlertTriangle className="w-4 h-4" />
          <span>Disconnected</span>
        </div>
        <p className="text-xs text-gray-400">{error}</p>
        <button
          onClick={resetError}
          className="text-xs text-primary-400 hover:text-primary-300"
        >
          Dismiss
        </button>
      </div>
    )
  }
  
  return (
    <div className="flex items-center gap-2 text-gray-400 text-sm">
      <WifiOff className="w-4 h-4" />
      <span>Disconnected</span>
    </div>
  )
}
