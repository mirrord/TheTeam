import { Outlet, Link, useLocation } from 'react-router-dom'
import { MessageSquare, Network, Settings, Activity } from 'lucide-react'
import ConnectionStatus from './ConnectionStatus'
import { ToastContainer } from './ToastContainer'

export default function Layout() {
  const location = useLocation()
  
  const navItems = [
    { path: '/flowcharts', icon: Network, label: 'Flowcharts' },
    { path: '/chat', icon: MessageSquare, label: 'Chat' },
    { path: '/agents', icon: Settings, label: 'Agents' },
  ]
  
  return (
    <div className="flex h-screen bg-gray-900 text-gray-100">
      <ToastContainer />
      {/* Sidebar */}
      <div className="w-64 bg-gray-800 border-r border-gray-700 flex flex-col">
        <div className="p-4 border-b border-gray-700">
          <h1 className="text-2xl font-bold text-primary-400 flex items-center gap-2">
            <Activity className="w-6 h-6" />
            TheTeam
          </h1>
          <p className="text-xs text-gray-400 mt-1">Agent Coordination Suite</p>
        </div>
        
        <nav className="flex-1 p-4 space-y-2">
          {navItems.map((item) => {
            const Icon = item.icon
            const isActive = location.pathname.startsWith(item.path)
            
            return (
              <Link
                key={item.path}
                to={item.path}
                className={`flex items-center gap-3 px-4 py-3 rounded-lg transition-colors ${
                  isActive
                    ? 'bg-primary-600 text-white'
                    : 'text-gray-300 hover:bg-gray-700'
                }`}
              >
                <Icon className="w-5 h-5" />
                <span>{item.label}</span>
              </Link>
            )
          })}
        </nav>
        
        <div className="p-4 border-t border-gray-700">
          <ConnectionStatus />
        </div>
      </div>
      
      {/* Main content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        <Outlet />
      </div>
    </div>
  )
}
