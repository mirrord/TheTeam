import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom'
import { useEffect } from 'react'
import Layout from './components/Layout'
import FlowchartEditor from './pages/FlowchartEditor'
import ChatInterface from './pages/ChatInterface'
import AgentConfig from './pages/AgentConfig'
import EdgeDebugger from './pages/EdgeDebugger'
import { useSocketStore } from './store/socketStore'
import { ReactFlowProvider } from 'reactflow'

function App() {
  const { connect } = useSocketStore()

  useEffect(() => {
    // Connect to WebSocket on mount
    connect()
  }, [connect])

  return (
    <Router>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Navigate to="/flowcharts" replace />} />
          <Route
            path="flowcharts"
            element={
              <ReactFlowProvider>
                <FlowchartEditor />
              </ReactFlowProvider>
            }
          />
          <Route
            path="flowcharts/:id"
            element={
              <ReactFlowProvider>
                <FlowchartEditor />
              </ReactFlowProvider>
            }
          />
          <Route path="chat" element={<ChatInterface />} />
          <Route path="chat/:id" element={<ChatInterface />} />
          <Route path="agents" element={<AgentConfig />} />
        </Route>
        <Route
          path="/debug/edges"
          element={
            <ReactFlowProvider>
              <EdgeDebugger />
            </ReactFlowProvider>
          }
        />
      </Routes>
    </Router>
  )
}

export default App
