# TheTeam Frontend

Modern React-based web interface for TheTeam agent coordination suite.

## Features

- **Flowchart Editor**: Interactive drag-and-drop visual flowchart builder using ReactFlow
  - Create, edit, and execute agent workflows
  - Modern node styling with hover effects and edit buttons
  - 5 color-coded edge types (default, conditional, loop, error, success)
  - Interactive edge type switching
  - Real-time execution visualization with node highlighting
  - Variable name display on hover
  - Import/export YAML configurations
  - Node-level output inspection

- **Chat Interface**: Real-time conversation with AI agents
  - Multiple conversation management
  - Base Model mode for direct model access
  - Dynamic agent/model selection and switching
  - Persistent conversation history
  - WebSocket-based real-time streaming responses

- **Agent Configuration**: Comprehensive agent management interface
  - Create and edit custom agents
  - Full parameter control: temperature, max tokens, top k, top p, repeat penalty
  - Multi-select tool assignment
  - Chain-of-thought flowchart selection
  - Tool auto-loop and iteration configuration
  - Memory system toggle
  - Test agents with sample prompts
  - Save configurations to file or runtime

- **Robust Networking**: Production-ready WebSocket implementation
  - Automatic reconnection with exponential backoff
  - Real-time connection status monitoring
  - Seamless error recovery
  - Ping/pong health checks every 15 seconds

## Technology Stack

- **React 18** with TypeScript
- **Vite** for fast development and building
- **ReactFlow** for flowchart visualization
- **Socket.IO Client** for WebSocket communication
- **Zustand** for state management
- **React Router** for client-side routing
- **Tailwind CSS** for styling

## Getting Started

### Prerequisites

- Node.js 18+ and npm
- Python 3.8+ (for backend)
- TheTeam backend server running

### Installation

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

### Development

Run the development server with hot-reload:

```bash
npm run dev
```

The interface will be available at `http://localhost:3000`

The Vite dev server proxies API requests to the backend at `http://localhost:5000`

### Building for Production

Build the frontend for production:

```bash
npm run build
```

This creates optimized static files in `../src/theteam/static/` which are served by the Flask backend.

### Running with Backend

To run the full stack:

1. Build the frontend (see above)
2. Start the backend server:
   ```bash
   # From project root
   theteam --host 0.0.0.0 --port 5000
   ```
3. Open `http://localhost:5000` in your browser

## Project Structure

```
frontend/
├── src/
│   ├── components/       # Reusable UI components
│   │   ├── Layout.tsx
│   │   └── ConnectionStatus.tsx
│   ├── pages/           # Main application pages
│   │   ├── FlowchartEditor.tsx
│   │   ├── ChatInterface.tsx
│   │   └── AgentConfig.tsx
│   ├── store/           # Zustand state stores
│   │   ├── socketStore.ts      # WebSocket connection management
│   │   ├── agentStore.ts       # Agent configuration state
│   │   ├── flowchartStore.ts   # Flowchart and execution state
│   │   └── chatStore.ts        # Chat and conversation state
│   ├── lib/             # Utility functions
│   ├── App.tsx          # Main application component
│   ├── main.tsx         # Application entry point
│   └── index.css        # Global styles
├── public/              # Static assets
├── package.json
├── tsconfig.json
├── vite.config.ts
└── tailwind.config.js
```

## WebSocket Events

### Client → Server

- `ping`: Health check
- `join_room`: Subscribe to updates for a specific resource
- `chat_message`: Send a chat message
- `execute_flowchart`: Start flowchart execution
- `stop_execution`: Stop a running execution

### Server → Client

- `connection_established`: Connection confirmation with client ID
- `pong`: Health check response
- `message_response`: Assistant response to chat message
- `execution_update`: Flowchart execution status update
- `node_execution`: Individual node execution event
- `execution_complete`: Flowchart execution completed
- `error`: Error message

## State Management

The application uses Zustand for state management with separate stores for different concerns:

- **socketStore**: WebSocket connection state and methods
- **agentStore**: Agent configurations and operations
- **flowchartStore**: Flowchart data and execution state
- **chatStore**: Conversation and message state

All stores integrate with the WebSocket connection for real-time updates.

## Styling

The application uses Tailwind CSS with a custom dark theme. The color palette is defined in `tailwind.config.js`.

## Development Tips

- **Hot Reload**: The dev server supports hot module replacement (HMR) for instant updates
- **Type Safety**: Use TypeScript types for all props and state
- **WebSocket Testing**: The ConnectionStatus component shows real-time connection state
- **Error Handling**: Check the browser console for detailed error logs

## Troubleshooting

### WebSocket Connection Issues

- Ensure the backend server is running on `http://localhost:5000`
- Check browser console for connection errors
- Verify firewall/proxy settings allow WebSocket connections

### Build Failures

- Clear node_modules and reinstall: `rm -rf node_modules && npm install`
- Check Node.js version: `node --version` (should be 18+)
- Clear Vite cache: `rm -rf node_modules/.vite`

### State Not Updating

- Check WebSocket connection status in the sidebar
- Verify the backend is sending the expected events
- Check for errors in the browser console

## License

MIT
