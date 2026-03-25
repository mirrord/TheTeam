# TheTeam Web Interface

Comprehensive guide to TheTeam's web-based interface for agent coordination and workflow management.

## Overview

TheTeam provides a modern, React-based web interface that allows you to:
- Visually create and edit agent workflows
- Chat with AI agents in real-time
- Manage agent configurations
- Execute and monitor workflow executions
- Import/export YAML configurations

The interface is built with production-ready features including robust WebSocket communication with automatic reconnection and real-time updates.

## Architecture

### Backend (Flask + SocketIO)

The backend is built with Flask and Flask-SocketIO, providing both REST API endpoints and WebSocket communication.

**Components:**
- `server.py`: Main Flask application
- `api/`: REST API endpoints (Flask Blueprints)
  - `agents.py`: Agent management endpoints
  - `flowcharts.py`: Flowchart CRUD operations
  - `chat.py`: Conversation management
  - `system.py`: System information
- `api/socketio_handlers.py`: WebSocket event handlers (plain module, registered via `register_handlers(socketio)`, not a Blueprint)
- `services/`: Business logic layer
  - `agent_service.py`: Agent operations
  - `flowchart_service.py`: Flowchart operations and execution
  - `chat_service.py`: Chat and conversation management

**WebSocket Events:**

*Client → Server:*
- `ping`: Connection health check
- `join_room`: Subscribe to resource updates
- `chat_message`: Send a message in a conversation
- `execute_flowchart`: Start workflow execution
- `stop_execution`: Stop running execution

*Server → Client:*
- `connection_established`: Connection confirmation
- `pong`: Health check response
- `message_response`: Chat response from agent
- `execution_update`: Execution status update
- `node_execution`: Individual node execution event
- `node_complete`: Node finished with output
- `execution_complete`: Workflow finished
- `error`: Error message

### Frontend (React + TypeScript)

The frontend is built with React 18 and TypeScript, using modern libraries for state management and UI components.

**Key Technologies:**
- **React 18**: Modern React with hooks
- **TypeScript**: Type safety throughout
- **ReactFlow**: Visual flowchart editor
- **Zustand**: Lightweight state management
- **Socket.IO Client**: WebSocket communication
- **Vite**: Fast development and building
- **Tailwind CSS**: Utility-first styling

**State Stores:**
- `socketStore`: WebSocket connection management
- `agentStore`: Agent configurations
- `flowchartStore`: Flowchart data and execution
- `chatStore`: Conversations and messages

## Getting Started

### Prerequisites

- Python 3.8+
- Node.js 18+
- Ollama installed and running

### Installation

1. **Install Python dependencies:**
   ```bash
   # From project root, with venv activated
   pip install -e .
   ```

2. **Install frontend dependencies:**
   ```bash
   cd frontend
   npm install
   ```

### Development Workflow

For active development with hot-reload:

1. **Terminal 1 - Backend:**
   ```bash
   # From project root
   theteam server --debug
   ```

2. **Terminal 2 - Frontend:**
   ```bash
   cd frontend
   npm run dev
   ```

3. **Open browser:**
   Navigate to `http://localhost:3000`

The Vite dev server proxies API and WebSocket requests to the backend at port 5000.

### Production Build

1. **Build frontend:**
   ```bash
   cd frontend
   npm run build
   ```
   This creates optimized static files in `src/theteam/static/`

2. **Start production server:**
   ```bash
   theteam --host 0.0.0.0 --port 5000
   ```

3. **Open browser:**
   Navigate to `http://localhost:5000`

## Features Deep Dive

### Flowchart Editor

The flowchart editor provides a visual interface for creating and managing agent workflows with modern, interactive node and edge components.

**Features:**
- Drag-and-drop node placement
- Visual edge connections with type indicators
- Interactive node editing
- Real-time execution visualization
- Node-level output inspection
- YAML import/export
- Validation before execution

**Workflow:**
1. Select or create a flowchart from the sidebar
2. Add nodes by dragging from the node palette
3. Connect nodes by dragging between connection points
4. Hover over nodes to see variable names
5. Click edit button on nodes to modify parameters
6. Click edge type labels to change edge types
7. Save the flowchart
8. Execute and watch real-time progress

**Node Types:**
- `prompt`: LLM prompt with variable interpolation (blue border)
- `condition`: Conditional branching (orange border)
- `toolcall`: Execute CLI tools (green border)
- `custom`: Custom Python code (purple border)
- `textparse`: Text parsing and extraction (amber border)

**Node Features:**
- Modern styling with hover effects and transitions
- Edit button (top-right) appears on hover
- Input variable names shown on left side when hovering (aligned precisely with handles)
- Output variable names shown on right side when hovering
- Color-coded connection handles
- Enhanced shadows and border highlights on hover
- Dynamic sizing: nodes automatically expand to accommodate multiple input variables
- Minimum height ensures comfortable spacing between connection sites (40px per handle)

**Edge Types & Colors:**

Edges are automatically color-coded based on their condition type. When loading flowcharts from YAML, the condition types are mapped to visual edge types:

- `default`: Slate gray (#64748b) - Standard flow (AlwaysCondition, or no condition)
- `conditional`: Amber (#f59e0b) - Conditional branches (RegexCondition, ContextCondition, CustomCondition)
- `loop`: Violet (#8b5cf6) - Loop connections (CountCondition, LoopCondition)
- `error`: Red (#ef4444) - Error handling paths (ErrorCondition)
- `success`: Green (#10b981) - Success paths (SuccessCondition)

**Edge Features:**
- Type-based color rendering (no manual styling required)
- Type label hidden by default
- Hover over edge to see type label
- 20px padded hover area for easier interaction
- Click type label to cycle through edge types
- Edge thickness increases on hover (2px default, 3px on hover)
- Smooth color transitions

**Execution Monitoring:**
- Active nodes highlighted in blue
- Completed nodes with checkmark
- Failed nodes in red
- Real-time output display per node

### Chat Interface

Real-time chat interface for conversing with AI agents.

**Features:**
- Multiple concurrent conversations
- Conversation history persistence
- Dynamic agent switching
- Base Model mode for direct model access
- Real-time responses via WebSocket
- Markdown rendering
- Conversation management (create, delete, rename)

**Workflow:**
1. Create a new conversation or select existing
2. Select an agent or Base Model mode
3. If Base Model selected, choose a model from the dropdown
4. Type message and press Enter or click Send
5. Responses stream in real-time
6. Switch agents mid-conversation seamlessly

**Agent Selection:**
- Click settings icon to open agent selector
- Choose "Base Model" for direct model access without agent configuration
- Choose from configured agents for specialized behavior
- When Base Model is selected, a model dropdown appears
- Base Model creates a temporary agent with default parameters (temperature: 0.7, max_tokens: 2048, no tools)
- Changes apply immediately without reload
- Current agent/model displayed in conversation header

### Agent Configuration

Comprehensive interface for managing agent configurations.

**Features:**
- Create custom agents
- Edit existing agents
- Test agents with sample prompts
- Save to file or runtime
- Full control over all agent parameters

**Configuration Sections:**

**Basic Information:**
- Agent ID (immutable after creation)
- Display name
- Model selection (dropdown of available models)
- System prompt (multiline text)

**Generation Parameters:**
- Temperature (0-2, step 0.1)
- Max Tokens (integer)
- Top K (sampling parameter)
- Top P (nucleus sampling, 0-1)
- Repeat Penalty (0-2)

**Tools & Capabilities:**
- Available Tools (multi-select checkboxes from discovered tools)
- Tool Auto Loop (boolean toggle)
- Max Tool Iterations (1-20)
- Memory Tools Enable (boolean toggle)

**Chain-of-Thought Flowchart:**
- Flowchart selection dropdown (optional)
- Select from available flowcharts to guide reasoning
- None option for standard behavior

**Persistence:**
- Save to file toggle for persistent agents across sessions

**Workflow:**
1. Select agent from sidebar or create new
2. Click Edit to modify configuration
3. Adjust parameters in organized sections
4. Configure tools and flowchart if needed
5. Test with sample prompt (optional)
6. Save changes

**Testing:**
- Enter a test prompt in the test section
- Click Test button
- View response in real-time
- Iterate on configuration as needed

### YAML Import/Export

Seamlessly work with YAML configuration files.

**Import:**
1. Click Import button in flowchart editor
2. Select YAML file from disk
3. Flowchart loads and renders immediately
4. Edit visually or export back to YAML

**Export:**
1. Select a flowchart
2. Click Export button
3. YAML file downloads automatically
4. Use in CLI tools or share with others

**Format:**
The web interface uses the same YAML format as the CLI tools, ensuring full compatibility.

## WebSocket Communication

The interface uses WebSocket for real-time bidirectional communication.

### Connection Management

**Features:**
- Automatic connection on page load
- Exponential backoff retry on disconnect
- Max 10 reconnection attempts
- Connection status indicator in sidebar
- Ping/pong health checks every 15 seconds

**Reconnection Logic:**
1. On disconnect, wait with exponential backoff
2. Initial delay: 1 second
3. Max delay: 30 seconds
4. Retry up to 10 times
5. Show error if max attempts exceeded

**Error Handling:**
- Network errors shown in connection status
- Failed messages queued for retry
- User notified of connection issues
- Graceful degradation to REST API if needed

### Real-time Updates

**Chat Messages:**
- User message sent via WebSocket
- Optimistic UI update
- Assistant response streams in real-time
- Messages persisted server-side

**Flowchart Execution:**
- Execution start notification
- Node-by-node progress updates
- Output captured and displayed
- Completion or error notification

**State Synchronization:**
- Changes propagate immediately
- No page reload required
- Multiple tabs stay in sync
- Conflict resolution server-side

## Customization

### Theming

The interface uses Tailwind CSS with a custom dark theme. Modify `frontend/tailwind.config.js` to customize colors:

```javascript
theme: {
  extend: {
    colors: {
      primary: {
        // Your custom color scale
      },
    },
  },
}
```

### Adding Components

1. Create component in `frontend/src/components/`
2. Use TypeScript for props and state
3. Import Tailwind classes for styling
4. Add to appropriate page or layout

### Extending API

1. Add endpoint in appropriate `api/*.py` file
2. Add corresponding service method in `services/*.py`
3. Update frontend store in `store/*.ts`
4. Use fetch or WebSocket in components

## Troubleshooting

### Backend Issues

**Server won't start:**
- Check if port 5000 is already in use
- Verify Flask dependencies installed
- Check Python version (3.8+)

**WebSocket connection fails:**
- Verify backend is running
- Check firewall settings
- Try disabling VPN/proxy

**API errors:**
- Check backend logs for details
- Verify Ollama is running
- Check file permissions on configs/

### Frontend Issues

**npm install fails:**
- Clear node_modules: `rm -rf node_modules`
- Check Node.js version (18+)
- Try `npm cache clean --force`

**Build fails:**
- Clear Vite cache: `rm -rf node_modules/.vite`
- Check for TypeScript errors
- Verify all dependencies installed

**WebSocket won't connect:**
- Check backend URL in browser console
- Verify CORS settings
- Check browser DevTools Network tab

**State not updating:**
- Check WebSocket connection status
- Verify event handlers registered
- Check browser console for errors

### Common Issues

**"Not connected to server":**
- Ensure backend is running
- Check connection status in sidebar
- Wait for automatic reconnection

**"Maximum reconnection attempts exceeded":**
- Backend is down or unreachable
- Check backend logs
- Restart backend server
- Refresh browser page

**Flowchart won't execute:**
- Validate flowchart first
- Check for missing agents
- Verify Ollama is running
- Check backend logs for errors

## Performance Tips

### Frontend

- Use React DevTools Profiler to identify slow renders
- Memoize expensive computations with `useMemo`
- Debounce rapid user inputs
- Lazy load large components
- Optimize ReactFlow rendering for large graphs

### Backend

- Use background threads for long operations
- Implement caching for frequent queries
- Batch database operations
- Monitor memory usage during executions
- Use connection pooling for Ollama

### WebSocket

- Subscribe only to needed events
- Unsubscribe when component unmounts
- Batch frequent updates
- Implement rate limiting server-side
- Use rooms for targeted updates

## Security Considerations

Since this is a local-only single-user application, security is simplified:

- No authentication required
- No user sessions
- No data encryption (local only)
- Execute permissions on system binaries

**Important:**
- Do NOT expose to public internet
- Run on localhost or secure local network only
- Review custom code before execution
- Back up configurations regularly

## Future Enhancements

Potential improvements for future versions:

- **Multi-user support**: Add authentication and per-user data
- **Collaboration**: Real-time collaborative editing
- **Version control**: Track flowchart versions
- **Templates**: Pre-built flowchart templates
- **Monitoring**: Dashboard for execution metrics
- **Plugins**: Plugin system for custom nodes
- **Mobile**: Responsive design for mobile devices
- **PWA**: Progressive web app with offline support

## Contributing

Contributions are welcome! Areas for improvement:

- UI/UX enhancements
- Additional node types
- Better error messages
- Performance optimizations
- Documentation improvements
- Test coverage
- Accessibility features

## License

MIT License - see LICENSE.txt for details
