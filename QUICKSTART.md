# Quick Start Guide - TheTeam Web Interface

Get up and running with TheTeam's web interface in minutes!

## Prerequisites

- Python 3.8 or higher
- Node.js 18 or higher  
- Ollama installed and running
- Git (for cloning)

## Step 1: Clone and Setup Backend

```bash
# Clone the repository
git clone https://github.com/Dane Howard/theteam.git
cd theteam

# Create and activate virtual environment
# Windows (PowerShell):
python -m venv .venv
.venv\Scripts\Activate.ps1

# Linux/macOS:
python -m venv .venv
source .venv/bin/activate

# Install Python dependencies
pip install -e .
```

## Step 2: Setup Frontend

```bash
cd frontend
npm install
```

## Step 3: Run the Application

### Option A: WebGUI Mode (Easiest - Recommended)

Start both backend and frontend with a single command:

```bash
# From project root (with virtual environment activated)
theteam webgui
```

This will:
- ✅ Start the backend server at `http://localhost:5000`
- ✅ Start the frontend dev server at `http://localhost:3000`
- ✅ Display live status updates (auto-refreshing every 5 seconds)
- ✅ Handle Ctrl+C gracefully to shut down both processes

Open your browser to `http://localhost:3000` when you see "Frontend URL" in the output.

**Troubleshooting:** If something goes wrong, enable debug mode for verbose error details:
```bash
theteam webgui --debug
```

**Notes:** 
- The status display refreshes in place to keep your terminal clean (debug mode preserves all output)
- Hit Ctrl+C once to initiate graceful shutdown, or twice to force stop

### Option B: Production Mode

```bash
# Build frontend
cd frontend
npm run build

# Start the server (from project root)
cd ..
theteam
```

Open your browser to `http://localhost:5000`

### Option C: Development Mode (Manual Control)

**Terminal 1 - Backend:**
```bash
# From project root
theteam server --debug
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```

Open your browser to `http://localhost:3000`

## Step 4: Explore the Interface

### Create Your First Agent

1. Click "Agents" in the sidebar
2. Click "New Agent"
3. Configure:
   - ID: `my-agent`
   - Name: `My First Agent`
   - Model: `llama3.2:latest`
   - (Optional) Add a system prompt
4. Click "Save"
5. Test with a sample prompt

### Have a Conversation

1. Click "Chat" in the sidebar
2. Click "New Chat"
3. Select your agent from the dropdown
4. Type a message and press Enter
5. Watch the real-time response!

### Create a Simple Flowchart

1. Click "Flowcharts" in the sidebar
2. Click "New Flowchart"
3. Drag nodes onto the canvas
4. Connect nodes by dragging between connection points
5. Configure node prompts in the inspector
6. Click "Save"
7. Click "Execute" to run the workflow

## Troubleshooting

### "Connection failed" error

**Solution:** Ensure the backend server is running and Ollama is installed.

```bash
# Check if Ollama is running
ollama list

# If not installed, visit: https://ollama.ai
```

### Port 5000 already in use

**Solution:** Use a different port:

```bash
theteam --port 8000
```

### npm install fails

**Solution:** Clear cache and retry:

```bash
rm -rf node_modules package-lock.json
npm cache clean --force
npm install
```

### WebSocket connection issues

**Solution:** 
- Check browser console for errors
- Verify backend is running on correct port
- Try disabling VPN/proxy temporarily

## Next Steps

- Read [WEB_INTERFACE.md](docs/WEB_INTERFACE.md) for detailed documentation
- Check [frontend/README.md](frontend/README.md) for frontend-specific docs
- Explore example configurations in `configs/`
- Join the community discussions

## Common Commands Reference

```bash
# Backend
theteam                          # Start server (default: localhost:5000)
theteam --host 0.0.0.0          # Allow external connections
theteam --port 8000             # Use custom port
theteam --debug                 # Enable debug mode

# Frontend Development
npm run dev                      # Dev server with hot-reload
npm run build                    # Build for production
npm run lint                     # Check for linting errors

# Testing
pytest                           # Run backend tests
pytest -v                        # Verbose output
pytest --cov                     # With coverage
```

## Getting Help

- Check the [documentation](docs/)
- Report issues on [GitHub](https://github.com/Dane Howard/theteam/issues)
- Read the troubleshooting section above

Happy coordinating! 🚀
