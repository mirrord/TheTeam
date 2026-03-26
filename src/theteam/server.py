"""
TheTeam Web Server

Flask application with SocketIO for real-time web interface.
Provides API endpoints for agent management, flowchart editing, and chat interactions.
"""

import os
import sys
import logging
import subprocess
import signal
import threading
import time
import shutil
import platform
from pathlib import Path
from typing import Optional
from flask import Flask, send_from_directory
from flask_socketio import SocketIO
from flask_cors import CORS


class VerboseSocketIOFilter(logging.Filter):
    """Filter out verbose engineio/socketio messages from logs."""

    def filter(self, record):
        message = record.getMessage()
        # Filter out protocol-level messages that clutter logs
        verbose_patterns = [
            'packet MESSAGE data 2["ping"',
            'packet MESSAGE data 2["pong"',
            'received event "ping"',
            'emitting event "pong"',
            "received packet PING",
            "Received packet PING",
            "Sending packet PONG",
            "Received packet PONG",
            "Sending packet PING",
            "Sending packet OPEN",
            "Received packet MESSAGE data 0",
            'emitting event "connection_established"',
            "Sending packet MESSAGE data 2[",
            "Sending packet MESSAGE data 0{",
        ]
        for pattern in verbose_patterns:
            if pattern in message:
                return False
        return True


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add filter to engineio and socketio loggers to suppress verbose messages
verbose_filter = VerboseSocketIOFilter()
logging.getLogger("engineio").addFilter(verbose_filter)
logging.getLogger("engineio.server").addFilter(verbose_filter)
logging.getLogger("socketio").addFilter(verbose_filter)
logging.getLogger("socketio.server").addFilter(verbose_filter)


def create_app(config=None):
    """Create and configure the Flask application."""
    app = Flask(__name__, static_folder="static", static_url_path="")

    # Configuration
    app.config["SECRET_KEY"] = os.environ.get(
        "SECRET_KEY", "dev-secret-key-change-in-production"
    )
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max file size

    if config:
        app.config.update(config)

    # Enable CORS for development
    CORS(app, resources={r"/*": {"origins": "*"}})

    # Create custom loggers with ping/pong filter
    socketio_logger = logging.getLogger("socketio")
    engineio_logger = logging.getLogger("engineio")

    # Initialize SocketIO with custom loggers
    socketio = SocketIO(
        app,
        cors_allowed_origins="*",
        async_mode="eventlet",
        logger=socketio_logger,
        engineio_logger=engineio_logger,
        ping_timeout=60,
        ping_interval=25,
    )

    # Register blueprints
    from theteam.api import agents, flowcharts, chat, system, database, tools

    app.register_blueprint(agents.bp)
    app.register_blueprint(flowcharts.bp)
    app.register_blueprint(chat.bp)
    app.register_blueprint(system.bp)
    app.register_blueprint(database.bp)
    app.register_blueprint(tools.bp)

    # Register SocketIO handlers
    from theteam.api import socketio_handlers

    socketio_handlers.register_handlers(socketio)

    # Serve React app
    @app.route("/")
    def index():
        if app.static_folder:
            static_dir = Path(app.static_folder)
            if (static_dir / "index.html").exists():
                return send_from_directory(app.static_folder, "index.html")
        return {"message": "TheTeam API Server", "status": "running"}, 200

    @app.route("/<path:path>")
    def serve_static(path):
        if app.static_folder:
            static_dir = Path(app.static_folder)
            if (static_dir / path).exists():
                return send_from_directory(app.static_folder, path)
            # Serve index.html for client-side routing
            if (static_dir / "index.html").exists():
                return send_from_directory(app.static_folder, "index.html")
        return {"error": "Not found"}, 404

    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        return {"error": "Not found"}, 404

    @app.errorhandler(500)
    def internal_error(error):
        logger.error(f"Internal error: {error}")
        return {"error": "Internal server error"}, 500

    return app, socketio


def start_webgui(
    host: str = "127.0.0.1", port: int = 5000, backend_debug: bool = False
):
    """
    Start both backend and frontend processes with live status updates.
    Handles Ctrl+C gracefully to shut down both processes.

    Args:
        host: Host to bind backend to
        port: Port to bind backend to
        backend_debug: Enable backend debug mode
    """
    # Set UTF-8 encoding for proper character display
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    # Enable ANSI escape sequences on Windows for terminal cursor control
    ansi_supported = False
    if platform.system() == "Windows":
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            # Get handle for stdout
            handle = kernel32.GetStdHandle(-11)
            # Get current console mode
            mode = ctypes.c_uint32()
            kernel32.GetConsoleMode(handle, ctypes.byref(mode))
            # Enable virtual terminal processing (0x0004)
            new_mode = mode.value | 0x0004
            kernel32.SetConsoleMode(handle, new_mode)
            ansi_supported = True
        except Exception:
            # If we can't enable ANSI codes, status updates will just accumulate
            ansi_supported = False

    backend_process: Optional[subprocess.Popen] = None
    frontend_process: Optional[subprocess.Popen] = None
    shutting_down = threading.Event()

    # Store ANSI support flag for later use (always refresh if ANSI is supported)
    use_ansi_refresh = ansi_supported

    if backend_debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)

    def signal_handler(signum, frame):
        """Handle Ctrl+C gracefully."""
        if shutting_down.is_set():
            # Force exit on second Ctrl+C
            print("\n\n⚠️  Force stopping processes...")
            sys.exit(1)

        shutting_down.set()
        print("\n\n🛑 Shutting down processes...")

        # Terminate both processes
        if backend_process and backend_process.poll() is None:
            print("   Stopping backend server...")
            try:
                backend_process.terminate()
                backend_process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                backend_process.kill()

        if frontend_process and frontend_process.poll() is None:
            print("   Stopping frontend dev server...")
            try:
                frontend_process.terminate()
                frontend_process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                frontend_process.kill()

        print("✅ All processes stopped\n")
        sys.exit(0)

    # Register signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, signal_handler)

    # Find project root and frontend directory
    current_file = Path(__file__)
    project_root = current_file.parent.parent.parent
    frontend_dir = project_root / "frontend"

    if not frontend_dir.exists():
        print(f"❌ Error: Frontend directory not found at {frontend_dir}")
        sys.exit(1)

    # Check if node_modules exists
    if not (frontend_dir / "node_modules").exists():
        print("📦 Installing frontend dependencies...")
        try:
            # On Windows, npm is a .cmd file, so we need shell=True or the full path
            is_windows = platform.system() == "Windows"
            npm_cmd = "npm" if is_windows else shutil.which("npm") or "npm"

            install_process = subprocess.run(
                [npm_cmd, "install"] if not is_windows else "npm install",
                cwd=str(frontend_dir),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=120,
                shell=is_windows,
            )
            if install_process.returncode != 0:
                print(
                    f"❌ Failed to install frontend dependencies:\n{install_process.stderr}"
                )
                if backend_debug:
                    logger.debug("Full npm install output:")
                    logger.debug("STDOUT: %s", install_process.stdout)
                    logger.debug("STDERR: %s", install_process.stderr)
                    logger.debug("Return code: %s", install_process.returncode)
                sys.exit(1)
            print("✅ Frontend dependencies installed\n")
        except subprocess.TimeoutExpired:
            print("❌ Frontend dependency installation timed out")
            if backend_debug:
                import traceback

                logger.debug("Timeout details:", exc_info=True)
            sys.exit(1)
        except FileNotFoundError:
            print("❌ npm not found. Please install Node.js and npm.")
            if backend_debug:
                logger.debug("FileNotFoundError details:", exc_info=True)
            sys.exit(1)
        except Exception as e:
            print(f"❌ Unexpected error during npm install: {e}")
            if backend_debug:
                logger.debug("Full traceback:", exc_info=True)
            sys.exit(1)

    print("🚀 Starting TheTeam WebGUI\n")
    print("=" * 60)

    # Start backend server
    print(f"🔧 Starting backend server on http://{host}:{port}")
    if backend_debug:
        logger.debug(
            "Backend command: %s",
            " ".join(
                [
                    sys.executable,
                    "-m",
                    "theteam.server",
                    "--host",
                    host,
                    "--port",
                    str(port),
                    "--debug",
                ]
            ),
        )
    try:
        backend_cmd = [
            sys.executable,
            "-m",
            "theteam.server",
            "--host",
            host,
            "--port",
            str(port),
        ]
        if backend_debug:
            backend_cmd.append("--debug")

        backend_process = subprocess.Popen(
            backend_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        print("✅ Backend server started")
        if backend_debug:
            logger.debug("Backend PID: %s", backend_process.pid)
    except Exception as e:
        print(f"❌ Failed to start backend: {e}")
        if backend_debug:
            logger.debug("Full traceback:", exc_info=True)
        sys.exit(1)

    # Give backend a moment to start
    time.sleep(1)

    # Start frontend dev server
    print("🎨 Starting frontend dev server")
    if backend_debug:
        logger.debug("Frontend command: npm run dev (cwd: %s)", frontend_dir)
    try:
        # On Windows, npm is a .cmd file, so we need shell=True
        is_windows = platform.system() == "Windows"

        if is_windows:
            frontend_process = subprocess.Popen(
                "npm run dev",
                cwd=str(frontend_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                shell=True,
            )
        else:
            npm_cmd = shutil.which("npm") or "npm"
            frontend_process = subprocess.Popen(
                [npm_cmd, "run", "dev"],
                cwd=str(frontend_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
        print("✅ Frontend dev server started")
        if backend_debug:
            logger.debug("Frontend PID: %s", frontend_process.pid)
    except Exception as e:
        print(f"❌ Failed to start frontend: {e}")
        if backend_debug:
            logger.debug("Full traceback:", exc_info=True)
        if backend_process:
            backend_process.terminate()
        sys.exit(1)

    print("=" * 60)
    print("\n📊 Live Status Monitor")
    print("   Press Ctrl+C to stop both processes\n")
    print("-" * 60)

    # Monitor both processes
    backend_ready = False
    frontend_ready = False
    last_status_update = time.time()

    def read_backend_output():
        """Read and process backend output."""
        nonlocal backend_ready
        if not backend_process or not backend_process.stdout:
            return

        try:
            for line in iter(backend_process.stdout.readline, ""):
                if not line or shutting_down.is_set():
                    break
                line = line.strip()
                if line:
                    if "Starting TheTeam server" in line or "Running on" in line:
                        backend_ready = True
                    # In debug mode, print all output; otherwise only important messages
                    if backend_debug:
                        print(f"   [Backend]  {line}")
                    elif any(
                        keyword in line
                        for keyword in ["ERROR", "WARNING", "Starting", "Werkzeug"]
                    ):
                        print(f"   [Backend]  {line}")
        except Exception as e:
            if backend_debug:
                logger.debug("Error reading backend output: %s", e, exc_info=True)

    def read_frontend_output():
        """Read and process frontend output."""
        nonlocal frontend_ready
        if not frontend_process or not frontend_process.stdout:
            return

        try:
            for line in iter(frontend_process.stdout.readline, ""):
                if not line or shutting_down.is_set():
                    break
                line = line.strip()
                if line:
                    if "Local:" in line or "ready in" in line:
                        frontend_ready = True
                    # In debug mode, print all output; otherwise only important messages
                    if backend_debug:
                        print(f"   [Frontend] {line}")
                    elif any(
                        keyword in line
                        for keyword in ["Local:", "error", "Error", "ready in"]
                    ):
                        print(f"   [Frontend] {line}")
        except Exception as e:
            if backend_debug:
                logger.debug("Error reading frontend output: %s", e, exc_info=True)

    # Start output monitoring threads
    backend_thread = threading.Thread(target=read_backend_output, daemon=True)
    frontend_thread = threading.Thread(target=read_frontend_output, daemon=True)

    backend_thread.start()
    frontend_thread.start()

    # Wait for processes to be ready
    print("\n⏳ Waiting for services to start...\n")

    # Track if we've shown the first status update (to avoid clearing initial output)
    first_status_shown = False
    status_line_count = 0

    # Main monitoring loop
    try:
        while not shutting_down.is_set():
            # Check if processes are still running
            backend_status = (
                "Running ✅" if backend_process.poll() is None else "Stopped ❌"
            )
            frontend_status = (
                "Running ✅" if frontend_process.poll() is None else "Stopped ❌"
            )

            # Print status update every 5 seconds
            current_time = time.time()
            if current_time - last_status_update > 5:
                # Clear previous status update
                if first_status_shown and use_ansi_refresh:
                    # Move cursor up to the beginning of the status block
                    # Use \033[F (move to beginning of previous line) which is more reliable
                    for _ in range(status_line_count):
                        sys.stdout.write("\033[F")
                    # Clear from cursor to end of screen
                    sys.stdout.write("\033[J")
                    sys.stdout.flush()

                # Print new status update
                status_lines = []
                status_lines.append("📊 Status Update:")
                status_lines.append(f"   Backend:  {backend_status}")
                status_lines.append(f"   Frontend: {frontend_status}")
                if backend_ready and frontend_ready:
                    status_lines.append("")
                    status_lines.append("🌐 Frontend URL: http://localhost:3000")
                    status_lines.append(f"🔌 Backend URL:  http://{host}:{port}")
                status_lines.append("")

                for line in status_lines:
                    print(line)

                # Track how many lines we printed for next clear operation
                status_line_count = len(status_lines)
                first_status_shown = True
                last_status_update = current_time

            # Exit if either process died unexpectedly
            if backend_process.poll() is not None:
                return_code = backend_process.poll()
                print(
                    f"\n❌ Backend process stopped unexpectedly (exit code: {return_code})"
                )
                if backend_debug:
                    logger.debug(
                        "Backend process details - PID: %s, exit code: %s",
                        backend_process.pid,
                        return_code,
                    )
                    try:
                        remaining_output = backend_process.stdout.read()
                        if remaining_output:
                            logger.debug("Remaining output: %s", remaining_output)
                    except Exception as e:
                        logger.debug("Could not read remaining output: %s", e)
                break
            if frontend_process.poll() is not None:
                return_code = frontend_process.poll()
                print(
                    f"\n❌ Frontend process stopped unexpectedly (exit code: {return_code})"
                )
                if backend_debug:
                    logger.debug(
                        "Frontend process details - PID: %s, exit code: %s",
                        frontend_process.pid,
                        return_code,
                    )
                    try:
                        remaining_output = frontend_process.stdout.read()
                        if remaining_output:
                            logger.debug("Remaining output: %s", remaining_output)
                    except Exception as e:
                        logger.debug("Could not read remaining output: %s", e)
                break

            time.sleep(0.5)

    except KeyboardInterrupt:
        pass

    finally:
        # Cleanup
        if not shutting_down.is_set():
            shutting_down.set()
            print("\n\n🛑 Cleaning up...")

            if backend_process and backend_process.poll() is None:
                backend_process.terminate()
                try:
                    backend_process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    backend_process.kill()

            if frontend_process and frontend_process.poll() is None:
                frontend_process.terminate()
                try:
                    frontend_process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    frontend_process.kill()

            print("✅ Cleanup complete\n")


def main():
    """Run the web server."""
    import argparse

    parser = argparse.ArgumentParser(description="TheTeam Web Server")

    # Add legacy arguments at main parser level for backward compatibility
    parser.add_argument(
        "--host", default="127.0.0.1", help="Host to bind to (legacy syntax)"
    )
    parser.add_argument(
        "--port", type=int, default=5000, help="Port to bind to (legacy syntax)"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug mode (legacy syntax)"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Server command (default behavior)
    server_parser = subparsers.add_parser(
        "server", help="Run backend server only", aliases=["serve"]
    )
    server_parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    server_parser.add_argument("--port", type=int, default=5000, help="Port to bind to")
    server_parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    # WebGUI command (starts both backend and frontend)
    webgui_parser = subparsers.add_parser(
        "webgui", help="Start both backend and frontend with live status monitoring"
    )
    webgui_parser.add_argument("--host", default="127.0.0.1", help="Backend host")
    webgui_parser.add_argument("--port", type=int, default=5000, help="Backend port")
    webgui_parser.add_argument(
        "--debug", action="store_true", help="Enable backend debug mode"
    )

    args = parser.parse_args()

    # Handle webgui command
    if args.command == "webgui":
        start_webgui(host=args.host, port=args.port, backend_debug=args.debug)
        return

    # Handle server command or default (no subcommand / legacy syntax)
    if args.command in ["server", "serve"] or args.command is None:
        # For all cases, use the parsed arguments (supports both legacy and subcommand syntax)
        host = args.host
        port = args.port
        debug = args.debug

        app, socketio = create_app()

        logger.info(f"Starting TheTeam server on http://{host}:{port}")

        try:
            socketio.run(
                app,
                host=host,
                port=port,
                debug=debug,
                use_reloader=debug,
            )
        except KeyboardInterrupt:
            logger.info("Server stopped by user")
        except Exception as e:
            logger.error(f"Server error: {e}", exc_info=True)
            raise
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
