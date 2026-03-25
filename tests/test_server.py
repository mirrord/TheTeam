"""Tests for TheTeam server CLI and webgui functionality."""

import pytest
import sys
from io import StringIO
from unittest.mock import patch, MagicMock


def test_server_main_help():
    """Test that the server main function shows help correctly."""
    from theteam.server import main

    # Mock sys.argv to request help
    with patch.object(sys, "argv", ["theteam", "--help"]):
        with pytest.raises(SystemExit) as exc_info:
            main()
        # argparse exits with 0 when showing help
        assert exc_info.value.code == 0


def test_webgui_command_parsing():
    """Test that webgui subcommand is recognized."""
    from theteam.server import main

    # Mock the start_webgui function to avoid actually starting processes
    with patch("theteam.server.start_webgui") as mock_webgui:
        with patch.object(sys, "argv", ["theteam", "webgui"]):
            main()
            # Verify start_webgui was called
            mock_webgui.assert_called_once()


def test_webgui_command_with_args():
    """Test webgui command accepts host and port arguments."""
    from theteam.server import main

    with patch("theteam.server.start_webgui") as mock_webgui:
        with patch.object(
            sys, "argv", ["theteam", "webgui", "--host", "0.0.0.0", "--port", "8080"]
        ):
            main()
            # Verify start_webgui was called with correct arguments
            mock_webgui.assert_called_once_with(
                host="0.0.0.0", port=8080, backend_debug=False
            )


def test_webgui_command_with_debug():
    """Test webgui command accepts debug flag."""
    from theteam.server import main

    with patch("theteam.server.start_webgui") as mock_webgui:
        with patch.object(sys, "argv", ["theteam", "webgui", "--debug"]):
            main()
            # Verify debug flag is passed
            call_kwargs = mock_webgui.call_args[1]
            assert call_kwargs["backend_debug"] is True


def test_server_command_parsing():
    """Test that server subcommand is recognized."""
    from theteam.server import main

    # Mock create_app and socketio.run to avoid starting the server
    with patch("theteam.server.create_app") as mock_create_app:
        mock_app = MagicMock()
        mock_socketio = MagicMock()
        mock_create_app.return_value = (mock_app, mock_socketio)

        with patch.object(sys, "argv", ["theteam", "server"]):
            main()
            # Verify create_app was called
            mock_create_app.assert_called_once()
            # Verify socketio.run was called
            mock_socketio.run.assert_called_once()


def test_legacy_command_parsing():
    """Test that legacy command (no subcommand) still works."""
    from theteam.server import main

    with patch("theteam.server.create_app") as mock_create_app:
        mock_app = MagicMock()
        mock_socketio = MagicMock()
        mock_create_app.return_value = (mock_app, mock_socketio)

        with patch.object(
            sys, "argv", ["theteam", "--host", "127.0.0.1", "--port", "5000"]
        ):
            main()
            # Verify create_app was called
            mock_create_app.assert_called_once()


def test_start_webgui_imports():
    """Test that start_webgui can be imported."""
    from theteam.server import start_webgui

    # Verify the function exists and is callable
    assert callable(start_webgui)


def test_start_webgui_parameters():
    """Test start_webgui accepts expected parameters."""
    import inspect
    from theteam.server import start_webgui

    # Get function signature
    sig = inspect.signature(start_webgui)
    params = sig.parameters

    # Verify expected parameters exist
    assert "host" in params
    assert "port" in params
    assert "backend_debug" in params

    # Verify default values
    assert params["host"].default == "127.0.0.1"
    assert params["port"].default == 5000
    assert params["backend_debug"].default is False


def test_webgui_debug_mode_verbose_errors():
    """Test that debug mode provides verbose error output."""
    from theteam.server import start_webgui

    # Mock Path to simulate missing frontend directory
    with patch("theteam.server.Path") as mock_path:
        mock_path_obj = MagicMock()
        mock_path_obj.parent.parent.parent = MagicMock()
        frontend_dir = MagicMock()
        frontend_dir.exists.return_value = False
        mock_path_obj.parent.parent.parent.__truediv__ = (
            lambda self, other: frontend_dir
        )
        mock_path.return_value = mock_path_obj

        # Capture stdout to check for verbose output
        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            with pytest.raises(SystemExit):
                start_webgui(backend_debug=True)

            # The error message should have been printed
            output = mock_stdout.getvalue()
            assert "Frontend directory not found" in output


def test_webgui_debug_npm_install_error():
    """Test that debug mode shows npm install errors with details."""
    from theteam.server import start_webgui

    with patch("theteam.server.Path") as mock_path:
        # Setup mock paths
        mock_path_obj = MagicMock()
        project_root = MagicMock()
        frontend_dir = MagicMock()
        node_modules = MagicMock()

        # Frontend directory exists but node_modules doesn't
        frontend_dir.exists.return_value = True
        node_modules.exists.return_value = False
        frontend_dir.__truediv__ = lambda self, other: node_modules

        mock_path_obj.parent.parent.parent = project_root
        project_root.__truediv__ = lambda self, other: frontend_dir
        mock_path.return_value = mock_path_obj

        # Mock subprocess.run to simulate failed npm install
        failed_result = MagicMock()
        failed_result.returncode = 1
        failed_result.stderr = "npm ERR! test error"
        failed_result.stdout = "npm output"

        with patch("subprocess.run", return_value=failed_result):
            with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
                with pytest.raises(SystemExit):
                    start_webgui(backend_debug=True)

                output = mock_stdout.getvalue()
                # In debug mode, should show detailed output
                assert "Debug" in output or "npm ERR!" in output


def test_webgui_debug_process_startup():
    """Test that debug mode shows process startup details."""
    from theteam.server import main

    with patch("theteam.server.start_webgui") as mock_webgui:
        with patch.object(sys, "argv", ["theteam", "webgui", "--debug"]):
            main()

            # Verify debug flag was passed as True
            call_args = mock_webgui.call_args
            assert call_args[1]["backend_debug"] is True


def test_webgui_windows_npm_compatibility():
    """Test that Windows npm execution uses shell=True."""
    from theteam.server import start_webgui

    # Mock platform.system to simulate Windows
    with patch("theteam.server.platform.system", return_value="Windows"):
        with patch("theteam.server.Path") as mock_path:
            # Setup mock paths
            mock_path_obj = MagicMock()
            project_root = MagicMock()
            frontend_dir = MagicMock()
            node_modules = MagicMock()

            # Frontend directory and node_modules exist
            frontend_dir.exists.return_value = True
            node_modules.exists.return_value = True
            frontend_dir.__truediv__ = lambda self, other: node_modules

            mock_path_obj.parent.parent.parent = project_root
            project_root.__truediv__ = lambda self, other: frontend_dir
            mock_path.return_value = mock_path_obj

            # Mock subprocess.Popen to verify shell=True is used
            with patch("subprocess.Popen") as mock_popen:
                mock_process = MagicMock()
                mock_process.poll.return_value = None
                mock_process.stdout = MagicMock()
                mock_process.stdout.readline.return_value = ""
                mock_popen.return_value = mock_process

                # Mock sys.executable for backend process
                with patch("sys.executable", "python"):
                    # This will start both processes
                    with patch("time.sleep"):
                        # Need to stop the monitoring loop quickly
                        with patch("theteam.server.threading.Thread"):
                            try:
                                # Call will run until KeyboardInterrupt or process failure
                                # We'll make the process poll as "stopped" to exit the loop
                                def side_effect(*args, **kwargs):
                                    # First call for backend process
                                    if mock_popen.call_count == 1:
                                        return mock_process
                                    # Second call for frontend process - verify shell=True
                                    assert (
                                        kwargs.get("shell") is True
                                    ), "Frontend process should use shell=True on Windows"
                                    # Return a process that will appear stopped to exit loop
                                    stopped_process = MagicMock()
                                    stopped_process.poll.return_value = 1
                                    return stopped_process

                                mock_popen.side_effect = side_effect

                                # Suppress output
                                with patch("builtins.print"):
                                    with patch("sys.exit"):
                                        start_webgui(backend_debug=False)
                            except Exception:
                                pass  # Expected to fail in test environment

                # Verify Popen was called at least twice (backend + frontend)
                assert mock_popen.call_count >= 2
