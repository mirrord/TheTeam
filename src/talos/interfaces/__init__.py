"""Talos interface implementations.

Voice and Telegram interfaces are imported lazily inside :mod:`talos.__main__`
to avoid pulling in heavy optional dependencies (whisper, bark, telegram)
when only the shell interface is used.
"""

from .shell import ShellInterface
from .system_commands import (
    SYSTEM_RESET_COMMAND,
    SYSTEM_RESET_RESPONSE,
    handle_system_command,
    is_system_command,
)

__all__ = [
    "ShellInterface",
    "SYSTEM_RESET_COMMAND",
    "SYSTEM_RESET_RESPONSE",
    "handle_system_command",
    "is_system_command",
]
