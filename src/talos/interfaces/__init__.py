"""Talos interface implementations.

Voice and Telegram interfaces are imported lazily inside :mod:`talos.__main__`
to avoid pulling in heavy optional dependencies (whisper, bark, telegram)
when only the shell interface is used.
"""

from .shell import ShellInterface

__all__ = ["ShellInterface"]
