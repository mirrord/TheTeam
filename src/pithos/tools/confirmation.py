"""Process-wide serialization for interactive tool-call confirmations.

A human can only answer one permission prompt at a time, so *every* tool-call
confirmation in the process is serialized through a single lock held for the
full duration of a prompt (sending it and waiting for the user's reply).

Without this, concurrently executing tool calls — e.g. several commands issued
in one agent step, or tools running on parallel worker threads — could each
raise their own approval prompt at the same time. The user would then see
multiple overlapping prompts with no way to tell which one a reply answers, and
the agent would keep working while approvals were still pending.

Both :class:`pithos.tools.executor.ToolExecutor` and the coding harness's
workspace environment acquire :func:`confirmation_lock` around their
confirmation step, guaranteeing that the whole system pauses on a single
pending approval until it is resolved.
"""

import threading

# Re-entrant so a confirmation path that (directly or indirectly) consults the
# gate again on the same thread cannot deadlock; cross-thread callers still
# block until the holder releases.
_CONFIRMATION_LOCK = threading.RLock()


def confirmation_lock() -> "threading.RLock":
    """Return the process-wide tool-confirmation lock.

    Use as a context manager around the confirmation prompt::

        with confirmation_lock():
            approved = confirm_callback(command)
    """
    return _CONFIRMATION_LOCK
