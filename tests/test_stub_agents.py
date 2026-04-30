"""Tests for stub agent backends (EXLAgent, LlamacppAgent).

The stubs must:
1. Raise ``NotImplementedError`` immediately on construction so callers fail
   fast instead of receiving a half-working agent.
2. Not be re-exported from ``pithos.agent`` — they require an explicit
   submodule import to opt in.
"""

from __future__ import annotations

import pytest


def test_exl_agent_raises_on_construction():
    from pithos.agent.exl_agent import EXLAgent

    with pytest.raises(NotImplementedError, match="EXLAgent"):
        EXLAgent(model="any")


def test_llamacpp_agent_raises_on_construction():
    from pithos.agent.llamacpp_agent import LlamacppAgent

    with pytest.raises(NotImplementedError, match="LlamacppAgent"):
        LlamacppAgent(model="any")


def test_stubs_not_in_package_namespace():
    import pithos.agent as agent_pkg

    assert not hasattr(agent_pkg, "EXLAgent")
    assert not hasattr(agent_pkg, "LlamacppAgent")
    assert "EXLAgent" not in agent_pkg.__all__
    assert "LlamacppAgent" not in agent_pkg.__all__
