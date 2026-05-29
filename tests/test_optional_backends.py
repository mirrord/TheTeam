"""Tests for optional-backend agents (EXLAgent, LlamacppAgent).

These backends are gated on optional packages (``exllamav2`` /
``llama-cpp-python``).  The classes must:

1. Always be importable, even without the backend package installed.
2. Raise :class:`ImportError` with installation guidance on construction
   when the backend is unavailable.
3. Not be re-exported from :mod:`pithos.agent` — users must opt in via an
   explicit submodule import.
"""

from __future__ import annotations

import pytest


def test_llamacpp_module_imports_without_backend():
    # The submodule must always be importable.
    import pithos.agent.llamacpp_agent  # noqa: F401


def test_exl_module_imports_without_backend():
    import pithos.agent.exl_agent  # noqa: F401


def test_llamacpp_agent_raises_when_backend_missing(monkeypatch):
    from pithos.agent import llamacpp_agent

    monkeypatch.setattr(llamacpp_agent, "_LLAMACPP_AVAILABLE", False)
    monkeypatch.setattr(llamacpp_agent, "Llama", None)
    with pytest.raises(ImportError, match="llama-cpp-python"):
        llamacpp_agent.LlamacppAgent(default_model="any")


def test_exl_agent_raises_when_backend_missing(monkeypatch):
    from pithos.agent import exl_agent

    monkeypatch.setattr(exl_agent, "_EXL_AVAILABLE", False)
    with pytest.raises(ImportError, match="exllamav2|ExLlamaV2"):
        exl_agent.EXLAgent(default_model="any")


def test_optional_backends_not_in_package_namespace():
    import pithos.agent as agent_pkg

    assert not hasattr(agent_pkg, "EXLAgent")
    assert not hasattr(agent_pkg, "LlamacppAgent")
    assert "EXLAgent" not in agent_pkg.__all__
    assert "LlamacppAgent" not in agent_pkg.__all__
