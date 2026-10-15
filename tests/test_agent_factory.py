"""Tests for the agent provider factory (:mod:`pithos.agent.factory`)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from pithos.agent import (
    Agent,
    OllamaAgent,
    build_agent,
    build_agent_from_config,
    canonical_provider,
    agent_class_for_provider,
)
from pithos.agent.llamacpp_agent import LlamacppAgent
from pithos.agent.exl_agent import EXLAgent


class TestCanonicalProvider:
    def test_default_when_missing(self):
        assert canonical_provider(None) == "ollama"
        assert canonical_provider("") == "ollama"
        assert canonical_provider("   ") == "ollama"

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("ollama", "ollama"),
            ("Ollama", "ollama"),
            ("llamacpp", "llamacpp"),
            ("llama.cpp", "llamacpp"),
            ("llama_cpp", "llamacpp"),
            ("LLAMA-CPP", "llamacpp"),
            ("exl", "exl"),
            ("exl2", "exl"),
            ("exllama", "exl"),
            ("ExLlamaV2", "exl"),
        ],
    )
    def test_aliases(self, raw, expected):
        assert canonical_provider(raw) == expected

    def test_unknown_rejected(self):
        with pytest.raises(ValueError, match="Unknown model provider"):
            canonical_provider("gpt-4")


class TestAgentClassForProvider:
    def test_maps_each_provider(self):
        assert agent_class_for_provider(None) is OllamaAgent
        assert agent_class_for_provider("ollama") is OllamaAgent
        assert agent_class_for_provider("llama.cpp") is LlamacppAgent
        assert agent_class_for_provider("exllamav2") is EXLAgent


class TestBuildAgent:
    def test_default_builds_ollama(self):
        agent = build_agent("some-model")
        assert isinstance(agent, OllamaAgent)
        assert agent.default_model == "some-model"

    def test_ollama_ignores_backend_options(self):
        # Ollama has no backend_options/base_url params; passing them must
        # not raise so callers can forward uniformly.
        agent = build_agent(
            "m", provider="ollama", backend_options={"n_ctx": 1}, base_url="x"
        )
        assert isinstance(agent, OllamaAgent)

    def test_llamacpp_server_mode(self):
        agent = build_agent(
            "qwen3", provider="llamacpp", base_url="http://127.0.0.1:8080"
        )
        assert isinstance(agent, LlamacppAgent)
        assert agent.server_mode is True
        assert agent.base_url == "http://127.0.0.1:8080"

    def test_unknown_provider_rejected(self):
        with pytest.raises(ValueError, match="Unknown model provider"):
            build_agent("m", provider="nope")


class TestBuildAgentFromConfig:
    def test_defaults_to_ollama(self):
        cm = MagicMock()
        agent = build_agent_from_config({"model": "m", "name": "x"}, cm)
        assert isinstance(agent, OllamaAgent)

    def test_dispatches_llamacpp(self):
        cm = MagicMock()
        agent = build_agent_from_config(
            {
                "model": "qwen3",
                "provider": "llama.cpp",
                "base_url": "http://127.0.0.1:8080",
            },
            cm,
        )
        assert isinstance(agent, LlamacppAgent)
        assert agent.base_url == "http://127.0.0.1:8080"


class TestBaseAgentDispatch:
    def test_from_dict_on_base_dispatches(self):
        cm = MagicMock()
        agent = Agent.from_dict(
            {"model": "m", "provider": "ollama", "name": "x"}, cm
        )
        assert isinstance(agent, OllamaAgent)

    def test_from_dict_on_subclass_ignores_provider(self):
        # Calling on a concrete subclass builds that subclass regardless of
        # the provider key (backward compatible).
        cm = MagicMock()
        agent = OllamaAgent.from_dict(
            {"model": "m", "provider": "llamacpp"}, cm
        )
        assert isinstance(agent, OllamaAgent)

    def test_from_config_missing_uses_factory(self):
        cm = MagicMock()
        cm.get_config.return_value = None
        agent = Agent.from_config("bare-model", cm)
        assert isinstance(agent, OllamaAgent)
        assert agent.default_model == "bare-model"


class TestProviderRoundTrip:
    def test_to_dict_includes_provider(self):
        agent = OllamaAgent("m")
        assert agent.to_dict()["provider"] == "ollama"

    def test_llamacpp_to_dict_provider(self):
        agent = build_agent(
            "qwen3", provider="llamacpp", base_url="http://127.0.0.1:8080"
        )
        assert agent.to_dict()["provider"] == "llamacpp"
