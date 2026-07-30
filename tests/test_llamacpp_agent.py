"""Behavioral tests for :class:`LlamacppAgent` with a mocked backend."""

from __future__ import annotations

from typing import Any
import pytest

from pithos.agent import llamacpp_agent
from pithos.agent.llamacpp_agent import LlamacppAgent


class _FakeLlama:
    """Minimal stand-in for ``llama_cpp.Llama`` used in tests.

    ``chunks`` is a list of strings to emit one-per-chunk; the mock then
    appends a final usage-bearing chunk so metrics can be exercised.
    """

    instances: list["_FakeLlama"] = []

    def __init__(self, model_path: str, **kwargs: Any) -> None:
        self.model_path = model_path
        self.kwargs = kwargs
        self.calls: list[dict[str, Any]] = []
        # Default behaviour: single "hello world" response.
        self.chunks: list[str] = ["hello", " world"]
        _FakeLlama.instances.append(self)

    def create_chat_completion(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        for tok in self.chunks:
            yield {"choices": [{"delta": {"content": tok}}]}
        yield {
            "choices": [{"delta": {}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": len(self.chunks)},
        }


@pytest.fixture
def fake_llama(monkeypatch):
    _FakeLlama.instances.clear()
    monkeypatch.setattr(llamacpp_agent, "Llama", _FakeLlama)
    monkeypatch.setattr(llamacpp_agent, "_LLAMACPP_AVAILABLE", True)
    return _FakeLlama


# ---------------------------------------------------------------------
# Construction & lazy loading
# ---------------------------------------------------------------------


def test_construction_succeeds_with_backend_available(fake_llama):
    agent = LlamacppAgent(default_model="/models/x.gguf")
    assert agent.default_model == "/models/x.gguf"
    # Lazy: model not loaded until first stream.
    assert agent._llama is None
    assert fake_llama.instances == []


def test_backend_options_passed_through(fake_llama):
    agent = LlamacppAgent(
        default_model="/models/x.gguf",
        backend_options={"n_ctx": 8192, "n_gpu_layers": 99},
    )
    list(agent.stream("Hi"))
    inst = fake_llama.instances[-1]
    assert inst.kwargs["n_ctx"] == 8192
    assert inst.kwargs["n_gpu_layers"] == 99
    assert inst.kwargs["verbose"] is False  # default applied


def test_default_n_ctx_applied(fake_llama):
    agent = LlamacppAgent(default_model="/models/x.gguf")
    list(agent.stream("Hi"))
    assert fake_llama.instances[-1].kwargs["n_ctx"] == 4096


# ---------------------------------------------------------------------
# Streaming behaviour
# ---------------------------------------------------------------------


def test_stream_yields_tokens_and_commits_response(fake_llama):
    agent = LlamacppAgent(default_model="/models/x.gguf")
    out = list(agent.stream("Hi"))
    # First two tokens from the canned chunks plus a trailing "" from the
    # usage chunk's empty delta.
    assert out[:2] == ["hello", " world"]
    history = agent.contexts["default"].message_history
    assert history[-1]["role"] == "assistant"
    assert history[-1]["content"] == "hello world"


def test_stream_passes_temperature_and_max_tokens(fake_llama):
    agent = LlamacppAgent(default_model="/models/x.gguf", temperature=0.3)
    agent.max_tokens = 64
    list(agent.stream("Hi"))
    call = fake_llama.instances[-1].calls[-1]
    assert call["temperature"] == 0.3
    assert call["max_tokens"] == 64
    assert call["stream"] is True


def test_stream_omits_max_tokens_when_unset(fake_llama):
    agent = LlamacppAgent(default_model="/models/x.gguf")
    list(agent.stream("Hi"))
    call = fake_llama.instances[-1].calls[-1]
    assert "max_tokens" not in call


def test_lazy_load_caches_instance(fake_llama):
    agent = LlamacppAgent(default_model="/models/x.gguf")
    list(agent.stream("first"))
    list(agent.stream("second"))
    # Same model path → only one Llama instance constructed.
    assert len(fake_llama.instances) == 1


def test_changing_model_triggers_reload(fake_llama):
    agent = LlamacppAgent(default_model="/models/x.gguf")
    list(agent.stream("first"))
    list(agent.stream("second", model="/models/y.gguf"))
    assert len(fake_llama.instances) == 2
    assert fake_llama.instances[1].model_path == "/models/y.gguf"


def test_stream_error_rolls_back_user_message(fake_llama):
    agent = LlamacppAgent(default_model="/models/x.gguf")

    def boom(**kw):  # noqa: ANN001
        raise RuntimeError("backend kaboom")

    # Force a load and then break the call.
    list(agent.stream("warmup"))
    fake_llama.instances[-1].create_chat_completion = boom  # type: ignore[assignment]

    history_len_before = len(agent.contexts["default"].message_history)
    with pytest.raises(RuntimeError, match="llama.cpp"):
        list(agent.stream("Hi"))
    assert len(agent.contexts["default"].message_history) == history_len_before


# ---------------------------------------------------------------------
# Metrics integration
# ---------------------------------------------------------------------


def test_metrics_records_token_usage(fake_llama):
    from pithos.metrics import MetricsCollector

    agent = LlamacppAgent(default_model="/models/x.gguf")
    metrics = MetricsCollector()
    agent.attach_metrics(metrics)
    list(agent.stream("Hi"))

    # Collector should have at least one token-usage entry.
    assert "/models/x.gguf" in metrics._token_usage
    tm = metrics._token_usage["/models/x.gguf"]
    assert tm.completion_tokens >= 2
    assert tm.prompt_tokens >= 5


# ---------------------------------------------------------------------
# Mid-stream tool execution
# ---------------------------------------------------------------------


def test_mid_stream_tool_execution(fake_llama, tmp_path, monkeypatch):
    from pithos.config_manager import ConfigManager

    cm = ConfigManager(config_dir=str(tmp_path))
    agent = LlamacppAgent(default_model="/models/x.gguf")
    agent.enable_tools(cm)

    # First call: emit a bracket tool invocation; second (continuation):
    # produce a final answer.
    fake_llama_cls = fake_llama
    call_idx = {"n": 0}

    def chunks_for_call(self_inst, **kw):  # noqa: ANN001
        call_idx["n"] += 1
        if call_idx["n"] == 1:
            yield {
                "choices": [
                    {"delta": {"content": "Let me check. [RUN]echo hi[/RUN]\n"}}
                ]
            }
            yield {
                "choices": [{"delta": {}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 6},
            }
        else:
            yield {"choices": [{"delta": {"content": "Done."}}]}
            yield {
                "choices": [{"delta": {}}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1},
            }

    # Patch instance method on every constructed FakeLlama.
    orig_init = fake_llama_cls.__init__

    def patched_init(self, model_path, **kwargs):  # noqa: ANN001
        orig_init(self, model_path, **kwargs)
        self.create_chat_completion = lambda **kw: chunks_for_call(self, **kw)

    monkeypatch.setattr(fake_llama_cls, "__init__", patched_init)

    out = "".join(agent.stream("Question"))
    # Final assembled output spans both calls.
    assert "Done." in out
    # System message containing the tool result must be in context.
    roles = [m["role"] for m in agent.contexts["default"].message_history]
    assert "system" in roles
