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

    # First call: emit a JSON tool invocation; second (continuation):
    # produce a final answer.
    fake_llama_cls = fake_llama
    call_idx = {"n": 0}

    def chunks_for_call(self_inst, **kw):  # noqa: ANN001
        call_idx["n"] += 1
        if call_idx["n"] == 1:
            yield {
                "choices": [
                    {
                        "delta": {
                            "content": (
                                "Let me check.\n```tool_call\n"
                                '{"name": "run", "arguments": {"command": "echo hi"}}'
                                "\n```\n"
                            )
                        }
                    }
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


# =====================================================================
# Server (HTTP, OpenAI-compatible) mode
# =====================================================================

import json as _json


class _FakeHTTPResponse:
    """Minimal stand-in for the object returned by ``urlopen``.

    Iterating yields SSE byte lines; it also supports the context-manager
    protocol used by ``with response:``.
    """

    def __init__(self, lines: list[bytes]) -> None:
        self._lines = lines

    def __iter__(self):
        return iter(self._lines)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _sse(*chunks: dict) -> list[bytes]:
    """Build SSE ``data:`` byte lines terminated by ``[DONE]``."""
    lines = [b"data: " + _json.dumps(c).encode("utf-8") for c in chunks]
    lines.append(b"data: [DONE]")
    return lines


def _delta(content=None, reasoning=None) -> dict:
    d: dict[str, Any] = {}
    if content is not None:
        d["content"] = content
    if reasoning is not None:
        d["reasoning_content"] = reasoning
    return {"choices": [{"index": 0, "delta": d, "finish_reason": None}]}


def _usage_chunk(prompt: int, completion: int) -> dict:
    return {
        "choices": [],
        "usage": {"prompt_tokens": prompt, "completion_tokens": completion},
    }


@pytest.fixture
def mock_urlopen(monkeypatch):
    """Patch ``urlopen`` to return canned SSE and capture the request."""
    captured: dict[str, Any] = {}

    def factory(lines: list[bytes]):
        def fake_urlopen(request, timeout=None):  # noqa: ANN001
            captured["url"] = request.full_url
            captured["headers"] = dict(request.header_items())
            captured["timeout"] = timeout
            captured["body"] = _json.loads(request.data.decode("utf-8"))
            return _FakeHTTPResponse(lines)

        monkeypatch.setattr(
            llamacpp_agent.urllib.request, "urlopen", fake_urlopen
        )
        return captured

    return factory


# ---- construction & endpoint normalisation --------------------------


def test_server_mode_requires_no_backend_package(monkeypatch):
    # Even with the backend package unavailable, server mode constructs fine.
    monkeypatch.setattr(llamacpp_agent, "_LLAMACPP_AVAILABLE", False)
    monkeypatch.setattr(llamacpp_agent, "Llama", None)
    agent = LlamacppAgent(default_model="qwen3", base_url="http://127.0.0.1:8080")
    assert agent.server_mode is True
    assert agent._chat_endpoint == "http://127.0.0.1:8080/v1/chat/completions"


@pytest.mark.parametrize(
    "base,expected",
    [
        ("http://h:8080", "http://h:8080/v1/chat/completions"),
        ("http://h:8080/", "http://h:8080/v1/chat/completions"),
        ("http://h:8080/v1", "http://h:8080/v1/chat/completions"),
        ("http://h:8080/v1/", "http://h:8080/v1/chat/completions"),
        (
            "http://h:8080/v1/chat/completions",
            "http://h:8080/v1/chat/completions",
        ),
    ],
)
def test_endpoint_normalisation(base, expected):
    agent = LlamacppAgent(default_model="m", base_url=base)
    assert agent._chat_endpoint == expected


def test_base_url_via_backend_options():
    agent = LlamacppAgent(
        default_model="m", backend_options={"base_url": "http://h:8080"}
    )
    assert agent.server_mode is True
    assert agent._chat_endpoint == "http://h:8080/v1/chat/completions"


# ---- streaming behaviour --------------------------------------------


def test_server_stream_yields_content_and_commits(mock_urlopen):
    mock_urlopen(_sse(_delta(content="Hello"), _delta(content=" world"),
                      _usage_chunk(5, 2)))
    agent = LlamacppAgent(default_model="qwen3", base_url="http://h:8080")
    out = [t for t in agent.stream("Hi") if t]
    assert out == ["Hello", " world"]
    history = agent.contexts["default"].message_history
    assert history[-1]["role"] == "assistant"
    assert history[-1]["content"] == "Hello world"


def test_server_reasoning_hidden_by_default(mock_urlopen):
    mock_urlopen(
        _sse(
            _delta(reasoning="thinking..."),
            _delta(reasoning=" more"),
            _delta(content="Answer"),
            _usage_chunk(3, 4),
        )
    )
    agent = LlamacppAgent(default_model="qwen3", base_url="http://h:8080")
    full = "".join(agent.stream("Hi"))
    assert full == "Answer"
    assert "<think>" not in full
    assert agent.contexts["default"].message_history[-1]["content"] == "Answer"


def test_server_reasoning_surfaced_when_enabled(mock_urlopen):
    mock_urlopen(
        _sse(
            _delta(reasoning="step1"),
            _delta(reasoning=" step2"),
            _delta(content="Final"),
            _usage_chunk(3, 4),
        )
    )
    agent = LlamacppAgent(
        default_model="qwen3",
        base_url="http://h:8080",
        backend_options={"include_reasoning": True},
    )
    full = "".join(agent.stream("Hi"))
    assert full == "<think>step1 step2</think>\nFinal"


def test_server_reasoning_only_closes_think_tag(mock_urlopen):
    # No content chunk ever arrives — the closing tag must still be emitted.
    mock_urlopen(_sse(_delta(reasoning="only thinking"), _usage_chunk(1, 1)))
    agent = LlamacppAgent(
        default_model="qwen3",
        base_url="http://h:8080",
        backend_options={"include_reasoning": True},
    )
    full = "".join(agent.stream("Hi"))
    assert full == "<think>only thinking</think>\n"


def test_server_request_body_contents(mock_urlopen):
    captured = mock_urlopen(_sse(_delta(content="x"), _usage_chunk(1, 1)))
    agent = LlamacppAgent(
        default_model="qwen3", base_url="http://h:8080", temperature=0.2
    )
    agent.max_tokens = 42
    list(agent.stream("Hi"))
    body = captured["body"]
    assert body["model"] == "qwen3"
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}
    assert body["temperature"] == 0.2
    assert body["max_tokens"] == 42
    assert body["messages"][-1]["content"] == "Hi"


def test_server_omits_max_tokens_when_unset(mock_urlopen):
    captured = mock_urlopen(_sse(_delta(content="x"), _usage_chunk(1, 1)))
    agent = LlamacppAgent(default_model="qwen3", base_url="http://h:8080")
    list(agent.stream("Hi"))
    assert "max_tokens" not in captured["body"]


def test_server_forwards_sampling_but_filters_load_options(mock_urlopen):
    captured = mock_urlopen(_sse(_delta(content="x"), _usage_chunk(1, 1)))
    agent = LlamacppAgent(
        default_model="qwen3",
        base_url="http://h:8080",
        backend_options={"top_p": 0.9, "repeat_penalty": 1.1, "n_ctx": 8192},
    )
    list(agent.stream("Hi"))
    body = captured["body"]
    assert body["top_p"] == 0.9
    assert body["repeat_penalty"] == 1.1
    # Server-launch-only option must NOT leak into the request body.
    assert "n_ctx" not in body


def test_server_sets_api_key_header(mock_urlopen):
    captured = mock_urlopen(_sse(_delta(content="x"), _usage_chunk(1, 1)))
    agent = LlamacppAgent(
        default_model="qwen3",
        base_url="http://h:8080",
        backend_options={"api_key": "secret"},
    )
    list(agent.stream("Hi"))
    # urllib title-cases header names.
    assert captured["headers"].get("Authorization") == "Bearer secret"


def test_server_metrics_token_usage(mock_urlopen):
    from pithos.metrics import MetricsCollector

    mock_urlopen(_sse(_delta(content="hi"), _usage_chunk(7, 3)))
    agent = LlamacppAgent(default_model="qwen3", base_url="http://h:8080")
    metrics = MetricsCollector()
    agent.attach_metrics(metrics)
    list(agent.stream("Hi"))
    tm = metrics._token_usage["qwen3"]
    assert tm.prompt_tokens == 7
    assert tm.completion_tokens == 3


def test_server_http_error_is_wrapped(monkeypatch):
    import urllib.error

    def boom(request, timeout=None):  # noqa: ANN001
        raise urllib.error.HTTPError(
            request.full_url, 500, "Server Error", {}, None
        )

    monkeypatch.setattr(llamacpp_agent.urllib.request, "urlopen", boom)
    agent = LlamacppAgent(default_model="qwen3", base_url="http://h:8080")
    with pytest.raises(RuntimeError, match="HTTP 500"):
        list(agent.stream("Hi"))


def test_server_connection_error_rolls_back_user_message(monkeypatch):
    def boom(request, timeout=None):  # noqa: ANN001
        raise OSError("connection refused")

    monkeypatch.setattr(llamacpp_agent.urllib.request, "urlopen", boom)
    agent = LlamacppAgent(default_model="qwen3", base_url="http://h:8080")
    before = len(agent.contexts["default"].message_history)
    with pytest.raises(RuntimeError, match="llama.cpp server"):
        list(agent.stream("Hi"))
    assert len(agent.contexts["default"].message_history) == before


# ---- config construction --------------------------------------------


def test_from_dict_builds_server_agent(tmp_path):
    from pithos.config_manager import ConfigManager

    cm = ConfigManager(config_dir=str(tmp_path))
    agent = LlamacppAgent.from_dict(
        {
            "model": "qwen3",
            "name": "qwen-server",
            "base_url": "http://127.0.0.1:8080",
            "temperature": 0.3,
            "backend_options": {"top_p": 0.95, "include_reasoning": True},
        },
        cm,
    )
    assert agent.server_mode is True
    assert agent.agent_name == "qwen-server"
    assert agent.temperature == 0.3
    assert agent.include_reasoning is True
    assert agent._chat_endpoint == "http://127.0.0.1:8080/v1/chat/completions"
    assert agent.backend_options["top_p"] == 0.95

