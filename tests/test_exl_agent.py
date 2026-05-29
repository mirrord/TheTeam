"""Behavioral tests for :class:`EXLAgent` with mocked ExLlamaV2 stack."""

from __future__ import annotations

from typing import Any
import pytest

from pithos.agent import exl_agent
from pithos.agent.exl_agent import EXLAgent

# ---------------------------------------------------------------------
# Minimal stand-in for the ExLlamaV2 stack
# ---------------------------------------------------------------------


class _FakeConfig:
    def __init__(self, model_dir: str) -> None:
        self.model_dir = model_dir
        self.max_seq_len = 4096


class _FakeModel:
    def __init__(self, cfg: _FakeConfig) -> None:
        self.cfg = cfg

    def load_autosplit(self, cache: Any) -> None:
        self.loaded = True


class _FakeCache:
    def __init__(self, model: _FakeModel, **kwargs: Any) -> None:
        self.model = model
        self.kwargs = kwargs


class _FakeTokenizer:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def encode(self, text: str) -> Any:
        # Return an object with a ``shape`` attribute mimicking torch tensors.
        class _Shape(tuple):
            pass

        n = max(1, len(text.split()))
        s = _Shape((1, n))
        out = type("T", (), {"shape": s})()
        return out


class _FakeSamplerSettings:
    def __init__(self) -> None:
        self.temperature = 1.0


class _FakeSampler:
    Settings = _FakeSamplerSettings


class _FakeGenerator:
    instances: list["_FakeGenerator"] = []

    def __init__(self, model: Any, cache: Any, tokenizer: Any) -> None:
        self.model = model
        self.cache = cache
        self.tokenizer = tokenizer
        self.calls: list[dict[str, Any]] = []
        self.chunks: list[str] = ["Hello", " from", " EXL"]
        _FakeGenerator.instances.append(self)

    def stream(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        for tok in self.chunks:
            yield tok


class _FakeHFTokenizer:
    @classmethod
    def from_pretrained(cls, path: str) -> "_FakeHFTokenizer":
        inst = cls()
        inst.path = path
        return inst

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return (
            "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"
        )


@pytest.fixture
def fake_exl(monkeypatch):
    _FakeGenerator.instances.clear()
    monkeypatch.setattr(exl_agent, "_EXL_AVAILABLE", True)
    monkeypatch.setattr(exl_agent, "ExLlamaV2", _FakeModel)
    monkeypatch.setattr(exl_agent, "ExLlamaV2Cache", _FakeCache)
    monkeypatch.setattr(exl_agent, "ExLlamaV2Config", _FakeConfig)
    monkeypatch.setattr(exl_agent, "ExLlamaV2Tokenizer", _FakeTokenizer)
    monkeypatch.setattr(exl_agent, "ExLlamaV2DynamicGenerator", _FakeGenerator)
    monkeypatch.setattr(exl_agent, "ExLlamaV2Sampler", _FakeSampler)
    monkeypatch.setattr(exl_agent, "_HF_AVAILABLE", True)
    monkeypatch.setattr(exl_agent, "AutoTokenizer", _FakeHFTokenizer)
    return _FakeGenerator


# ---------------------------------------------------------------------
# Construction & lazy loading
# ---------------------------------------------------------------------


def test_construction_succeeds(fake_exl):
    agent = EXLAgent(default_model="/models/exl-dir")
    assert agent.default_model == "/models/exl-dir"
    assert agent._exl_generator is None  # lazy


def test_lazy_load_on_first_stream(fake_exl):
    agent = EXLAgent(default_model="/models/exl-dir")
    list(agent.stream("Hi"))
    assert len(fake_exl.instances) == 1
    assert agent._loaded_model_path == "/models/exl-dir"


def test_changing_model_triggers_reload(fake_exl):
    agent = EXLAgent(default_model="/models/a")
    list(agent.stream("first"))
    list(agent.stream("second", model="/models/b"))
    assert len(fake_exl.instances) == 2


# ---------------------------------------------------------------------
# Streaming + chat templating
# ---------------------------------------------------------------------


def test_stream_yields_tokens_and_commits_response(fake_exl):
    agent = EXLAgent(default_model="/models/exl-dir")
    out = "".join(agent.stream("Question?"))
    assert "Hello from EXL" in out
    history = agent.contexts["default"].message_history
    assert history[-1]["role"] == "assistant"
    assert "Hello from EXL" in history[-1]["content"]


def test_chat_template_used_for_prompt(fake_exl):
    agent = EXLAgent(default_model="/models/exl-dir")
    list(agent.stream("Question?"))
    prompt = fake_exl.instances[-1].calls[-1]["prompt"]
    assert "user: Question?" in prompt
    assert prompt.rstrip().endswith("assistant:")


def test_temperature_passed_to_sampler(fake_exl):
    agent = EXLAgent(default_model="/models/exl-dir", temperature=0.4)
    list(agent.stream("Hi"))
    settings = fake_exl.instances[-1].calls[-1]["gen_settings"]
    assert settings.temperature == 0.4


def test_max_new_tokens_default(fake_exl):
    agent = EXLAgent(default_model="/models/exl-dir")
    list(agent.stream("Hi"))
    assert fake_exl.instances[-1].calls[-1]["max_new_tokens"] == 512


def test_max_new_tokens_uses_max_tokens(fake_exl):
    agent = EXLAgent(default_model="/models/exl-dir")
    agent.max_tokens = 128
    list(agent.stream("Hi"))
    assert fake_exl.instances[-1].calls[-1]["max_new_tokens"] == 128


# ---------------------------------------------------------------------
# Backend options
# ---------------------------------------------------------------------


def test_backend_options_max_seq_len_applied(fake_exl):
    agent = EXLAgent(
        default_model="/models/exl-dir",
        backend_options={"max_seq_len": 16384},
    )
    list(agent.stream("Hi"))
    # The cached cfg should reflect the override.
    cache = fake_exl.instances[-1].cache
    assert cache.model.cfg.max_seq_len == 16384


# ---------------------------------------------------------------------
# Error wrapping
# ---------------------------------------------------------------------


def test_stream_error_rolls_back_user_message(fake_exl):
    agent = EXLAgent(default_model="/models/exl-dir")
    list(agent.stream("warmup"))

    def boom(**kw):  # noqa: ANN001
        raise RuntimeError("kaboom")

    fake_exl.instances[-1].stream = boom  # type: ignore[assignment]

    history_len = len(agent.contexts["default"].message_history)
    with pytest.raises(RuntimeError, match="ExLlamaV2"):
        list(agent.stream("Hi"))
    assert len(agent.contexts["default"].message_history) == history_len


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------


def test_metrics_token_usage_recorded(fake_exl):
    from pithos.metrics import MetricsCollector

    agent = EXLAgent(default_model="/models/exl-dir")
    metrics = MetricsCollector()
    agent.attach_metrics(metrics)
    list(agent.stream("Hi"))

    assert "/models/exl-dir" in metrics._token_usage
    tm = metrics._token_usage["/models/exl-dir"]
    assert tm.completion_tokens == 3  # three chunks emitted
    assert tm.prompt_tokens >= 1
