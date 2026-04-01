"""Integration tests for context compaction using a real Ollama LLM.

These tests are marked ``integration`` and are **excluded** from the default
test suite.  Run them explicitly with::

    pytest -m integration tests/test_compaction_steps_llm.py

They require a running Ollama instance with the ``glm-4.7-flash`` model
(or whichever model is configured below).
"""

import os as _os
import sys as _sys

# OLLAMA_HOST may be a bind address like 0.0.0.0 which the Python client
# can't connect to.  Normalise to localhost *before* any ollama import so
# the default client gets the corrected value.
_host = _os.environ.get("OLLAMA_HOST", "http://localhost:11434")
if _host in ("0.0.0.0", "0.0.0.0:11434"):
    _os.environ["OLLAMA_HOST"] = "http://localhost:11434"
    # Drop any cached ollama modules so they pick up the new env var
    for _k in [k for k in _sys.modules if k == "ollama" or k.startswith("ollama.")]:
        del _sys.modules[_k]

import pytest
from unittest.mock import MagicMock

from pithos.context import AgentContext
from pithos.agent.compaction import CompactionConfig, MemoryCompactor

# ---------------------------------------------------------------------------
# Configuration — change MODEL if you use a different local model
# ---------------------------------------------------------------------------
MODEL = "glm-4.7-flash"

# ---------------------------------------------------------------------------
# Ollama availability check — skip the whole module when unreachable
# ---------------------------------------------------------------------------
_ollama_available = False
try:
    from ollama import Client as _OllamaClient

    _client = _OllamaClient(host="http://localhost:11434")
    _client.list()
    _ollama_available = True
except Exception:
    pass

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _ollama_available, reason="Ollama is not running"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_context_with_conversation(turns: list[tuple[str, str]]) -> AgentContext:
    """Build a context from a list of (role, content) tuples."""
    ctx = AgentContext("test", "You are a helpful assistant.")
    for role, content in turns:
        ctx.message_history.append({"role": role, "content": content})
    return ctx


def _make_agent(model: str = MODEL) -> MagicMock:
    """Mock agent with just the attributes MemoryCompactor needs."""
    agent = MagicMock()
    agent.default_model = model
    agent.memory_enabled = False
    agent.memory_store = None
    return agent


def _make_compactor(
    threshold: int = 8, keep_last: int = 2, **kwargs
) -> MemoryCompactor:
    return MemoryCompactor(
        CompactionConfig(
            threshold=threshold,
            keep_last=keep_last,
            summary_model=MODEL,
            # Use -1 (unlimited) because glm-4.7-flash returns empty
            # responses when num_predict is a positive value.
            **kwargs,
        )
    )


def _generate_summary_with_retry(
    compactor: MemoryCompactor, agent, messages: list[dict], retries: int = 3
) -> tuple[str, str]:
    """Retry _generate_summary up to *retries* times to handle LLM non-determinism.

    The local LLM occasionally returns an empty response; retrying is the
    simplest way to get a usable result for content-assertion tests.
    """
    for attempt in range(retries):
        summary, entities = compactor._generate_summary(agent, messages)
        if summary.strip() and "unavailable" not in summary.lower():
            return summary, entities
    return summary, entities


# ===========================================================================
# 1. _generate_summary — LLM produces a non-empty summary
# ===========================================================================


class TestGenerateSummaryLLM:
    """Validate that the real LLM returns usable summaries."""

    def test_returns_non_empty_summary(self):
        compactor = _make_compactor()
        agent = _make_agent()
        messages = [
            {"role": "user", "content": "What is Python?"},
            {
                "role": "assistant",
                "content": "Python is a high-level programming language.",
            },
            {"role": "user", "content": "What about JavaScript?"},
            {"role": "assistant", "content": "JavaScript is a language for the web."},
        ]

        summary, entities = compactor._generate_summary(agent, messages)

        assert isinstance(summary, str)
        assert len(summary.strip()) > 0

    def test_returns_entities_string(self):
        compactor = _make_compactor()
        agent = _make_agent()
        messages = [
            {"role": "user", "content": "Tell me about Alice and Bob in cryptography."},
            {
                "role": "assistant",
                "content": "Alice and Bob are placeholder names used in cryptography to describe communication protocols.",
            },
        ]

        summary, entities = compactor._generate_summary(agent, messages)

        assert isinstance(entities, str)
        # entities should be a non-empty string (could be "none" or actual names)
        assert len(entities.strip()) > 0

    def test_summary_captures_key_topic(self):
        """The LLM summary should mention the topic discussed."""
        compactor = _make_compactor()
        agent = _make_agent()
        messages = [
            {"role": "user", "content": "Explain photosynthesis."},
            {
                "role": "assistant",
                "content": "Photosynthesis is the process by which plants convert sunlight, water, and CO2 into glucose and oxygen.",
            },
            {"role": "user", "content": "Where does it take place?"},
            {
                "role": "assistant",
                "content": "It takes place in the chloroplasts of plant cells.",
            },
        ]

        summary, entities = _generate_summary_with_retry(compactor, agent, messages)

        # The summary should reference the topic
        summary_lower = summary.lower()
        assert (
            "photosynth" in summary_lower
            or "plant" in summary_lower
            or "chloro" in summary_lower
        )

    def test_single_message_summary(self):
        """Even a single message should produce a valid summary."""
        compactor = _make_compactor()
        agent = _make_agent()
        messages = [{"role": "user", "content": "Hello, world!"}]

        summary, entities = compactor._generate_summary(agent, messages)

        assert isinstance(summary, str)
        assert len(summary.strip()) > 0

    def test_long_conversation_summary(self):
        """Summarising a longer conversation should still work."""
        compactor = _make_compactor()
        agent = _make_agent()
        messages = [
            {"role": "user", "content": "What is machine learning?"},
            {
                "role": "assistant",
                "content": "Machine learning is a subset of AI where systems learn from data.",
            },
            {"role": "user", "content": "What are the main types?"},
            {
                "role": "assistant",
                "content": "Supervised, unsupervised, and reinforcement learning.",
            },
            {"role": "user", "content": "Give an example of supervised learning."},
            {
                "role": "assistant",
                "content": "Email spam classification is a classic supervised learning example.",
            },
            {"role": "user", "content": "What about unsupervised?"},
            {
                "role": "assistant",
                "content": "Customer segmentation using clustering is a common unsupervised learning task.",
            },
            {"role": "user", "content": "And reinforcement learning?"},
            {
                "role": "assistant",
                "content": "Training a robot to walk or game-playing AI like AlphaGo.",
            },
        ]

        summary, entities = compactor._generate_summary(agent, messages)

        assert isinstance(summary, str)
        assert len(summary.strip()) > 10  # reasonably substantial


# ===========================================================================
# 2. _parse_summary_response with real LLM output
# ===========================================================================


class TestParseSummaryResponseWithLLMOutput:
    """Feed real LLM output through the parser to verify robustness."""

    def test_parse_llm_generated_response(self):
        """Generate a real response and verify parsing doesn't crash."""
        compactor = _make_compactor()
        agent = _make_agent()
        messages = [
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "The capital of France is Paris."},
        ]

        summary, entities = compactor._generate_summary(agent, messages)

        # Re-parse: the output from _generate_summary already went through
        # _parse_summary_response, so just verify the types
        assert isinstance(summary, str)
        assert isinstance(entities, str)
        assert summary.strip() != ""


# ===========================================================================
# 3. Full compact() flow with real LLM
# ===========================================================================


class TestCompactWithRealLLM:
    """End-to-end compaction using a live Ollama model."""

    def test_compact_reduces_message_count(self):
        compactor = _make_compactor(threshold=6, keep_last=2)
        ctx = _make_context_with_conversation(
            [
                ("user", "What is Python?"),
                ("assistant", "Python is a programming language."),
                ("user", "What about Java?"),
                ("assistant", "Java is a statically-typed language."),
                ("user", "Which is faster?"),
                ("assistant", "Java is generally faster at runtime."),
                ("user", "Which is easier to learn?"),
                ("assistant", "Python is generally considered easier."),
            ]
        )
        agent = _make_agent()
        original_count = len(ctx.message_history)

        compactor.compact(agent, ctx, "test_chat")

        assert len(ctx.message_history) < original_count

    def test_compact_inserts_summary_message(self):
        compactor = _make_compactor(threshold=6, keep_last=2)
        ctx = _make_context_with_conversation(
            [
                ("user", "Explain gravity."),
                ("assistant", "Gravity is a force that attracts objects with mass."),
                ("user", "Who discovered it?"),
                (
                    "assistant",
                    "Isaac Newton formulated the law of universal gravitation.",
                ),
                ("user", "What about Einstein?"),
                ("assistant", "Einstein refined it with general relativity."),
                ("user", "Are there gravitational waves?"),
                ("assistant", "Yes, first detected by LIGO in 2015."),
            ]
        )
        agent = _make_agent()

        compactor.compact(agent, ctx, "test_chat")

        summary_msgs = [
            m
            for m in ctx.message_history
            if "[CONTEXT SUMMARY]" in m.get("content", "")
        ]
        assert len(summary_msgs) == 1

    def test_summary_is_protected(self):
        compactor = _make_compactor(threshold=6, keep_last=2)
        ctx = _make_context_with_conversation(
            [
                ("user", "What is DNA?"),
                ("assistant", "DNA carries genetic instructions."),
                ("user", "What is RNA?"),
                ("assistant", "RNA helps translate DNA into proteins."),
                ("user", "What's the difference?"),
                ("assistant", "DNA is double-stranded, RNA is single-stranded."),
                ("user", "What is a gene?"),
                ("assistant", "A gene is a segment of DNA that codes for a protein."),
            ]
        )
        agent = _make_agent()

        compactor.compact(agent, ctx, "test_chat")

        summary_msg = next(
            m
            for m in ctx.message_history
            if "[CONTEXT SUMMARY]" in m.get("content", "")
        )
        assert summary_msg["_pithos_no_compact"] is True
        assert summary_msg["role"] == "system"

    def test_kept_messages_are_the_most_recent(self):
        compactor = _make_compactor(threshold=6, keep_last=3)
        last_three = [
            ("user", "What about TypeScript?"),
            ("assistant", "TypeScript adds static types to JavaScript."),
            ("user", "Should I use it?"),
        ]
        turns = [
            ("user", "What is JavaScript?"),
            ("assistant", "A web scripting language."),
            ("user", "What is Python?"),
            ("assistant", "A general-purpose language."),
            ("user", "Compare them."),
            ("assistant", "Python for backends, JS for frontends."),
        ] + last_three
        ctx = _make_context_with_conversation(turns)
        agent = _make_agent()
        expected_kept = [t[1] for t in last_three]

        compactor.compact(agent, ctx, "test_chat")

        non_summary = [
            m["content"]
            for m in ctx.message_history
            if "[CONTEXT SUMMARY]" not in m.get("content", "")
        ]
        assert non_summary == expected_kept

    def test_summary_content_is_meaningful(self):
        """The LLM-generated summary should reference the conversation topic."""
        compactor = _make_compactor(threshold=6, keep_last=1)
        messages = [
            {"role": "user", "content": "Tell me about the solar system."},
            {
                "role": "assistant",
                "content": "The solar system has 8 planets orbiting the Sun.",
            },
            {"role": "user", "content": "Name the gas giants."},
            {
                "role": "assistant",
                "content": "Jupiter, Saturn, Uranus, and Neptune are the gas giants.",
            },
            {"role": "user", "content": "Which is largest?"},
            {
                "role": "assistant",
                "content": "Jupiter is the largest planet in our solar system.",
            },
            {"role": "user", "content": "How far is it from the Sun?"},
            {
                "role": "assistant",
                "content": "Jupiter is about 778 million km from the Sun.",
            },
        ]
        agent = _make_agent()

        # Test the summary generation directly with retry to handle LLM
        # non-determinism, then verify the content is meaningful.
        summary, entities = _generate_summary_with_retry(compactor, agent, messages)

        content_lower = summary.lower()
        # Should mention something space-related
        assert any(
            word in content_lower
            for word in ["solar", "planet", "jupiter", "sun", "gas giant"]
        )

    def test_protected_messages_survive_compaction(self):
        """Protected messages (like previous summaries) should not be removed."""
        compactor = _make_compactor(threshold=5, keep_last=1)
        ctx = AgentContext("test", "sys")
        # Insert a protected message at the start
        ctx.message_history.append(
            {
                "role": "system",
                "content": "Previous summary: user asked about math.",
                "_pithos_no_compact": True,
            }
        )
        turns = [
            ("user", "What is calculus?"),
            ("assistant", "Calculus is the study of change."),
            ("user", "Who invented it?"),
            ("assistant", "Newton and Leibniz independently developed it."),
            ("user", "What is a derivative?"),
            ("assistant", "A derivative measures instantaneous rate of change."),
        ]
        for role, content in turns:
            ctx.message_history.append({"role": role, "content": content})
        agent = _make_agent()

        compactor.compact(agent, ctx, "test_chat")

        # The protected message must still be present
        protected = [
            m
            for m in ctx.message_history
            if m.get("content", "").startswith("Previous summary")
        ]
        assert len(protected) == 1

    def test_consecutive_compactions_with_real_llm(self):
        """Two rounds of compaction should both produce valid summaries."""
        compactor = _make_compactor(threshold=5, keep_last=2)
        ctx = _make_context_with_conversation(
            [
                ("user", "What is an atom?"),
                ("assistant", "The basic unit of matter."),
                ("user", "What are protons?"),
                ("assistant", "Positively charged subatomic particles."),
                ("user", "And neutrons?"),
                ("assistant", "Neutral subatomic particles in the nucleus."),
            ]
        )
        agent = _make_agent()

        # First compaction
        compactor.compact(agent, ctx, "test_chat")
        summaries_after_first = [
            m
            for m in ctx.message_history
            if "[CONTEXT SUMMARY]" in m.get("content", "")
        ]
        assert len(summaries_after_first) == 1

        # Add more messages
        more_turns = [
            ("user", "What are electrons?"),
            ("assistant", "Negatively charged particles orbiting the nucleus."),
            ("user", "What is an ion?"),
            ("assistant", "An atom with a net electric charge."),
            ("user", "What is a molecule?"),
            ("assistant", "Two or more atoms bonded together."),
        ]
        for role, content in more_turns:
            ctx.message_history.append({"role": role, "content": content})

        # Second compaction
        compactor.compact(agent, ctx, "test_chat")
        summaries_after_second = [
            m
            for m in ctx.message_history
            if "[CONTEXT SUMMARY]" in m.get("content", "")
        ]
        assert len(summaries_after_second) == 2
        # Both are protected
        for s in summaries_after_second:
            assert s["_pithos_no_compact"] is True

    def test_compact_with_keep_last_zero(self):
        """With keep_last=0, all non-protected messages are compacted."""
        compactor = _make_compactor(threshold=4, keep_last=0)
        ctx = _make_context_with_conversation(
            [
                ("user", "Hello"),
                ("assistant", "Hi there!"),
                ("user", "Goodbye"),
                ("assistant", "See you later!"),
            ]
        )
        agent = _make_agent()

        compactor.compact(agent, ctx, "test_chat")

        # Only the summary should remain
        assert len(ctx.message_history) == 1
        assert "[CONTEXT SUMMARY]" in ctx.message_history[0]["content"]

    def test_compact_archives_to_memory_store(self):
        """With memory enabled, the summary is archived to the mock store."""
        compactor = _make_compactor(threshold=6, keep_last=2)
        ctx = _make_context_with_conversation(
            [
                ("user", "What is recursion?"),
                ("assistant", "A function that calls itself."),
                ("user", "Give an example."),
                ("assistant", "Factorial: f(n) = n * f(n-1)."),
                ("user", "What about base case?"),
                ("assistant", "The termination condition, e.g. f(0) = 1."),
                ("user", "Is recursion always efficient?"),
                (
                    "assistant",
                    "Not always — it can cause stack overflow for deep calls.",
                ),
            ]
        )
        agent = _make_agent()
        agent.memory_enabled = True
        mock_store = MagicMock()
        agent.memory_store = mock_store

        compactor.compact(agent, ctx, "test_chat")

        # store.store() should have been called once
        mock_store.store.assert_called_once()
        args, kwargs = mock_store.store.call_args
        # First arg is the category
        assert args[0] == "context_summaries"
        # Second arg is the summary text (non-empty)
        assert len(args[1].strip()) > 0
        # Metadata
        assert kwargs["metadata"]["type"] == "compaction_summary"
        assert kwargs["metadata"]["context_name"] == "test_chat"
