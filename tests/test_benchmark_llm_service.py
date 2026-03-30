"""Tests for LLM service interface and dummy implementation."""

from benchmarks.easy_problems.llm_service import (
    LLMService,
    dummy_llm_service,
)


class TestLLMServiceABC:
    def test_cannot_instantiate_abc(self):
        """LLMService ABC cannot be instantiated directly."""
        try:
            LLMService()
            assert False, "Should have raised TypeError"
        except TypeError:
            pass

    def test_subclass_must_implement_completion(self):
        """A subclass that doesn't implement completion() raises TypeError."""

        class BadService(LLMService):
            pass

        try:
            BadService()
            assert False, "Should have raised TypeError"
        except TypeError:
            pass

    def test_subclass_with_completion_works(self):
        """A properly implemented subclass can be instantiated."""

        class GoodService(LLMService):
            def completion(self, messages, **kwargs):
                return "ok"

        svc = GoodService()
        assert svc.completion([]) == "ok"


class TestDummyLLMService:
    def test_always_returns_answer_a(self):
        """Dummy service always returns JSON with ANSWER A."""
        svc = dummy_llm_service("test-model")
        result = svc.completion([{"role": "user", "content": "test"}])
        assert '"ANSWER"' in result
        assert '"A"' in result

    def test_is_llm_service_subclass(self):
        """dummy_llm_service is a proper LLMService."""
        svc = dummy_llm_service("test-model")
        assert isinstance(svc, LLMService)

    def test_accepts_kwargs(self):
        """completion() accepts arbitrary kwargs without error."""
        svc = dummy_llm_service("test-model")
        result = svc.completion([], temperature=0.5)
        assert result == '{"ANSWER": "A"}'
