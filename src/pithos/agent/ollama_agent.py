"""OllamaAgent — streaming-first LLM agent backed by Ollama."""

from typing import Optional, Any, Iterator
import logging
import uuid

from ollama import chat
from ollama._types import ResponseError as OllamaResponseError
import ollama

from ..context import Msg, UserMsg, AgentMsg, AgentContext
from .agent import Agent

logger = logging.getLogger(__name__)


class OllamaAgent(Agent):
    """LLM agent backed by Ollama.

    Streaming is the primary execution path.  :meth:`stream` yields tokens
    as they arrive and performs tool calls *inline*: when a complete tool
    invocation is detected in the accumulated buffer, streaming is interrupted,
    the tool is executed, the result is injected into the context as a system
    message, and a continuation stream is started transparently.

    :meth:`send` is a convenience wrapper that collects the full stream into a
    single string.
    """

    def stream(
        self,
        content: str,
        context_name: Optional[str] = None,
        workspace: Optional[str] = None,
        verbose: bool = False,
        model: Optional[str] = None,
    ) -> Iterator[str]:
        """Stream response tokens from Ollama with mid-stream tool execution.

        Yields chunks as they arrive.  When a complete tool call is detected
        in the accumulated output the stream is interrupted: the partial
        response is committed to context, the tool is executed and its result
        injected as a system message, then a new continuation stream is
        started and its chunks yielded seamlessly.

        Memory operations (STORE / RETRIEVE) and auto-compaction are performed
        after the *final* continuation exits — i.e. after the model produces
        a response that contains no further tool calls.

        Callers MUST consume the iterator to completion for all side-effects
        (context update, history, compaction) to take place.

        Yields:
            Text chunks produced by the model.
        """
        import time as _time

        ctx = context_name or self.current_context
        if not ctx:
            raise ValueError("No context selected.")
        if ctx not in self.contexts:
            self.create_context(ctx)

        context = self.contexts[ctx]

        # Auto-recall: inject relevant memories before the user message
        if self.recall_enabled and self._auto_recall:
            try:
                self._auto_recall.inject_recall(
                    agent=self, context=context, content=content, model=model
                )
            except Exception as exc:
                logger.warning("Auto-recall failed (non-fatal): %s", exc)

        # If an inference flowchart is set and we are not already inside one,
        # run it and yield the full result as a single chunk (flowchart
        # execution is inherently non-streaming).
        if self.inference_flowchart and not self._running_inference:
            result = self._inference_send(
                content, ctx, context, workspace, verbose, model
            )
            yield result
            return

        context.add_message(UserMsg(content))

        try:
            messages = context.get_messages(workspace)
            if verbose:
                logger.debug(">>> STREAM: %s", content)

            options: dict[str, Any] = {"temperature": self.temperature}
            if self.max_tokens != -1:
                options["num_predict"] = self.max_tokens

            model_to_use = model or self.default_model

            raw_stream = chat(
                model=model_to_use,
                messages=messages,
                options=options,
                stream=True,
            )

            accumulated = ""
            _t0 = _time.monotonic()
            _last_chunk = None
            # Hashes of raw_text for tool calls already executed this turn, so
            # that re-reading old text in accumulated doesn't re-trigger them.
            _seen_raw: set[str] = set()

            for chunk in raw_stream:
                token = chunk.message.content or ""
                accumulated += token
                if verbose:
                    logger.debug("%s", token)
                _last_chunk = chunk
                yield token

                # Mid-stream tool detection: only execute newly-seen complete calls.
                if self.tools_enabled and self.tool_registry and self.tool_executor:
                    all_calls = self._extract_tool_calls(accumulated)
                    new_calls = [c for c in all_calls if c.raw_text not in _seen_raw]
                    if new_calls:
                        for c in new_calls:
                            _seen_raw.add(c.raw_text)
                        # Commit the partial response accumulated so far.
                        context.add_message(AgentMsg(accumulated))
                        # Execute the tool(s) and inject result.
                        result_msg = self._execute_tools(new_calls)
                        context.add_message(Msg("system", result_msg))
                        # Persist the user turn the first time (content != "").
                        if content:
                            self._history_persist(ctx, "user", content)
                        # Continue via a recursive stream call (empty content =
                        # no new user turn; model sees its own output + tool
                        # result and continues).
                        yield from self.stream(
                            "",
                            context_name=ctx,
                            workspace=workspace,
                            verbose=verbose,
                            model=model,
                        )
                        # The recursive call owns compaction / history / memory
                        # for the remainder of this interaction.
                        return

            _response_ms = (_time.monotonic() - _t0) * 1000.0

            if verbose:
                logger.debug("-" * 40)

            # Record token usage from the final chunk
            if self.metrics is not None:
                try:
                    usage = getattr(_last_chunk, "usage", None) if _last_chunk else None
                    prompt_tok = getattr(usage, "prompt_tokens", 0) or 0
                    completion_tok = getattr(usage, "completion_tokens", 0) or 0
                    self.metrics.record_token_usage(
                        model=model_to_use,
                        prompt_tokens=prompt_tok,
                        completion_tokens=completion_tok,
                        response_time_ms=_response_ms,
                    )
                except Exception:
                    pass

        except OllamaResponseError as e:
            context.remove_last_message()
            raise e
        except Exception as exc:
            context.remove_last_message()
            raise RuntimeError(
                f"Failed to communicate with Ollama: {exc}. "
                "Ensure Ollama is running. "
                "If using localhost, try setting "
                "OLLAMA_HOST=http://127.0.0.1:11434 to avoid IPv6 resolution issues."
            ) from exc

        # No tool interruption occurred — commit the full response.
        context.add_message(AgentMsg(accumulated))

        # Persist to conversation history (skip empty content = continuation turns)
        if content:
            self._history_persist(ctx, "user", content)
        self._history_persist(ctx, "assistant", accumulated, set_as_last=True)

        # Memory operations post-stream
        if self.memory_enabled and self.memory_store:
            mem_ops = self._extract_memory_ops(accumulated)
            if mem_ops:
                result_msg = self._execute_memory_ops(mem_ops)
                context.add_message(Msg("system", result_msg))

        # Auto-compaction
        if self.compaction_enabled and self._compactor:
            try:
                self._compactor.compact(agent=self, context=context, context_name=ctx)
            except Exception as exc:
                logger.warning("Auto-compaction failed (non-fatal): %s", exc)

    # ------------------------------------------------------------------
    # Inference flowchart path (non-streaming; yields a single chunk)
    # ------------------------------------------------------------------

    def _inference_send(
        self,
        content: str,
        ctx: str,
        context: AgentContext,
        workspace: Optional[str],
        verbose: bool,
        model: Optional[str],
    ) -> str:
        """Run the chain-of-thought inference flowchart for a user message.

        Executes the inference flowchart with *content* as initial input.
        PromptNodes inside the flowchart invoke the agent's underlying LLM
        call (the ``_running_inference`` guard prevents infinite recursion).

        The user message and final response are recorded in the main
        conversation context (*context*), but all intermediate flowchart
        reasoning happens in a temporary context that is discarded
        afterwards.

        Returns:
            The final response string produced by the flowchart.
        """
        context.add_message(UserMsg(content))

        tmp_ctx_name = f"_cot_{uuid.uuid4().hex[:8]}"
        self.create_context(tmp_ctx_name, self.default_system_prompt)
        saved_current = ctx

        self._running_inference = True
        try:
            fc = self.inference_flowchart
            assert fc is not None
            fc.reset()
            fc._initialize_message_routing()

            fc.message_router.shared_context["agent"] = self
            fc.message_router.shared_context["context_name"] = tmp_ctx_name
            fc.message_router.shared_context["model"] = model or self.default_model
            fc.message_router.shared_context["verbose"] = verbose

            result = fc.run_message_based(initial_data=content)
            response = ""
            if result.get("messages"):
                response = str(result["messages"][-1].data)
        except Exception as exc:
            logger.error("Inference flowchart failed: %s", exc)
            context.remove_last_message()
            raise RuntimeError(f"Inference flowchart execution failed: {exc}") from exc
        finally:
            self._running_inference = False
            if tmp_ctx_name in self.contexts:
                del self.contexts[tmp_ctx_name]
            self.current_context = saved_current

        context.add_message(AgentMsg(response))

        if content:
            self._history_persist(ctx, "user", content)
        self._history_persist(ctx, "assistant", response, set_as_last=True)

        # Tool calls post-processing on the final response.
        if self.tools_enabled and self.tool_registry and self.tool_executor:
            tool_requests = self._extract_tool_calls(response)
            if tool_requests:
                result_msg = self._execute_tools(tool_requests)
                context.add_message(Msg("system", result_msg))
                if self.tool_auto_loop:
                    return self.send(
                        "", context_name=ctx, workspace=workspace, verbose=verbose
                    )

        # Memory operations post-processing.
        if self.memory_enabled and self.memory_store:
            mem_ops = self._extract_memory_ops(response)
            if mem_ops:
                result_msg = self._execute_memory_ops(mem_ops)
                context.add_message(Msg("system", result_msg))
                if self.tool_auto_loop:
                    return self.send(
                        "", context_name=ctx, workspace=workspace, verbose=verbose
                    )

        # Auto-compaction.
        if self.compaction_enabled and self._compactor:
            try:
                self._compactor.compact(agent=self, context=context, context_name=ctx)
            except Exception as exc:
                logger.warning("Auto-compaction failed (non-fatal): %s", exc)

        return response
