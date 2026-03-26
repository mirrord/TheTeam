import json
from abc import ABC, abstractmethod
from httpx import Timeout
from ollama import chat
from pithos import OllamaAgent, ConfigManager, Flowchart

DEFAULT_TIMEOUT = 120


class LLMService(ABC):
    """Abstract base class for all benchmark LLM services."""

    @abstractmethod
    def completion(self, messages: list, **kwargs) -> str:
        """Run a completion request and return the text response."""
        ...


def ollama_message_parse(response: dict | list):
    """Parse Ollama API response to extract message content."""
    # print(f"\tollama parsing: {response=}")
    try:
        if isinstance(response, dict):
            text_response = response["message"]["content"]
        else:
            text_response = "".join(
                [line["message"]["content"] for line in response if "message" in line]
            )
        return text_response
    except Exception as e:
        if response in [{}, []]:
            return "None"
        else:
            raise ValueError(
                f"Response does not contain the expected key 'content'. "
                f"Error: {e}. Response: {response}"
            )


def null_parse(response: dict | list):
    """Pass-through parser that returns response unchanged."""
    return response


class dummy_llm_service(LLMService):
    """Dummy LLM service for testing that always returns 'A'."""

    def __init__(self, model: str):
        self.model = model

    def completion(self, message: list, **kwargs):
        return '{"ANSWER": "A"}'


class ollama_llm_service(LLMService):
    """Basic Ollama LLM service for making completion requests."""

    def __init__(self, model: str):
        self.model = model

    def ollama_query(
        self, messages: list, temperature=0, max_tokens=2048, timeout=DEFAULT_TIMEOUT
    ):
        # print(f"*************\n{messages=}")
        options = {"temperature": temperature}
        # Only include num_predict if max_tokens is not -1 (unlimited)
        if max_tokens != -1:
            options["num_predict"] = max_tokens

        a = json.loads(
            chat(
                model=self.model,
                messages=messages,
                options=options,
                timeout=Timeout(timeout),
            ).model_dump_json()
        )  # .message.content
        # print(f"{a}\n******************")
        return a

    def completion(self, messages: list, **kwargs):
        # print(kwargs)
        return ollama_message_parse(self.ollama_query(messages, **kwargs))


class ollama_stepper(ollama_llm_service):
    def ollama_query(
        self, messages: list, temperature=0, max_tokens=2048, timeout=DEFAULT_TIMEOUT
    ):
        messages[0]["content"] += "\nLet's think step by step:"
        options = {"temperature": temperature}
        # Only include num_predict if max_tokens is not -1 (unlimited)
        if max_tokens != -1:
            options["num_predict"] = max_tokens

        a = json.loads(
            chat(
                model=self.model,
                messages=messages,
                options=options,
                timeout=Timeout(timeout),
            ).model_dump_json()
        )
        return a


class ollama_agent_service(ollama_llm_service):
    """Ollama service using pithos OllamaAgent."""

    def __init__(self, model: str):
        self.model = OllamaAgent(model)

    def ollama_query(self, messages: list, temperature=0):
        return self.model.send(messages[0]["content"])

    def completion(self, messages, **kwargs):
        self.model.wipe_memory()
        return super().completion(messages, **kwargs)


class ollama_named(ollama_agent_service):
    """Ollama service using a named agent configuration."""

    def __init__(self, agent_name: str, config_manager: ConfigManager):
        self.model = OllamaAgent.from_config(
            agent_name,
            config_manager,
        )

    def ollama_query(self, messages: list, temperature=0):
        return {
            "message": {
                "role": "assistant",
                "content": self.model.reason(messages[0]["content"]),
            }
        }


class ollama_flow(ollama_named):
    def __init__(self, agent_name: str, flowchart: str, config_manager: ConfigManager):
        super().__init__(agent_name, config_manager)
        self.flowchart_name = flowchart
        self.flowchart = Flowchart.from_registered(flowchart, config_manager)

    def ollama_query(self, messages: list, temperature=0):
        result = self.model.follow_flowchart(self.flowchart, messages[0]["content"])
        return {
            "message": {
                "role": "assistant",
                "content": result,
            }
        }


class ollama_reflector(ollama_agent_service):
    def ollama_query(self, messages: list, temperature=0):
        self.model.create_context("default")
        rsp = self.model.reflect_in_situ(generation_prompt=messages[0]["content"])
        return {"message": {"role": "assistant", "content": rsp}}


class ollama_reflector_simple(ollama_agent_service):
    def ollama_query(self, messages: list, temperature=0):
        self.model.create_context("default")
        rsp = self.model.reflect_simple(generation_prompt=messages[0]["content"])
        return {"message": {"role": "assistant", "content": rsp}}


class ollama_reflector_adversarial(ollama_agent_service):
    def ollama_query(self, messages: list, temperature=0):
        self.model.create_context("default")
        rsp = self.model.reflect_adversarial(messages[0]["content"])
        return {"message": {"role": "assistant", "content": rsp}}


class ollama_reflector_adversarial_pointed(ollama_agent_service):
    def ollama_query(self, messages: list, temperature=0):
        self.model.create_context("default")
        relfection_prompt = "Analyze the user's answer to the question and provide feedback, highlighting any mistakes you find."
        rsp = self.model.reflect_adversarial(
            messages[0]["content"],
            relfection_prompt,
        )
        rsp = self.model.send(
            "Therefore, the answer to the question (formatted as JSON) is:"
        )
        return {"message": {"role": "assistant", "content": rsp}}
