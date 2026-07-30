"""Adapts :class:`Prompt2ImageGenerator` into a virtual ``prompt2image`` tool.

The agent invokes the tool as ``prompt2image <prompt text>``: everything after the
leading token is treated as the prompt. Generation parameters (size, steps,
negative prompt, backend, ...) come from ``prompt2image`` config defaults.

Note: the agent-facing tool name is ``prompt2image`` rather than ``prompt2image``
to avoid colliding with the real ``prompt2image`` binary shipped by Tesseract's
OCR training tools, which some systems have on PATH and which ``CLIToolProvider``
would otherwise discover under the same name.

On success the tool returns a markdown summary including the saved file path and
generation metadata; on failure it returns a :class:`ToolResult` with an
``error_hint`` to guide the agent.
"""

from __future__ import annotations

import time
from typing import Any, Optional

from ...config_manager import ConfigManager
from ..models import ToolMetadata, ToolResult
from ..provider import ToolProvider
from .backends import Prompt2ImageError
from .config import Prompt2ImageConfig
from .generator import Prompt2ImageGenerator


class Prompt2ImageToolProvider(ToolProvider):
    """Virtual ``prompt2image`` tool backed by a local image model."""

    TOOL_NAME = "prompt2image"

    def __init__(
        self,
        config_manager: ConfigManager,
        config: Optional[Prompt2ImageConfig] = None,
        generator: Optional[Prompt2ImageGenerator] = None,
    ) -> None:
        self.config_manager = config_manager
        self._config = config
        self._generator = generator

    @property
    def config(self) -> Prompt2ImageConfig:
        """Lazily load the tool config from ``prompt2image_config.yaml``."""
        if self._config is None:
            raw = self.config_manager.get_config("prompt2image_config", "tools")
            self._config = Prompt2ImageConfig.from_dict(raw)
        return self._config

    @property
    def generator(self) -> Prompt2ImageGenerator:
        """Lazily construct the generator (and its backend)."""
        if self._generator is None:
            self._generator = Prompt2ImageGenerator(self.config)
        return self._generator

    def discover(self, platform: str = "cross-platform") -> dict[str, ToolMetadata]:
        """Return the metadata entry for this virtual tool."""
        return {
            self.TOOL_NAME: ToolMetadata(
                name=self.TOOL_NAME,
                path="",
                description=(
                    "Generate an image from a text prompt using a local "
                    "text-to-image model. Usage: prompt2image <prompt text>"
                ),
                platform=platform,
                source="virtual",
                tool_type="image",
            )
        }

    def can_execute(self, tool_name: str) -> bool:
        """Return True for the ``prompt2image`` tool name."""
        return tool_name == self.TOOL_NAME

    def execute(
        self,
        command: str,
        context: Optional[dict[str, Any]] = None,
    ) -> ToolResult:
        """Execute the prompt2image call extracted from *command*.

        Strips the leading ``prompt2image`` token and treats the rest as the prompt.
        """
        parts = command.strip().split(None, 1)
        prompt = parts[1].strip() if len(parts) > 1 else ""
        return self.run(prompt)

    def run(self, prompt: str) -> ToolResult:
        """Generate an image for *prompt* and wrap the result as a ToolResult."""
        start = time.time()
        command = f"prompt2image {prompt}"
        if not prompt or not prompt.strip():
            return ToolResult(
                success=False,
                stdout="",
                stderr="empty prompt",
                exit_code=-1,
                execution_time=0.0,
                command=command,
                error_hint="Usage: prompt2image <prompt text>",
            )
        try:
            metadata = self.generator.generate(prompt)
        except Prompt2ImageError as exc:
            return ToolResult(
                success=False,
                stdout="",
                stderr=f"image generation failed: {exc}",
                exit_code=-1,
                execution_time=time.time() - start,
                command=command,
                error_hint=(
                    "Check prompt2image_config.yaml: confirm the backend is reachable "
                    "and required extras are installed (.[image] for diffusers)."
                ),
            )
        except Exception as exc:
            return ToolResult(
                success=False,
                stdout="",
                stderr=f"image generation error: {exc}",
                exit_code=-1,
                execution_time=time.time() - start,
                command=command,
                error_hint="Unexpected error during image generation.",
            )
        return ToolResult(
            success=True,
            stdout=self._format_result(metadata),
            stderr="",
            exit_code=0,
            execution_time=time.time() - start,
            command=command,
            image_paths=[metadata["path"]],
        )

    @staticmethod
    def _format_result(metadata: dict[str, Any]) -> str:
        """Render generation metadata as a compact markdown summary."""
        lines = [
            "Image generated successfully.",
            f"- Path: {metadata.get('path')}",
            f"- Backend: {metadata.get('backend')}",
            f"- Model: {metadata.get('model')}",
            f"- Size: {metadata.get('width')}x{metadata.get('height')}",
            f"- Steps: {metadata.get('steps')}",
            f"- Seed: {metadata.get('seed')}",
            f"- Time: {metadata.get('elapsed')}s",
        ]
        return "\n".join(lines)
