"""Orchestration layer for the text2image tool.

:class:`Text2ImageGenerator` ties a :class:`~pithos.tools.text2image.config.Text2ImageConfig`
to a backend, runs generation, writes the resulting PNG to disk and returns the
file path plus metadata. The backend is created lazily so importing this module
(and constructing the generator) is cheap and never pulls in heavy deps.
"""

from __future__ import annotations

import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .backends import (
    GenerationParams,
    ImageBackend,
    build_backend,
)
from .config import Text2ImageConfig

_MAX_SLUG_LEN = 40


def _slugify(prompt: str) -> str:
    """Return a short filesystem-safe slug derived from *prompt*."""
    lowered = prompt.strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "-", lowered).strip("-")
    if not slug:
        slug = "image"
    return slug[:_MAX_SLUG_LEN].strip("-") or "image"


class Text2ImageGenerator:
    """Generate images from text prompts and persist them to disk."""

    def __init__(
        self,
        config: Text2ImageConfig,
        backend: Optional[ImageBackend] = None,
    ) -> None:
        self.config = config
        self._backend = backend

    @property
    def backend(self) -> ImageBackend:
        """Lazily construct (and cache) the configured backend."""
        if self._backend is None:
            self._backend = build_backend(self.config)
        return self._backend

    def generate(self, prompt: str) -> dict[str, Any]:
        """Generate an image for *prompt* and save it under ``output_dir``.

        Args:
            prompt: The text prompt. Must be non-empty.

        Returns:
            A metadata dict including ``path`` (the saved PNG), generation
            parameters, ``seed``, ``backend``, ``model`` and ``elapsed``.

        Raises:
            ValueError: If *prompt* is empty.
            Text2ImageError: If the backend fails to produce an image.
        """
        if not prompt or not prompt.strip():
            raise ValueError("prompt must not be empty")

        prompt = prompt.strip()
        params = GenerationParams.from_config(prompt, self.config)

        start = time.time()
        image = self.backend.generate(params)
        elapsed = time.time() - start

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"{timestamp}_{_slugify(prompt)}.png"
        path = output_dir / filename
        path.write_bytes(image.png_bytes)

        metadata: dict[str, Any] = {
            "path": str(path),
            "prompt": prompt,
            "elapsed": round(elapsed, 3),
        }
        metadata.update(image.metadata)
        return metadata
