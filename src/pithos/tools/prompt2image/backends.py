"""Pluggable image-generation backends for the prompt2image tool.

A backend takes a text prompt plus generation parameters and returns rendered
PNG bytes together with metadata describing how the image was produced.

Two backends ship by default:

- :class:`DiffusersBackend` — loads a Hugging Face ``diffusers`` pipeline in
  process. Heavy (pulls in ``torch``); gated behind the optional ``image`` extra.
- :class:`HttpBackend` — POSTs to a running Automatic1111/Forge server at
  ``/sdapi/v1/txt2img``. Only needs ``requests``.
- :class:`ComfyUIBackend` — submits a node-graph workflow to a running ComfyUI
  server (``/prompt`` + ``/history`` + ``/view``). Only needs ``requests``.

Both raise :class:`Prompt2ImageError` on failure so the provider can surface a
clean error to the agent. Heavy third-party imports happen lazily inside the
backends so that importing this module never fails when optional deps are
missing.
"""

from __future__ import annotations

import base64
import json
import random
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .config import Prompt2ImageConfig


class Prompt2ImageError(RuntimeError):
    """Raised when an image backend fails to produce an image."""


@dataclass
class GenerationParams:
    """Per-call generation parameters derived from config (and the prompt)."""

    prompt: str
    negative_prompt: str = ""
    width: int = 512
    height: int = 512
    steps: int = 30
    guidance_scale: float = 7.5
    seed: Optional[int] = None
    sampler: str = "Euler a"
    model: str = ""

    @classmethod
    def from_config(cls, prompt: str, config: Prompt2ImageConfig) -> "GenerationParams":
        """Build params from a prompt and the tool config defaults."""
        return cls(
            prompt=prompt,
            negative_prompt=config.negative_prompt,
            width=config.width,
            height=config.height,
            steps=config.steps,
            guidance_scale=config.guidance_scale,
            seed=config.seed,
            sampler=config.sampler,
            model=config.model,
        )


@dataclass
class GeneratedImage:
    """A rendered image plus metadata describing its generation."""

    png_bytes: bytes
    metadata: dict[str, Any] = field(default_factory=dict)


def _resolve_seed(seed: Optional[int]) -> int:
    """Return *seed* if set, otherwise a random non-negative 32-bit seed."""
    if seed is not None and seed >= 0:
        return seed
    return random.randint(0, 2**32 - 1)


class ImageBackend(ABC):
    """Interface every prompt2image backend must implement."""

    name: str = "base"

    @abstractmethod
    def generate(self, params: GenerationParams) -> GeneratedImage:
        """Render an image for *params* and return PNG bytes + metadata.

        Raises:
            Prompt2ImageError: If generation fails for any reason.
        """


class DiffusersBackend(ImageBackend):
    """In-process Hugging Face ``diffusers`` Stable Diffusion backend."""

    name = "diffusers"

    def __init__(self, config: Prompt2ImageConfig) -> None:
        self.config = config
        self._pipe: Any = None

    def _load_pipeline(self) -> Any:
        """Lazily load and cache the diffusion pipeline.

        Raises:
            Prompt2ImageError: If ``diffusers``/``torch`` are unavailable or the
                model cannot be loaded.
        """
        if self._pipe is not None:
            return self._pipe
        if not self.config.model:
            raise Prompt2ImageError(
                "diffusers backend requires 'model' (a HF repo id or local path) "
                "to be set in prompt2image config."
            )
        try:
            import torch
            from diffusers import AutoPipelineForPrompt2Image
        except ImportError as exc:  # pragma: no cover - exercised via skip
            raise Prompt2ImageError(
                "diffusers backend unavailable: install with 'pip install -e .[image]'"
            ) from exc
        try:
            dtype = (
                torch.float16
                if self.config.device.startswith("cuda")
                else torch.float32
            )
            pipe = AutoPipelineForPrompt2Image.from_pretrained(
                self.config.model, torch_dtype=dtype
            )
            pipe = pipe.to(self.config.device)
        except Exception as exc:
            raise Prompt2ImageError(
                f"failed to load model '{self.config.model}': {exc}"
            ) from exc
        self._pipe = pipe
        return pipe

    def generate(self, params: GenerationParams) -> GeneratedImage:
        import io

        pipe = self._load_pipeline()
        seed = _resolve_seed(params.seed)
        try:
            import torch

            generator = torch.Generator(device=self.config.device).manual_seed(seed)
            result = pipe(
                prompt=params.prompt,
                negative_prompt=params.negative_prompt or None,
                width=params.width,
                height=params.height,
                num_inference_steps=params.steps,
                guidance_scale=params.guidance_scale,
                generator=generator,
            )
            image = result.images[0]
        except Prompt2ImageError:
            raise
        except Exception as exc:
            raise Prompt2ImageError(f"diffusers generation failed: {exc}") from exc

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return GeneratedImage(
            png_bytes=buffer.getvalue(),
            metadata={
                "backend": self.name,
                "model": self.config.model,
                "seed": seed,
                "width": params.width,
                "height": params.height,
                "steps": params.steps,
                "guidance_scale": params.guidance_scale,
            },
        )


class HttpBackend(ImageBackend):
    """Automatic1111/Forge HTTP backend (``POST /sdapi/v1/txt2img``)."""

    name = "http"

    def __init__(self, config: Prompt2ImageConfig) -> None:
        self.config = config

    def generate(self, params: GenerationParams) -> GeneratedImage:
        try:
            import requests
        except ImportError as exc:  # pragma: no cover - exercised via skip
            raise Prompt2ImageError(
                "http backend unavailable: install with 'pip install -e .[web]'"
            ) from exc

        seed = _resolve_seed(params.seed)
        payload: dict[str, Any] = {
            "prompt": params.prompt,
            "negative_prompt": params.negative_prompt,
            "width": params.width,
            "height": params.height,
            "steps": params.steps,
            "cfg_scale": params.guidance_scale,
            "sampler_name": params.sampler,
            "seed": seed,
        }
        url = self.config.base_url.rstrip("/") + "/sdapi/v1/txt2img"
        try:
            response = requests.post(url, json=payload, timeout=self.config.timeout)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            raise Prompt2ImageError(
                f"http backend request to {url} failed: {exc}"
            ) from exc

        images = data.get("images") or []
        if not images:
            raise Prompt2ImageError("http backend returned no images")
        try:
            png_bytes = base64.b64decode(images[0])
        except Exception as exc:
            raise Prompt2ImageError(f"failed to decode image data: {exc}") from exc

        return GeneratedImage(
            png_bytes=png_bytes,
            metadata={
                "backend": self.name,
                "model": params.model or "(server default)",
                "seed": seed,
                "width": params.width,
                "height": params.height,
                "steps": params.steps,
                "guidance_scale": params.guidance_scale,
                "sampler": params.sampler,
            },
        )


# Built-in ComfyUI API-format workflow (SD1.5 txt2img). Input values use
# placeholder tokens that ComfyUIBackend substitutes per call.
_DEFAULT_COMFYUI_WORKFLOW: dict[str, Any] = {
    "4": {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {"ckpt_name": "%model%"},
    },
    "5": {
        "class_type": "EmptyLatentImage",
        "inputs": {"width": "%width%", "height": "%height%", "batch_size": 1},
    },
    "6": {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": "%prompt%", "clip": ["4", 1]},
    },
    "7": {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": "%negative_prompt%", "clip": ["4", 1]},
    },
    "3": {
        "class_type": "KSampler",
        "inputs": {
            "seed": "%seed%",
            "steps": "%steps%",
            "cfg": "%cfg%",
            "sampler_name": "%sampler%",
            "scheduler": "normal",
            "denoise": 1.0,
            "model": ["4", 0],
            "positive": ["6", 0],
            "negative": ["7", 0],
            "latent_image": ["5", 0],
        },
    },
    "8": {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["3", 0], "vae": ["4", 2]},
    },
    "9": {
        "class_type": "SaveImage",
        "inputs": {"filename_prefix": "theteam", "images": ["8", 0]},
    },
}


class ComfyUIBackend(ImageBackend):
    """ComfyUI HTTP backend (``/prompt`` → poll ``/history`` → fetch ``/view``).

    Submits a node-graph workflow (ComfyUI "API format") with per-call values
    substituted into placeholder tokens, waits for the run to finish, then
    downloads the first produced image.
    """

    name = "comfyui"

    def __init__(self, config: Prompt2ImageConfig) -> None:
        self.config = config

    def _load_workflow(self) -> dict[str, Any]:
        """Return the workflow template (from file if configured, else built-in)."""
        path = (self.config.comfyui_workflow_path or "").strip()
        if not path:
            return json.loads(json.dumps(_DEFAULT_COMFYUI_WORKFLOW))  # deep copy
        try:
            return json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception as exc:
            raise Prompt2ImageError(
                f"failed to load ComfyUI workflow '{path}': {exc}"
            ) from exc

    @staticmethod
    def _substitute(
        workflow: dict[str, Any],
        tokens: dict[str, Any],
        variables: dict[str, Any],
    ) -> dict[str, Any]:
        """Fill workflow input values with per-call generation values.

        Two placeholder styles are supported inside node input values:

        - Exact ``%token%`` values (e.g. ``"%steps%"``) are replaced with the
          token's typed value, preserving ints/floats.
        - ``{name}`` variables embedded in a string (e.g. a ComfyUI
          ``StringFormat`` node's ``"masterpiece, {prompt}"``) are substituted
          textually with ``str(value)``. Only known variable names are replaced;
          any other braces are left untouched.
        """
        for node in workflow.values():
            inputs = node.get("inputs")
            if not isinstance(inputs, dict):
                continue
            for key, value in list(inputs.items()):
                if not isinstance(value, str):
                    continue
                if value in tokens:
                    inputs[key] = tokens[value]
                    continue
                if "{" in value:
                    for name, replacement in variables.items():
                        placeholder = "{" + name + "}"
                        if placeholder in value:
                            value = value.replace(placeholder, str(replacement))
                    inputs[key] = value
        return workflow

    def generate(self, params: GenerationParams) -> GeneratedImage:
        try:
            import requests
        except ImportError as exc:  # pragma: no cover - exercised via skip
            raise Prompt2ImageError(
                "comfyui backend unavailable: install with 'pip install -e .[web]'"
            ) from exc

        # Guard: the built-in workflow requires a checkpoint name.
        using_builtin = not (self.config.comfyui_workflow_path or "").strip()
        if using_builtin and not (params.model or "").strip():
            raise Prompt2ImageError(
                "comfyui backend: 'model' must be set to a checkpoint filename "
                "when using the built-in workflow (e.g. 'v1-5-pruned-emaonly.safetensors'). "
                "Set 'model' in prompt2image_config.yaml or use --model on the CLI."
            )

        seed = _resolve_seed(params.seed)
        tokens: dict[str, Any] = {
            "%prompt%": params.prompt,
            "%negative_prompt%": params.negative_prompt,
            "%seed%": seed,
            "%steps%": params.steps,
            "%cfg%": params.guidance_scale,
            "%width%": params.width,
            "%height%": params.height,
            "%sampler%": params.sampler,
            "%model%": params.model,
        }
        # Named variables for `{name}` substitution inside string inputs (e.g. a
        # ComfyUI StringFormat node's f_string containing "... {prompt}").
        variables: dict[str, Any] = {
            "prompt": params.prompt,
            "negative_prompt": params.negative_prompt,
            "seed": seed,
            "steps": params.steps,
            "cfg": params.guidance_scale,
            "width": params.width,
            "height": params.height,
            "sampler": params.sampler,
            "model": params.model,
        }
        workflow = self._substitute(self._load_workflow(), tokens, variables)

        base = self.config.comfyui_base_url.rstrip("/")
        # Use standard dashed UUID format; some ComfyUI versions validate the format.
        client_id = str(uuid.uuid4())
        try:
            resp = requests.post(
                f"{base}/prompt",
                json={"prompt": workflow, "client_id": client_id},
                timeout=self.config.timeout,
            )
        except requests.exceptions.ConnectionError as exc:
            raise Prompt2ImageError(
                f"comfyui: could not reach server at {base} — "
                f"check that ComfyUI is running and 'comfyui_base_url' is correct. "
                f"Detail: {exc}"
            ) from exc
        except Exception as exc:
            raise Prompt2ImageError(
                f"comfyui /prompt request to {base} failed: {exc}"
            ) from exc

        # Surface ComfyUI's validation error body when the server returns one.
        if not resp.ok:
            try:
                body = resp.json()
            except Exception:
                body = resp.text
            raise Prompt2ImageError(
                f"comfyui /prompt returned HTTP {resp.status_code}: {body}"
            )

        data = resp.json()
        # ComfyUI returns HTTP 200 with an 'error' key when workflow validation fails.
        if "error" in data:
            node_errors = data.get("node_errors", {})
            raise Prompt2ImageError(
                f"comfyui workflow validation failed: {data['error']}. "
                f"Node errors: {node_errors}"
            )

        prompt_id = data.get("prompt_id")
        if not prompt_id:
            raise Prompt2ImageError(
                f"comfyui did not return a prompt_id. Response: {data}"
            )

        history = self._wait_for_history(requests, base, prompt_id)
        image_ref = self._first_image_ref(history)
        if image_ref is None:
            raise Prompt2ImageError("comfyui run produced no images")

        try:
            view = requests.get(
                f"{base}/view", params=image_ref, timeout=self.config.timeout
            )
            view.raise_for_status()
            png_bytes = view.content
        except Exception as exc:
            raise Prompt2ImageError(f"comfyui /view request failed: {exc}") from exc

        return GeneratedImage(
            png_bytes=png_bytes,
            metadata={
                "backend": self.name,
                "model": params.model or "(workflow default)",
                "seed": seed,
                "width": params.width,
                "height": params.height,
                "steps": params.steps,
                "guidance_scale": params.guidance_scale,
                "sampler": params.sampler,
                "prompt_id": prompt_id,
            },
        )

    def _wait_for_history(
        self, requests: Any, base: str, prompt_id: str
    ) -> dict[str, Any]:
        """Poll ``/history/{prompt_id}`` until the run appears or timeout elapses."""
        deadline = time.time() + self.config.timeout
        poll_interval = 0.5
        while time.time() < deadline:
            try:
                resp = requests.get(
                    f"{base}/history/{prompt_id}", timeout=self.config.timeout
                )
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:
                raise Prompt2ImageError(
                    f"comfyui /history request failed: {exc}"
                ) from exc
            entry = data.get(prompt_id)
            if entry:
                return entry
            time.sleep(poll_interval)
        raise Prompt2ImageError(
            f"comfyui run timed out after {self.config.timeout}s waiting for results"
        )

    @staticmethod
    def _first_image_ref(history_entry: dict[str, Any]) -> Optional[dict[str, str]]:
        """Return ``/view`` query params for the first image in the outputs."""
        outputs = history_entry.get("outputs") or {}
        for node_output in outputs.values():
            images = (
                node_output.get("images") if isinstance(node_output, dict) else None
            )
            if not images:
                continue
            first = images[0]
            return {
                "filename": first.get("filename", ""),
                "subfolder": first.get("subfolder", ""),
                "type": first.get("type", "output"),
            }
        return None


def build_backend(config: Prompt2ImageConfig) -> ImageBackend:
    """Construct the backend selected by ``config.backend``.

    Raises:
        Prompt2ImageError: If the backend name is not recognised.
    """
    backend = (config.backend or "").strip().lower()
    if backend == "diffusers":
        return DiffusersBackend(config)
    if backend == "http":
        return HttpBackend(config)
    if backend == "comfyui":
        return ComfyUIBackend(config)
    raise Prompt2ImageError(
        f"unknown prompt2image backend '{config.backend}'; "
        "expected 'diffusers', 'http' or 'comfyui'"
    )
