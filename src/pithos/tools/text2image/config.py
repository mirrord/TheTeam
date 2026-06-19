"""Configuration model for the text2image tool.

The tool supports multiple local backends selected via :attr:`Text2ImageConfig.backend`:

- ``"diffusers"`` — in-process Hugging Face ``diffusers`` pipeline (needs ``torch``).
- ``"http"``      — HTTP call to a running Automatic1111/Forge server
                    (``/sdapi/v1/txt2img``); only needs ``requests``.
- ``"comfyui"``   — HTTP call to a running ComfyUI server (``/prompt`` +
                    ``/history`` + ``/view``); only needs ``requests``.

All generation parameters have sensible defaults so the agent only needs to
supply a prompt (``text2image <prompt text>``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class Text2ImageConfig:
    """Runtime configuration for image generation.

    Attributes:
        enabled: Whether the tool is active.
        backend: Which backend to use — ``"diffusers"`` or ``"http"``.
        output_dir: Directory where generated PNGs are written.
        model: Model identifier. For ``diffusers`` this is a HF repo id or local
            path; for ``http`` it is an optional checkpoint name (may be empty to
            use the server's currently loaded model).
        width: Output image width in pixels.
        height: Output image height in pixels.
        steps: Number of denoising/sampling steps.
        guidance_scale: Classifier-free guidance scale (prompt adherence).
        negative_prompt: Default negative prompt applied to every generation.
        seed: Fixed RNG seed, or ``None`` for a random seed each call.
        device: Torch device for the diffusers backend (``"cuda"``, ``"cpu"``...).
        sampler: Sampler/scheduler name (primarily for the http backend).
        base_url: Base URL of the http backend server.
        timeout: Per-request timeout (seconds) for the http backend.
        comfyui_base_url: Base URL of the ComfyUI server (``comfyui`` backend).
        comfyui_workflow_path: Path to a ComfyUI API-format workflow JSON whose
            input values may contain placeholder tokens (``%prompt%``,
            ``%negative_prompt%``, ``%seed%``, ``%steps%``, ``%cfg%``,
            ``%width%``, ``%height%``, ``%sampler%``, ``%model%``). Empty uses a
            built-in SD1.5 txt2img workflow.
    """

    enabled: bool = False
    backend: str = "http"
    output_dir: str = "./data/generated_images"
    model: str = ""
    width: int = 512
    height: int = 512
    steps: int = 30
    guidance_scale: float = 7.5
    negative_prompt: str = ""
    seed: Optional[int] = None
    device: str = "cuda"
    sampler: str = "Euler a"
    base_url: str = "http://127.0.0.1:7860"
    timeout: float = 120.0
    comfyui_base_url: str = "http://127.0.0.1:8188"
    comfyui_workflow_path: str = ""

    @classmethod
    def from_dict(cls, data: Optional[dict[str, Any]]) -> "Text2ImageConfig":
        """Build a config from a (possibly partial) dict, applying defaults.

        Unknown keys are ignored so the YAML can carry extra annotations without
        breaking construction.
        """
        if not data:
            return cls()
        known = set(cls.__dataclass_fields__)
        kwargs = {k: v for k, v in data.items() if k in known}
        return cls(**kwargs)
