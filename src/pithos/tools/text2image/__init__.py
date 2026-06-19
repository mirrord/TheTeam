"""text2image tool - generate images from text prompts with a local model.

The tool exposes a single virtual ``text2image`` tool that turns a prompt into a
PNG saved under ``data/generated_images``. The actual rendering is delegated to a
pluggable backend selected via config:

- ``diffusers`` — in-process Hugging Face pipeline (needs the ``image`` extra:
  ``torch`` + ``diffusers`` + ``Pillow``).
- ``http``      — Automatic1111/Forge HTTP API (needs ``requests`` from the
  ``web`` extra).
- ``comfyui``   — ComfyUI HTTP API (needs ``requests`` from the ``web`` extra).

Importing this package never fails when optional deps are missing; consumers
should check ``TEXT2IMAGE_AVAILABLE`` (or the per-backend flags) before
registering the provider.
"""

from .config import Text2ImageConfig

try:
    import requests  # noqa: F401

    HTTP_BACKEND_AVAILABLE = True
except ImportError:
    HTTP_BACKEND_AVAILABLE = False

try:
    import torch  # noqa: F401
    import diffusers  # noqa: F401

    DIFFUSERS_BACKEND_AVAILABLE = True
except ImportError:
    DIFFUSERS_BACKEND_AVAILABLE = False

# The tool is usable if at least one backend's dependencies are present.
TEXT2IMAGE_AVAILABLE = HTTP_BACKEND_AVAILABLE or DIFFUSERS_BACKEND_AVAILABLE


# Lazy imports for components that may pull in heavy deps.
def __getattr__(name):  # pragma: no cover - thin lazy-import shim
    if name == "Text2ImageGenerator":
        from .generator import Text2ImageGenerator

        return Text2ImageGenerator
    if name == "Text2ImageToolProvider":
        from .provider import Text2ImageToolProvider

        return Text2ImageToolProvider
    if name in {
        "ImageBackend",
        "DiffusersBackend",
        "HttpBackend",
        "ComfyUIBackend",
        "build_backend",
        "Text2ImageError",
    }:
        from . import backends

        return getattr(backends, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Text2ImageConfig",
    "Text2ImageGenerator",
    "Text2ImageToolProvider",
    "ImageBackend",
    "DiffusersBackend",
    "HttpBackend",
    "ComfyUIBackend",
    "build_backend",
    "Text2ImageError",
    "TEXT2IMAGE_AVAILABLE",
    "HTTP_BACKEND_AVAILABLE",
    "DIFFUSERS_BACKEND_AVAILABLE",
]
