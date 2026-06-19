"""Command-line entry point for the text2image tool.

Examples::

    pithos-text2image "a red fox in a snowy forest"
    pithos-text2image --backend comfyui --steps 20 "a glowing crystal cave"
    pithos-text2image --output-dir /tmp/imgs --seed 42 "a futuristic cityscape"
    pithos-text2image --config-dir ./my_configs "a cat on a windowsill"
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Optional

from ...config_manager import ConfigManager


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pithos-text2image",
        description=(
            "Generate an image from a text prompt using a local model and save "
            "it to disk. Backends: 'http' (Automatic1111/Forge), 'comfyui', "
            "or 'diffusers' (in-process HF pipeline)."
        ),
    )
    p.add_argument("prompt", nargs="+", help="The image generation prompt.")
    p.add_argument(
        "--backend",
        default=None,
        choices=["http", "comfyui", "diffusers"],
        help="Override the backend from config.",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        dest="output_dir",
        help="Override the output directory from config.",
    )
    p.add_argument(
        "--model",
        default=None,
        help="Override the model identifier from config.",
    )
    p.add_argument(
        "--width",
        type=int,
        default=None,
        help="Override output width in pixels.",
    )
    p.add_argument(
        "--height",
        type=int,
        default=None,
        help="Override output height in pixels.",
    )
    p.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Override number of denoising steps.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Fixed RNG seed for reproducibility.",
    )
    p.add_argument(
        "--negative-prompt",
        default=None,
        dest="negative_prompt",
        help="Override the negative prompt from config.",
    )
    p.add_argument(
        "--config-dir",
        default=None,
        dest="config_dir",
        help="Override the configs/ directory (defaults to project configs/).",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress non-error logging.",
    )
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    from . import TEXT2IMAGE_AVAILABLE

    if not TEXT2IMAGE_AVAILABLE:
        print(
            "text2image backends unavailable: install at least one of:\n"
            "  pip install -e .[web]    # http / comfyui backends\n"
            "  pip install -e .[image]  # diffusers backend",
            file=sys.stderr,
        )
        return 2

    cm = ConfigManager(args.config_dir) if args.config_dir else ConfigManager()

    from .config import Text2ImageConfig

    raw = cm.get_config("text2image_config", "tools") or {}
    config = Text2ImageConfig.from_dict(raw)

    # Apply CLI overrides on top of config-file values.
    if args.backend is not None:
        config.backend = args.backend
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    if args.model is not None:
        config.model = args.model
    if args.width is not None:
        config.width = args.width
    if args.height is not None:
        config.height = args.height
    if args.steps is not None:
        config.steps = args.steps
    if args.seed is not None:
        config.seed = args.seed
    if args.negative_prompt is not None:
        config.negative_prompt = args.negative_prompt

    prompt = " ".join(args.prompt).strip()
    if not prompt:
        print("prompt must not be empty", file=sys.stderr)
        return 2

    from .generator import Text2ImageGenerator

    generator = Text2ImageGenerator(config)
    try:
        metadata = generator.generate(prompt)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"image generation failed: {exc}", file=sys.stderr)
        return 1

    print(f"Saved: {metadata['path']}")
    print(f"Backend: {metadata.get('backend')}  Model: {metadata.get('model')}")
    print(
        f"Size: {metadata.get('width')}x{metadata.get('height')}  "
        f"Steps: {metadata.get('steps')}  Seed: {metadata.get('seed')}  "
        f"Time: {metadata.get('elapsed')}s"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
