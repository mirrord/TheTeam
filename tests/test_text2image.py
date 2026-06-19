"""Tests for the text2image tool (config, backends, generator, provider).

External image backends (diffusers/torch and the A1111 HTTP server) are mocked
so the tests run fast and offline. A 1x1 PNG byte string stands in for real
rendered output.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Optional
from unittest.mock import Mock

import pytest

from pithos.tools.text2image.backends import (
    ComfyUIBackend,
    GeneratedImage,
    GenerationParams,
    HttpBackend,
    ImageBackend,
    Text2ImageError,
    build_backend,
)
from pithos.tools.text2image.config import Text2ImageConfig
from pithos.tools.text2image.generator import Text2ImageGenerator
from pithos.tools.text2image.provider import Text2ImageToolProvider

# Minimal valid 1x1 transparent PNG.
_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


class _FakeBackend(ImageBackend):
    """Backend that returns fixed PNG bytes and records the params it received."""

    name = "fake"

    def __init__(self) -> None:
        self.last_params: Optional[GenerationParams] = None

    def generate(self, params: GenerationParams) -> GeneratedImage:
        self.last_params = params
        return GeneratedImage(
            png_bytes=_PNG_BYTES,
            metadata={
                "backend": self.name,
                "model": params.model,
                "seed": params.seed if params.seed is not None else 123,
                "width": params.width,
                "height": params.height,
                "steps": params.steps,
                "guidance_scale": params.guidance_scale,
            },
        )


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


class TestText2ImageConfig:
    def test_defaults(self) -> None:
        cfg = Text2ImageConfig()
        assert cfg.enabled is False
        assert cfg.backend == "http"
        assert cfg.output_dir == "./data/generated_images"
        assert cfg.width == 512 and cfg.height == 512

    def test_from_dict_partial_applies_defaults(self) -> None:
        cfg = Text2ImageConfig.from_dict({"backend": "diffusers", "steps": 10})
        assert cfg.backend == "diffusers"
        assert cfg.steps == 10
        assert cfg.guidance_scale == 7.5  # default preserved

    def test_from_dict_ignores_unknown_keys(self) -> None:
        cfg = Text2ImageConfig.from_dict({"backend": "http", "bogus": 1})
        assert cfg.backend == "http"
        assert not hasattr(cfg, "bogus")

    def test_from_dict_none(self) -> None:
        assert Text2ImageConfig.from_dict(None) == Text2ImageConfig()


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #


class TestBackendFactory:
    def test_build_http_backend(self) -> None:
        backend = build_backend(Text2ImageConfig(backend="http"))
        assert isinstance(backend, HttpBackend)

    def test_build_unknown_backend_raises(self) -> None:
        with pytest.raises(Text2ImageError):
            build_backend(Text2ImageConfig(backend="nope"))

    def test_params_from_config(self) -> None:
        cfg = Text2ImageConfig(width=256, height=256, steps=12, seed=7)
        params = GenerationParams.from_config("a cat", cfg)
        assert params.prompt == "a cat"
        assert params.width == 256 and params.steps == 12 and params.seed == 7


class TestHttpBackend:
    def _config(self) -> Text2ImageConfig:
        return Text2ImageConfig(backend="http", base_url="http://localhost:7860")

    def test_generate_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        encoded = base64.b64encode(_PNG_BYTES).decode("ascii")
        response = Mock()
        response.raise_for_status = Mock()
        response.json = Mock(return_value={"images": [encoded]})
        requests_mod = Mock()
        requests_mod.post = Mock(return_value=response)
        monkeypatch.setitem(__import__("sys").modules, "requests", requests_mod)

        backend = HttpBackend(self._config())
        params = GenerationParams.from_config("a dog", self._config())
        result = backend.generate(params)

        assert result.png_bytes == _PNG_BYTES
        assert result.metadata["backend"] == "http"
        # Endpoint and payload sanity.
        called_url = requests_mod.post.call_args[0][0]
        assert called_url.endswith("/sdapi/v1/txt2img")
        payload = requests_mod.post.call_args[1]["json"]
        assert payload["prompt"] == "a dog"

    def test_generate_no_images_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        response = Mock()
        response.raise_for_status = Mock()
        response.json = Mock(return_value={"images": []})
        requests_mod = Mock()
        requests_mod.post = Mock(return_value=response)
        monkeypatch.setitem(__import__("sys").modules, "requests", requests_mod)

        backend = HttpBackend(self._config())
        params = GenerationParams.from_config("x", self._config())
        with pytest.raises(Text2ImageError):
            backend.generate(params)

    def test_generate_request_error_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        requests_mod = Mock()
        requests_mod.post = Mock(side_effect=RuntimeError("connection refused"))
        monkeypatch.setitem(__import__("sys").modules, "requests", requests_mod)

        backend = HttpBackend(self._config())
        params = GenerationParams.from_config("x", self._config())
        with pytest.raises(Text2ImageError):
            backend.generate(params)


class TestComfyUIBackend:
    def _config(self) -> Text2ImageConfig:
        return Text2ImageConfig(
            backend="comfyui",
            comfyui_base_url="http://localhost:8188",
            model="sd15.safetensors",
            timeout=5,
        )

    def _install_requests(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        prompt_id: str = "pid-1",
        history: Optional[dict] = None,
        view_content: bytes = _PNG_BYTES,
    ) -> Mock:
        """Install a fake `requests` module that emulates ComfyUI endpoints."""
        post_resp = Mock()
        post_resp.raise_for_status = Mock()
        post_resp.json = Mock(return_value={"prompt_id": prompt_id})

        if history is None:
            history = {
                prompt_id: {
                    "outputs": {
                        "9": {
                            "images": [
                                {
                                    "filename": "theteam_001.png",
                                    "subfolder": "",
                                    "type": "output",
                                }
                            ]
                        }
                    }
                }
            }

        def _get(url: str, **kwargs: Any) -> Mock:
            resp = Mock()
            resp.raise_for_status = Mock()
            if "/history/" in url:
                resp.json = Mock(return_value=history)
            elif url.endswith("/view"):
                resp.content = view_content
            return resp

        requests_mod = Mock()
        requests_mod.post = Mock(return_value=post_resp)
        requests_mod.get = Mock(side_effect=_get)
        monkeypatch.setitem(__import__("sys").modules, "requests", requests_mod)
        # Avoid real sleeps during history polling.
        monkeypatch.setattr(
            "pithos.tools.text2image.backends.time.sleep", lambda *_: None
        )
        return requests_mod

    def test_factory_builds_comfyui(self) -> None:
        assert isinstance(build_backend(self._config()), ComfyUIBackend)

    def test_generate_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        requests_mod = self._install_requests(monkeypatch)
        backend = ComfyUIBackend(self._config())
        params = GenerationParams.from_config("a fox", self._config())

        result = backend.generate(params)

        assert result.png_bytes == _PNG_BYTES
        assert result.metadata["backend"] == "comfyui"
        assert result.metadata["prompt_id"] == "pid-1"
        # /prompt received a substituted workflow (no placeholder tokens remain).
        submitted = requests_mod.post.call_args[1]["json"]["prompt"]
        prompt_node = submitted["6"]["inputs"]["text"]
        assert prompt_node == "a fox"
        assert submitted["3"]["inputs"]["steps"] == params.steps
        # Endpoint sanity.
        assert requests_mod.post.call_args[0][0].endswith("/prompt")

    def test_workflow_placeholders_fully_substituted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        requests_mod = self._install_requests(monkeypatch)
        backend = ComfyUIBackend(self._config())
        backend.generate(GenerationParams.from_config("hi", self._config()))
        submitted = requests_mod.post.call_args[1]["json"]["prompt"]
        # No remaining %...% placeholder strings anywhere in the inputs.
        leftover = [
            v
            for node in submitted.values()
            for v in node.get("inputs", {}).values()
            if isinstance(v, str) and v.startswith("%") and v.endswith("%")
        ]
        assert leftover == []

    def test_generate_no_images_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._install_requests(monkeypatch, history={"pid-1": {"outputs": {}}})
        backend = ComfyUIBackend(self._config())
        with pytest.raises(Text2ImageError):
            backend.generate(GenerationParams.from_config("x", self._config()))

    def test_generate_no_prompt_id_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        post_resp = Mock()
        post_resp.raise_for_status = Mock()
        post_resp.json = Mock(return_value={})
        requests_mod = Mock()
        requests_mod.post = Mock(return_value=post_resp)
        monkeypatch.setitem(__import__("sys").modules, "requests", requests_mod)
        backend = ComfyUIBackend(self._config())
        with pytest.raises(Text2ImageError):
            backend.generate(GenerationParams.from_config("x", self._config()))

    def test_timeout_when_history_never_ready(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._install_requests(monkeypatch, history={})  # prompt_id never appears
        cfg = self._config()
        cfg.timeout = 0  # force immediate deadline
        backend = ComfyUIBackend(cfg)
        with pytest.raises(Text2ImageError):
            backend.generate(GenerationParams.from_config("x", cfg))

    def test_custom_workflow_loaded_from_file(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        import json

        wf = {
            "1": {"class_type": "CLIPTextEncode", "inputs": {"text": "%prompt%"}},
            "2": {"class_type": "SaveImage", "inputs": {"images": ["1", 0]}},
        }
        wf_path = tmp_path / "wf.json"
        wf_path.write_text(json.dumps(wf), encoding="utf-8")

        requests_mod = self._install_requests(monkeypatch)
        cfg = self._config()
        cfg.comfyui_workflow_path = str(wf_path)
        backend = ComfyUIBackend(cfg)
        backend.generate(GenerationParams.from_config("custom", cfg))

        submitted = requests_mod.post.call_args[1]["json"]["prompt"]
        assert submitted["1"]["inputs"]["text"] == "custom"

    def test_bad_workflow_path_raises(self, tmp_path: Path) -> None:
        cfg = self._config()
        cfg.comfyui_workflow_path = str(tmp_path / "missing.json")
        backend = ComfyUIBackend(cfg)
        with pytest.raises(Text2ImageError):
            backend.generate(GenerationParams.from_config("x", cfg))


# --------------------------------------------------------------------------- #
# Generator
# --------------------------------------------------------------------------- #


class TestText2ImageGenerator:
    def test_generate_writes_file_and_returns_metadata(self, tmp_path: Path) -> None:
        cfg = Text2ImageConfig(output_dir=str(tmp_path / "imgs"), model="m")
        backend = _FakeBackend()
        gen = Text2ImageGenerator(cfg, backend=backend)

        meta = gen.generate("A red cube on grass")

        saved = Path(meta["path"])
        assert saved.exists()
        assert saved.suffix == ".png"
        assert saved.read_bytes() == _PNG_BYTES
        assert meta["prompt"] == "A red cube on grass"
        assert meta["backend"] == "fake"
        assert "elapsed" in meta
        # Slug is derived from the prompt.
        assert "red-cube-on-grass" in saved.name

    def test_generate_empty_prompt_raises(self, tmp_path: Path) -> None:
        cfg = Text2ImageConfig(output_dir=str(tmp_path))
        gen = Text2ImageGenerator(cfg, backend=_FakeBackend())
        with pytest.raises(ValueError):
            gen.generate("   ")

    def test_generate_creates_output_dir(self, tmp_path: Path) -> None:
        nested = tmp_path / "a" / "b" / "c"
        cfg = Text2ImageConfig(output_dir=str(nested))
        gen = Text2ImageGenerator(cfg, backend=_FakeBackend())
        meta = gen.generate("hello world")
        assert Path(meta["path"]).parent == nested

    def test_lazy_backend_built_from_config(self) -> None:
        gen = Text2ImageGenerator(Text2ImageConfig(backend="http"))
        assert isinstance(gen.backend, HttpBackend)


# --------------------------------------------------------------------------- #
# Provider
# --------------------------------------------------------------------------- #


class TestText2ImageToolProvider:
    def _provider(self, gen: Optional[Any] = None) -> Text2ImageToolProvider:
        cm = Mock()
        return Text2ImageToolProvider(
            config_manager=cm,
            config=Text2ImageConfig(),
            generator=gen,
        )

    def test_discover_metadata(self) -> None:
        meta = self._provider().discover()
        assert "text2image" in meta
        entry = meta["text2image"]
        assert entry.tool_type == "image"
        assert entry.source == "virtual"

    def test_can_execute(self) -> None:
        provider = self._provider()
        assert provider.can_execute("text2image") is True
        assert provider.can_execute("python") is False

    def test_execute_success(self) -> None:
        gen = Mock()
        gen.generate = Mock(
            return_value={
                "path": "/tmp/out.png",
                "backend": "fake",
                "model": "m",
                "width": 512,
                "height": 512,
                "steps": 30,
                "seed": 1,
                "elapsed": 0.1,
            }
        )
        provider = self._provider(gen=gen)
        result = provider.execute("text2image a blue sphere")

        assert result.success is True
        assert "Image generated successfully" in result.stdout
        assert "/tmp/out.png" in result.stdout
        gen.generate.assert_called_once_with("a blue sphere")

    def test_execute_empty_prompt(self) -> None:
        result = self._provider(gen=Mock()).execute("text2image")
        assert result.success is False
        assert result.error_hint is not None

    def test_execute_backend_error(self) -> None:
        gen = Mock()
        gen.generate = Mock(side_effect=Text2ImageError("server down"))
        result = self._provider(gen=gen).execute("text2image a tree")
        assert result.success is False
        assert "server down" in result.stderr
        assert result.error_hint is not None

    def test_execute_unexpected_error(self) -> None:
        gen = Mock()
        gen.generate = Mock(side_effect=RuntimeError("boom"))
        result = self._provider(gen=gen).execute("text2image a tree")
        assert result.success is False
        assert "boom" in result.stderr

    def test_config_lazy_loaded_from_config_manager(self) -> None:
        cm = Mock()
        cm.get_config = Mock(return_value={"backend": "diffusers", "steps": 5})
        provider = Text2ImageToolProvider(config_manager=cm)
        assert provider.config.backend == "diffusers"
        assert provider.config.steps == 5
        cm.get_config.assert_called_once_with("text2image_config", "tools")
