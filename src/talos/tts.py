"""Kokoro-ONNX text-to-speech wrapper for Talos.

Wraps the ``kokoro-onnx`` package with the same ``TextToSpeechService`` API
the rest of Talos was built against (originally for Bark).  Model and voice
files are downloaded once into a Talos-managed cache directory so subsequent
runs are fully offline.
"""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np

# Silence phonemizer's cosmetic "words count mismatch" warning emitted via its
# logger whenever espeak's input/output word counts differ (contractions,
# numbers, punctuation, etc.).  Does not affect synthesized audio quality.
logging.getLogger("phonemizer").setLevel(logging.ERROR)

# Upstream model artefacts — pinned to the v1.0 release.
_MODEL_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
    "model-files-v1.0/kokoro-v1.0.onnx"
)
_VOICES_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
    "model-files-v1.0/voices-v1.0.bin"
)
_MODEL_FILENAME = "kokoro-v1.0.onnx"
_VOICES_FILENAME = "voices-v1.0.bin"

_DEFAULT_CACHE_DIR = Path.home() / ".talos" / "models" / "kokoro"
_DEFAULT_VOICE = "af_heart"
_SAMPLE_RATE = 24_000


def _download(url: str, dest: Path) -> None:
    """Download *url* to *dest* with a simple progress indicator."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"Downloading {dest.name} from {url} ...")

    last_pct = [-1]

    def _progress(block_num: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        pct = min(100, int(block_num * block_size * 100 / total_size))
        if pct != last_pct[0] and pct % 5 == 0:
            print(f"  {dest.name}: {pct}%", end="\r", flush=True)
            last_pct[0] = pct

    urllib.request.urlretrieve(url, tmp, _progress)
    tmp.replace(dest)
    print(f"  {dest.name}: done.    ")


class TextToSpeechService:
    """Kokoro-ONNX TTS service.

    Args:
        device: Kept for API compatibility with the previous Bark-based
            service.  ONNX Runtime selects its execution provider
            automatically; this argument is currently unused.
        model_cache: Directory in which model artefacts are cached.  Defaults
            to ``~/.talos/models/kokoro``.
    """

    def __init__(
        self,
        device: str = "cpu",
        model_cache: Optional[Path] = None,
    ) -> None:
        # Lazy import — keeps the rest of Talos importable when the optional
        # ``kokoro-onnx`` extra isn't installed.
        from kokoro_onnx import Kokoro  # type: ignore

        self.device = device
        cache = Path(model_cache) if model_cache is not None else _DEFAULT_CACHE_DIR
        cache.mkdir(parents=True, exist_ok=True)

        model_path = cache / _MODEL_FILENAME
        voices_path = cache / _VOICES_FILENAME

        if not model_path.exists():
            _download(_MODEL_URL, model_path)
        if not voices_path.exists():
            _download(_VOICES_URL, voices_path)

        self.kokoro = Kokoro(str(model_path), str(voices_path))
        self.sample_rate = _SAMPLE_RATE

    def _resolve_voice(self, voice_preset: str) -> str:
        """Map an unknown / legacy voice preset to a sensible default."""
        try:
            available = set(self.kokoro.get_voices())
        except Exception:
            return voice_preset
        if voice_preset in available:
            return voice_preset
        print(
            f"[talos.tts] voice {voice_preset!r} not available; "
            f"falling back to {_DEFAULT_VOICE!r}"
        )
        return _DEFAULT_VOICE

    def synthesize(
        self, text: str, voice_preset: str = _DEFAULT_VOICE
    ) -> tuple[int, np.ndarray]:
        """Synthesize *text* and return ``(sample_rate, audio_array)``."""
        voice = self._resolve_voice(voice_preset)
        samples, sample_rate = self.kokoro.create(
            text, voice=voice, speed=1.0, lang="en-us"
        )
        return sample_rate, np.asarray(samples)

    def long_form_synthesize(
        self, text: str, voice_preset: str = _DEFAULT_VOICE
    ) -> tuple[int, np.ndarray]:
        """Synthesize a longer passage.

        Kokoro handles internal phoneme batching, so we delegate directly to
        :meth:`synthesize`.  Method retained for API compatibility.
        """
        return self.synthesize(text, voice_preset)
