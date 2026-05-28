"""Voice interface — wake-word triggered speech-to-speech using whisper STT
for both wake-word detection and command transcription, pithos agent, and
Kokoro TTS.

Wake-word detection works by continuously recording short audio chunks and
transcribing them with whisper; the configured wake-word phrase is matched
(case-insensitive substring) against the transcript.  This avoids the
additional dependency footprint of a dedicated keyword-spotting model.

All heavyweight dependencies are imported lazily so the rest of the Talos
package remains importable without them.
"""

from __future__ import annotations

import logging
import threading
import time
import string
from queue import Queue
from typing import Any

from pithos.agent import Agent

from ..config import VoiceConfig
from ..utils import clean_agent_response

logger = logging.getLogger(__name__)

VOICE_CONTEXT_NAME = "voice"
VOICE_PROMPT_PREFIX = "Respond in 20 words or less. "

# Whisper expects 16 kHz mono int16 audio.
SAMPLE_RATE = 16_000

# Notification tone parameters (generated programmatically — no asset file needed).
_TONE_FREQ_HZ = 440.0
_TONE_DURATION_S = 0.3
_TONE_AMPLITUDE = 0.5


def _require(module: str, extra: str = "talos") -> None:
    """Raise an informative ImportError when an optional dep is missing."""
    raise ImportError(
        f"Talos voice interface requires '{module}'. "
        f"Install voice extras with: pip install -e .[{extra}]"
    )


class VoiceInterface:
    """Always-on wake-word listener that records, transcribes, responds, speaks."""

    def __init__(self, agent: Agent, config: VoiceConfig) -> None:
        self.agent = agent
        self.config = config
        self._stop = threading.Event()

        # Lazy-loaded heavy resources.
        self._sd: Any = None
        self._np: Any = None
        self._whisper_model: Any = None
        self._tts: Any = None

        # Create / configure the dedicated voice context with the brevity prefix.
        prefix_prompt = VOICE_PROMPT_PREFIX + (agent.default_system_prompt or "")
        if VOICE_CONTEXT_NAME in agent.contexts:
            agent.contexts[VOICE_CONTEXT_NAME].set_system_prompt(prefix_prompt)
        else:
            agent.create_context(VOICE_CONTEXT_NAME, system_prompt=prefix_prompt)

    # ------------------------------------------------------------------
    # Lazy loaders
    # ------------------------------------------------------------------

    def _load_audio_libs(self) -> None:
        if self._sd is not None:
            return
        try:
            import numpy as np  # type: ignore
            import sounddevice as sd  # type: ignore
        except ImportError:
            _require("numpy/sounddevice")
        self._np = np
        self._sd = sd

    def _load_whisper(self) -> None:
        if self._whisper_model is not None:
            return
        try:
            import whisper  # type: ignore
        except ImportError:
            _require("openai-whisper")
        logger.info("Loading whisper base.en model on %s", self.config.device)
        self._whisper_model = whisper.load_model("base.en", device=self.config.device)

    def _load_tts(self) -> None:
        if self._tts is not None:
            return
        try:
            from ..tts import TextToSpeechService
        except ImportError:
            _require("kokoro-onnx")
        logger.info("Loading Kokoro TTS on %s", self.config.device)
        self._tts = TextToSpeechService(device=self.config.device)

    # ------------------------------------------------------------------
    # Audio helpers
    # ------------------------------------------------------------------

    def _play_tone(self) -> None:
        """Play a short sine-wave beep to signal that recording has begun."""
        np = self._np
        n_samples = int(_TONE_DURATION_S * SAMPLE_RATE)
        t = np.linspace(0.0, _TONE_DURATION_S, n_samples, endpoint=False)
        tone = (_TONE_AMPLITUDE * np.sin(2.0 * np.pi * _TONE_FREQ_HZ * t)).astype(
            np.float32
        )
        self._sd.play(tone, SAMPLE_RATE)
        self._sd.wait()

    def _record_until_silence(self) -> Any:
        """Block-record from the mic until silence_duration of quiet audio."""
        np = self._np
        sd = self._sd

        q: Queue[bytes] = Queue()

        def callback(indata, frames, time_info, status):  # noqa: ARG001
            if status:
                logger.debug("sounddevice status: %s", status)
            q.put(bytes(indata))

        chunks: list[bytes] = []
        silent_for = 0.0
        spoke = False
        start = time.monotonic()

        with sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            dtype="int16",
            channels=1,
            device=self.config.microphone_device,
            callback=callback,
        ):
            while True:
                if time.monotonic() - start > self.config.max_record_seconds:
                    break
                try:
                    data = q.get(timeout=0.5)
                except Exception:
                    continue
                chunks.append(data)
                frame = np.frombuffer(data, dtype=np.int16).astype(np.float32)
                rms = float(np.sqrt(np.mean(frame * frame))) if frame.size else 0.0
                frame_seconds = len(frame) / SAMPLE_RATE
                if rms < self.config.silence_threshold:
                    silent_for += frame_seconds
                    if spoke and silent_for >= self.config.silence_duration:
                        break
                else:
                    spoke = True
                    silent_for = 0.0

        raw = b"".join(chunks)
        if not raw:
            return np.zeros(0, dtype=np.float32)
        return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    def _transcribe(self, audio: Any) -> str:
        if audio.size == 0:
            return ""
        result = self._whisper_model.transcribe(
            audio, fp16=(self.config.device == "cuda")
        )
        return result.get("text", "").strip()

    def _speak(self, text: str) -> None:
        if not text.strip():
            return
        sample_rate, audio = self._tts.long_form_synthesize(
            text, voice_preset=self.config.tts_voice_preset
        )
        self._sd.play(audio, sample_rate)
        self._sd.wait()

    # ------------------------------------------------------------------
    # Wake-word loop (whisper-based)
    # ------------------------------------------------------------------

    def _wait_for_wake_word(self) -> bool:
        """Continuously record short audio windows and run whisper STT on each;
        return True when the configured wake-word phrase appears in the
        transcript. Returns False if :attr:`_stop` is set while waiting.
        """
        np = self._np
        sd = self._sd
        punct_stripper = str.maketrans("", "", string.punctuation)
        wake_word = self.config.wake_word.lower().translate(punct_stripper).strip()
        if not wake_word:
            raise ValueError("VoiceConfig.wake_word must not be empty")
        chunk_samples = max(1, int(SAMPLE_RATE * self.config.wake_word_chunk_seconds))
        chunk_num = 0

        logger.debug(
            "Wake-word loop starting: phrase=%r, chunk=%.1fs (%d samples)",
            wake_word,
            self.config.wake_word_chunk_seconds,
            chunk_samples,
        )

        while not self._stop.is_set():
            chunk_num += 1
            logger.debug("Recording wake-word chunk #%d ...", chunk_num)
            recording = sd.rec(
                chunk_samples,
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="int16",
                device=self.config.microphone_device,
                blocking=True,
            )
            if self._stop.is_set():
                break
            audio = recording.flatten().astype(np.float32) / 32768.0
            rms = float(np.sqrt(np.mean(audio * audio))) if audio.size else 0.0
            logger.debug("  chunk #%d: RMS=%.4f", chunk_num, rms)
            # Let whisper decide whether there is speech — skipping based on
            # RMS proved unreliable across mic hardware and gain settings.
            transcript = (
                self._transcribe(audio).lower().translate(punct_stripper).strip()
            )
            logger.debug("  chunk #%d: transcript=%r", chunk_num, transcript)
            if wake_word in transcript:
                return True
        return False

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Main loop — load models, then listen → transcribe → respond → speak."""
        print("Loading voice models (this may take a moment)...")
        self._load_audio_libs()
        self._load_whisper()
        self._load_tts()
        print(
            f"Talos is listening for wake word: {self.config.wake_word!r}. "
            "Press Ctrl+C to exit."
        )

        try:
            while not self._stop.is_set():
                if not self._wait_for_wake_word():
                    break
                self._play_tone()
                audio = self._record_until_silence()
                text = self._transcribe(audio)
                if not text:
                    print("(no speech detected)")
                    continue
                print(f"You: {text}")
                response = self.agent.send(text, context_name=VOICE_CONTEXT_NAME)
                print(f"Talos: {response}")
                # Strip tool-call syntax so it isn't spoken aloud.
                speech_text = clean_agent_response(response)
                self._speak(speech_text)
        except KeyboardInterrupt:
            print("\nExiting voice interface.")
        finally:
            self._stop.set()


def test_microphone(
    device: int | None = None,
    duration: float = 5.0,
    transcribe: bool = True,
    whisper_device: str = "cpu",
) -> None:
    """Interactive microphone test.

    Lists all input devices, records *duration* seconds from *device* (or the
    system default when *None*), prints a peak-level meter, and optionally
    transcribes the recording with whisper so the user can verify audio
    quality before committing to a ``microphone_device`` value in their config.

    Args:
        device: sounddevice device index to test.  ``None`` uses the system
            default input device.
        duration: Recording length in seconds.
        transcribe: When *True*, load whisper and print the transcript.
        whisper_device: ``"cuda"`` or ``"cpu"`` passed to whisper.load_model.
    """
    try:
        import sounddevice as sd  # type: ignore
    except ImportError:
        _require("numpy/sounddevice")

    # ------------------------------------------------------------------ list
    print("Available input devices")
    print("-" * 60)
    default_in = sd.default.device[0]
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] < 1:
            continue
        marker = "  <-- default" if i == default_in else ""
        print(f"  [{i:2d}]  {d['name']}{marker}")
    print("-" * 60)

    chosen = default_in if device is None else device
    chosen_name = sd.query_devices(chosen)["name"]
    print(f"\nRecording {duration:.1f}s from device [{chosen}] {chosen_name!r} ...")
    print("Speak now!\n")

    # ---------------------------------------------------------------- record
    samples = int(SAMPLE_RATE * duration)
    recording = sd.rec(
        samples,
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
        device=chosen,
        blocking=True,
    )
    float_audio = recording.flatten().astype(float) / 32768.0

    # ----------------------------------------------------------- level meter
    import numpy as _np  # already imported above, but keep local ref tidy

    peak = float(_np.max(_np.abs(float_audio)))
    rms = float(_np.sqrt(_np.mean(float_audio**2)))
    bar_width = 40
    print(f"  Peak : [{'#' * int(peak * bar_width):<{bar_width}}]  {peak:.3f}")
    print(f"  RMS  : [{'#' * int(rms * bar_width):<{bar_width}}]  {rms:.3f}")

    if peak < 0.01:
        print("\nWARNING: Very low signal — the microphone may not be capturing audio.")
    elif peak < 0.05:
        print("\nWARNING: Low signal — consider moving closer or raising input gain.")
    else:
        print("\nSignal looks good.")

    # ------------------------------------------------------- optional whisper
    if transcribe:
        print("\nLoading whisper to transcribe (this may take a moment)...")
        try:
            import whisper  # type: ignore
        except ImportError:
            _require("openai-whisper")
        model = whisper.load_model("base.en", device=whisper_device)
        result = model.transcribe(
            float_audio.astype("float32"), fp16=(whisper_device == "cuda")
        )
        text = result.get("text", "").strip()
        print(f"\nTranscript: {text!r}")
        if not text:
            print("(nothing transcribed — try speaking louder or closer)")

    print(
        f"\nTo use this device, set 'microphone_device: {chosen}' under 'voice:' "
        "in ~/.talos/config.yaml"
    )


__all__ = ["VoiceInterface", "VOICE_CONTEXT_NAME", "test_microphone"]
