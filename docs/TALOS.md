# Talos

Talos is a local-first AI assistant built on the **pithos** agent framework. It provides three ready-to-use interfaces — a terminal shell, a wake-word voice assistant, and a Telegram bot — all driven by the same configurable pithos agent.

Configuration lives in `~/.talos/config.yaml`. On first run a guided setup wizard creates the file interactively. Every subsequent run loads it automatically, and `--reconfigure` forces the wizard to run again at any time.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Interfaces](#interfaces)
  - [Shell](#shell)
  - [Voice](#voice)
  - [Telegram](#telegram)
- [Configuration](#configuration)
  - [Agent Settings](#agent-settings)
  - [Tool Settings](#tool-settings)
  - [Memory Settings](#memory-settings)
  - [Voice Settings](#voice-settings)
  - [Telegram Settings](#telegram-settings)
- [Installation & Extras](#installation--extras)
- [CLI Reference](#cli-reference)

---

## Quick Start

```bash
# Install talos extras (voice deps are large; skip if you only want shell/telegram)
pip install -e ".[talos]"

# First run — launches the setup wizard, then starts the shell
talos shell
```

The wizard asks for:
- Ollama model name
- System prompt
- Whether to enable tools and vector memory
- Compute device (cpu / cuda)
- Wake word
- Optional Telegram bot token

The completed config is saved to `~/.talos/config.yaml`.

---

## Interfaces

### Shell

An interactive stdin/stdout chat that wraps pithos's built-in REPL.

```bash
talos shell
```

- Type your message and press Enter.
- `Ctrl+C` exits.
- Runs in the terminal with no extra dependencies beyond `pithos`.

### Voice

An always-on, wake-word-triggered speech-to-speech loop.

```bash
talos voice
```

**How it works:**

1. The microphone is sampled in short overlapping chunks.
2. Each chunk is transcribed by **Whisper** (`base.en`).
3. When the transcript contains the configured wake word (default: `"hey talos"`), a short beep sounds to confirm activation.
4. Recording continues until silence is detected (configurable RMS threshold and duration) or `max_record_seconds` is reached.
5. The full utterance is transcribed and sent to the pithos agent.
6. The agent's reply is synthesised by **Kokoro-ONNX** TTS and played back.

**Mic test:**

```bash
# List devices and record a short clip to verify input works
talos mic-test

# Test a specific device index
talos mic-test --device 2 --duration 5
```

**Requirements:** `pip install -e ".[talos]"` (installs `openai-whisper`, `sounddevice`, `kokoro-onnx`, `torch`, `pydub`). Model weights are downloaded once into `~/.talos/models/kokoro/` and cached for offline use.

**Voice context:** The voice interface uses a dedicated agent context (`voice`) with a brevity prefix prepended to the system prompt, keeping responses short enough to speak comfortably.

### Telegram

A Telegram bot that gives each user their own persistent agent context.

```bash
talos telegram
```

**Setup:**

1. Create a bot with [@BotFather](https://t.me/BotFather) and copy the token.
2. Add it to `~/.talos/config.yaml` under `telegram.bot_token`, or re-run `talos --reconfigure`.
3. Start the bot with `talos telegram`.

**Behaviour:**

- Each Telegram user (by user ID) gets an isolated pithos context so conversation history is per-user.
- Tool-call syntax is stripped from responses before delivery so users see clean prose.
- Messages longer than 4 000 characters are automatically split to stay within Telegram's limit.

**Requirements:** `pip install -e ".[talos]"` (includes `python-telegram-bot`).

---

## Configuration

Config file location: `~/.talos/config.yaml`.

Run `talos config` to re-run the wizard at any time without starting an interface.

### Agent Settings

```yaml
agent:
  model: glm-4.7-flash          # Ollama model name
  system_prompt: "You are Talos, a helpful AI assistant. Answer concisely in 20 words or less."
  temperature: 0.7
  tools:
    enabled: false              # Enable CLI tool calling
    mode: strict               # "strict" | "standard" | "permissive"
    auto_loop: false            # Re-prompt after tool calls automatically
    max_iterations: 5           # Safety cap on auto-loop iterations
  memory:
    enabled: false              # Enable vector memory (ChromaDB)
    compaction: false           # Auto-compact old messages
    compaction_threshold: 20    # Message count that triggers compaction
    recall: false               # Inject relevant memories before each turn
    history: false              # Persist conversation history to SQLite
```

### Tool Settings

When `tools.enabled` is `true`, the agent can execute CLI commands. The `mode` field mirrors the tool-config modes from pithos:

| mode | behaviour |
|------|----------|
| `strict` | only tools in `configs/tools/tool_config.yaml` include list; confirm-listed tools need user approval |
| `standard` | all tools except those in the exclude list; confirm-listed tools need user approval |
| `permissive` | all tools except those in the exclude list; confirm-listed tools are auto-approved |

See [TOOL_CALLING.md](TOOL_CALLING.md) for full tool configuration documentation.

### Memory Settings

When `memory.enabled` is `true`, the agent is wired to a ChromaDB vector store for persistent knowledge. Enabling `recall` automatically surfaces relevant memories before each user turn (RAG injection). Enabling `history` persists conversations to SQLite/ChromaDB so they are searchable across sessions.

See [MEMORY.md](MEMORY.md) for detailed memory documentation.

### Voice Settings

```yaml
voice:
  device: cuda                  # "cuda" or "cpu"
  microphone_device: null       # null = system default; integer = device index
  wake_word: "hey talos"        # Whisper must detect this phrase to activate
  wake_word_chunk_seconds: 2.0  # Audio window length Whisper scans for wake word
  tts_voice_preset: af_heart    # Kokoro-ONNX voice preset
  silence_threshold: 500.0      # RMS energy below this = silence
  silence_duration: 1.5         # Seconds of silence before ending recording
  max_record_seconds: 30.0      # Hard cap on a single utterance
```

**Choosing a microphone device:**

```bash
talos mic-test   # lists all input devices with their index numbers
```

Then set `microphone_device` to the desired integer index.

### Telegram Settings

```yaml
telegram:
  bot_token: ""    # Token from @BotFather — leave empty to disable
```

---

## Installation & Extras

The base `pithos` install is sufficient for the shell interface. The voice and Telegram interfaces need additional packages provided by the `talos` extras group:

```bash
pip install -e ".[talos]"
```

This installs:

| Package | Purpose |
|---------|---------|
| `rich` | Pretty terminal output and the setup wizard UI |
| `numpy` | Audio buffer manipulation |
| `sounddevice` | Microphone capture and audio playback |
| `openai-whisper` | Speech-to-text for wake-word detection and command transcription |
| `torch` | Required by Whisper and Kokoro-ONNX |
| `kokoro-onnx` | Neural TTS for voice responses |
| `pydub` | Audio format conversion (MP3 beep) |
| `python-telegram-bot` | Telegram bot framework |

The `talos` extras are intentionally separate from the base install because `torch` and `openai-whisper` are large downloads. If you only need the shell or Telegram interface you can install only what you need:

```bash
# Shell only — no extra deps beyond pithos
pip install -e "."

# Telegram only
pip install "python-telegram-bot>=21.0"

# Full voice + telegram
pip install -e ".[talos]"
```

---

## CLI Reference

```
talos [--config PATH] [--reconfigure] [--debug] <interface>
```

| Subcommand | Description |
|------------|-------------|
| `shell` | Interactive stdin/stdout chat (default if no subcommand given) |
| `voice` | Wake-word speech-to-speech assistant |
| `telegram` | Telegram bot |
| `config` | Run the setup wizard and exit without starting an interface |
| `mic-test` | List microphone devices and record a short test clip |

**Global flags:**

| Flag | Description |
|------|-------------|
| `--config PATH` | Override the default config path (`~/.talos/config.yaml`) |
| `--reconfigure` | Force re-run of the setup wizard even if a config exists |
| `--debug` | Enable DEBUG-level logging |

**`mic-test` flags:**

| Flag | Description |
|------|-------------|
| `--device N` | Device index to test (default: system default) |
| `--duration S` | Recording duration in seconds (default: 3) |
