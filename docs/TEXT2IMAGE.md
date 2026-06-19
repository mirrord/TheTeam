# Text-to-Image Tool

The `text2image` virtual tool generates a PNG from a text prompt using a
local image model. It follows the same `ToolProvider` plugin pattern as
`web-research` and `flowchart`: registered through `ToolRegistry`,
dispatched in-process, and invoked with any of pithos's standard tool
call syntaxes.

It is exposed two ways:

1. **CLI**: [`pithos-text2image`](#cli).
2. **Agent tool call**: `RUN: text2image <prompt>` (virtual tool, no
   external binary required).

The tool saves the PNG to `./data/generated_images/` and returns a
markdown summary (path, backend, model, dimensions, steps, seed, time).

## Backends

Three backends are supported, selected by the `backend` key in
`configs/tools/text2image_config.yaml`:

| Backend | Value | Dependencies | Server required |
|---------|-------|-------------|-----------------|
| Automatic1111/Forge | `http` | `requests` (the `web` extra) | Yes |
| ComfyUI | `comfyui` | `requests` (the `web` extra) | Yes |
| Hugging Face diffusers | `diffusers` | `torch`, `diffusers`, `Pillow`, `accelerate`, `transformers` (the `image` extra) | No |

The HTTP-based backends (`http`, `comfyui`) require only the lightweight
`web` extra. The `diffusers` backend loads the model in-process and requires
the heavier `image` extra.

### http backend (Automatic1111/Forge)

POSTs a JSON payload to `/sdapi/v1/txt2img` on a running
Automatic1111/Stable Diffusion WebUI or Forge server:

```http
POST http://127.0.0.1:7860/sdapi/v1/txt2img
{
  "prompt": "...",
  "negative_prompt": "...",
  "width": 512, "height": 512,
  "steps": 30, "cfg_scale": 7.5,
  "sampler_name": "Euler a",
  "seed": 12345
}
```

The first entry in `response.json()["images"]` is base64-decoded to obtain
the PNG bytes.

### comfyui backend

Submits a node-graph workflow to a running ComfyUI server using three
endpoints:

1. `POST /prompt` — submit the workflow, receive a `prompt_id`.
2. `GET /history/{prompt_id}` — poll until the run appears in history.
3. `GET /view?filename=...` — download the first output image.

The workflow is a standard ComfyUI API-format JSON. You can supply your own
via `comfyui_workflow_path`; leave the field empty to use the built-in SD1.5
txt2img workflow.

**Placeholder tokens** in workflow input values are substituted per call:

| Token | Value |
|-------|-------|
| `%prompt%` | Text prompt |
| `%negative_prompt%` | Negative prompt (default: empty) |
| `%seed%` | Resolved seed (fixed or random) |
| `%steps%` | Denoising steps |
| `%cfg%` | Guidance scale |
| `%width%` | Image width |
| `%height%` | Image height |
| `%sampler%` | Sampler name |
| `%model%` | Checkpoint name |

The built-in workflow uses a `CheckpointLoaderSimple` node, so `model` must
be set to a checkpoint filename known to your ComfyUI installation (e.g.
`"v1-5-pruned-emaonly.safetensors"`).

### diffusers backend

Loads a Hugging Face `AutoPipelineForText2Image` in-process (lazy, cached
after first call). Requires the `image` extra:

```bash
pip install -e ".[image]"
```

Set `model` to a HF repo id or a local path (e.g.
`"stabilityai/sd-turbo"`, `"./models/realisticVision"`). Set `device` to
`"cuda"`, `"cpu"`, or `"mps"` depending on your hardware.

## Architecture

```
Text2ImageToolProvider   (src/pithos/tools/text2image/provider.py)
        |
        | lazy init
        v
Text2ImageGenerator      (src/pithos/tools/text2image/generator.py)
        |
        | build_backend(config)
        v
ImageBackend  ──────────────┬─────────────────────┬──────────────────
                            │                     │                  │
                   HttpBackend          ComfyUIBackend     DiffusersBackend
             /sdapi/v1/txt2img    /prompt+/history+/view   diffusers pipeline
                            └─────────────────────┴──────────────────
                                        |
                              GeneratedImage(png_bytes, metadata)
                                        |
                              writes PNG to output_dir/
                                        |
                              ToolResult(stdout=markdown summary)
```

## Installation

The tool ships **disabled by default** because the backends require a
running local server or heavy Python dependencies.

```bash
# HTTP-based backends (A1111/Forge or ComfyUI)
pip install -e ".[web]"

# diffusers in-process backend
pip install -e ".[image]"
```

## Configuration

Enable the tool and choose a backend in `configs/tools/tool_config.yaml`:

```yaml
text2image:
  enabled: true
```

All generation parameters live in `configs/tools/text2image_config.yaml`:

```yaml
# Backend: "http", "comfyui", or "diffusers"
backend: http

output_dir: ./data/generated_images

# Generation defaults
model: ""           # checkpoint name or HF repo id
width: 512
height: 512
steps: 30
guidance_scale: 7.5
negative_prompt: ""
seed: null          # null → random each call

# diffusers only
device: cuda

# http / comfyui shared
sampler: "Euler a"

# http only
base_url: "http://127.0.0.1:7860"
timeout: 120

# comfyui only
comfyui_base_url: "http://127.0.0.1:8188"
comfyui_workflow_path: ""   # empty → built-in SD1.5 workflow
```

### Quick-start: ComfyUI

1. Start ComfyUI (default port 8188).
2. Set `backend: comfyui` and `model: <your-checkpoint>.safetensors`.
3. Enable the tool in `tool_config.yaml` (`text2image.enabled: true`).
4. Enable tools for your agent:

```python
agent.enable_tools(config_manager)
```

5. Prompt the agent:

```
RUN: text2image a wide ocean at dusk, photorealistic
```

### Quick-start: Automatic1111/Forge

1. Start A1111/Forge with `--api` flag (default port 7860).
2. Set `backend: http` (this is already the default).
3. Enable the tool and prompt as above.

### Quick-start: diffusers

```bash
pip install -e ".[image]"
```

```yaml
# text2image_config.yaml
backend: diffusers
model: "stabilityai/sd-turbo"
device: cuda
steps: 1
guidance_scale: 0.0   # sd-turbo doesn't use CFG
```

## CLI

```bash
# Basic usage (backend + settings from text2image_config.yaml)
pithos-text2image "a red fox in a snowy forest"

# Override backend and steps for this run
pithos-text2image --backend comfyui --steps 20 "a glowing crystal cave"

# Custom output directory and fixed seed
pithos-text2image --output-dir /tmp/imgs --seed 42 "a futuristic cityscape"

# Full option reference
pithos-text2image --help
```

**Options**

| Flag | Description |
|------|-------------|
| `--backend` | Override config backend (`http`, `comfyui`, `diffusers`). |
| `--output-dir` | Override `output_dir`. |
| `--model` | Override checkpoint name / HF repo id. |
| `--width` / `--height` | Override image dimensions. |
| `--steps` | Override denoising step count. |
| `--seed` | Fixed RNG seed. |
| `--negative-prompt` | Override the negative prompt. |
| `--config-dir` | Use a custom `configs/` directory. |
| `--quiet` | Suppress non-error logging. |

## Output

Images are saved as `<timestamp>_<prompt-slug>.png` inside `output_dir`.
The tool result returned to the agent looks like:

```
Image generated successfully.
- Path: ./data/generated_images/20260618-143201_a-red-fox-sitting.png
- Backend: http
- Model: (server default)
- Size: 512x512
- Steps: 30
- Seed: 2847391024
- Time: 4.271s
```

## Module Layout

```
src/pithos/tools/text2image/
├── __init__.py     # Optional-dep flags (TEXT2IMAGE_AVAILABLE,
│                   # HTTP_BACKEND_AVAILABLE, DIFFUSERS_BACKEND_AVAILABLE)
│                   # + lazy __getattr__ shim
├── config.py       # Text2ImageConfig dataclass with from_dict()
├── backends.py     # ImageBackend ABC, DiffusersBackend, HttpBackend,
│                   # ComfyUIBackend, build_backend() factory
├── generator.py    # Text2ImageGenerator: orchestrates backend + file I/O
└── provider.py     # Text2ImageToolProvider(ToolProvider)
```

## Security Notes

- The tool only writes PNG files to a configurable local directory.
- No subprocesses are spawned.
- HTTP requests are made to a locally configured server URL only; the URL
  is never derived from agent input.
- `comfyui_workflow_path` is a server-side config file path, not exposed to
  the agent.
