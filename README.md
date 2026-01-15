# Video Transcription Pipeline

Transform educational videos into comprehensive technical documentation. Designed for learning engineers who need to extract and synthesize content from video courses.

## Quick Start

```bash
# Activate environment
source .venv/bin/activate

# Set API key (for Gemini scripts)
export GOOGLE_API_KEY="your-key-here"

# Process a video
python process_video_gemini_finisher.py video.mp4
```

## Choose Your Script

| Script | Description | Requirements |
|--------|-------------|--------------|
| `process_video_gemini_finisher.py` | **Recommended.** Local ASR + Gemini synthesis | GPU + API key |
| `process_video_gemini_only.py` | Cloud-only, simplest setup | API key only |
| `process_video_local_only.py` | Fully offline, multiple ASR options | GPU (24GB VRAM) |

## Features

- **Stable Keyframe Extraction** - Captures only frames that remain stable (no transitions)
- **Multiple ASR Options** - WhisperX, Canary (technical jargon), Whisper, Parakeet, Granite
- **Smart Audio Chunking** - Handles long audio with VAD-based silence detection
- **Batch Processing** - Process directories or file lists with error recovery
- **Output Formats** - Clean Markdown or HTML with embedded images

## Usage

### Single Video
```bash
python process_video_gemini_finisher.py video.mp4
```

### Batch Processing
```bash
# Process all videos in a directory
python process_video_gemini_finisher.py ./videos/

# Process from a list file
python process_video_gemini_finisher.py video_list.txt
```

### Options
```bash
# Technical content (better for jargon like BGP, OSPF, VXLAN)
python process_video_gemini_finisher.py --asr canary video.mp4

# Use WhisperX with local-only script for faster offline processing
python process_video_local_only.py --asr whisperx video.mp4

# HTML output with embedded screenshots
python process_video_gemini_finisher.py --format html video.mp4

# Require frames to be stable for 5 seconds
python process_video_gemini_finisher.py --min-stable 5.0 video.mp4

# Custom output directory
python process_video_gemini_finisher.py -o ./output video.mp4
```

## Output

```
transcriptions/
└── video_name_analysis/
    ├── transcript.md          # Synthesized document
    ├── audio_transcript.txt   # Raw transcription with timestamps
    └── keyframes/             # Extracted screenshots
        ├── frame_0001.png
        └── ...
```

## Installation

### Prerequisites
- Python 3.12+
- NVIDIA GPU with CUDA (for local ASR)
- FFmpeg

### Setup
```bash
# Create virtual environment
uv venv .venv
source .venv/bin/activate

# PyTorch with CUDA (install first)
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install based on your use case:

# Gemini scripts (recommended)
uv pip install -e ".[gemini]"

# Full local-only processing
uv pip install -e ".[local]"

# Everything except special installs
uv pip install -e ".[full]"

# Additional ASR models (require special installation):

# WhisperX (not on PyPI)
uv pip install whisperx

# Canary/Parakeet (NeMo from git)
uv pip install "nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git"
```

### API Key
Get a Gemini API key at https://aistudio.google.com/app/apikey

```bash
export GOOGLE_API_KEY="your-key-here"
```

## ASR Models

| Model | Flag | VRAM | Speed | Notes |
|-------|------|------|-------|-------|
| WhisperX | `--asr whisperx` | ~3GB | 70x | Default, word-level timestamps |
| Canary | `--asr canary` | ~5GB | 1x | Best for technical terminology |
| Whisper | `--asr whisper` | ~3GB | 1x | Original OpenAI model |
| Parakeet | `--asr parakeet` | ~2GB | 6x | Fast, NVIDIA |
| Granite | `--asr granite` | ~16GB | 0.5x | Best accuracy, no timestamps* |

> **\*Granite limitation:** Processes entire audio as one segment without internal timestamps. In HTML output, images will be grouped at the end rather than placed inline with corresponding content. Use WhisperX or Canary if timeline-correlated image placement matters.

## Project Structure

The project uses a modular architecture with shared code in the `video_transcription/` package:

```
video_transcription/
├── process_video_gemini_finisher.py   # Main scripts
├── process_video_gemini_only.py
├── process_video_local_only.py
└── video_transcription/               # Shared library
    ├── video_utils.py                 # FFmpeg operations
    ├── keyframe.py                    # Stable keyframe extraction
    ├── vad.py                         # VAD-based audio chunking
    ├── asr.py                         # ASR backends (canary, whisperx, etc.)
    ├── output.py                      # HTML assembly
    ├── batch.py                       # Batch processing utilities
    ├── gpu.py                         # GPU memory management
    ├── compat.py                      # Compatibility patches
    └── prompts.py                     # LLM system prompts
```

The shared library can also be imported for custom pipelines:

```python
from video_transcription import (
    extract_keyframes,
    dedupe_keyframes,
    collect_video_files,
    cleanup_gpu,
    # ASR functions
    transcribe_audio,
    ASR_MODELS,
)
```

## Troubleshooting

### GPU Memory Issues
```bash
# Clear orphaned GPU memory
python clear_gpu.py

# Or use the library function
python -c "from video_transcription import cleanup_gpu; cleanup_gpu(aggressive=True)"
```

### Import Errors
The scripts include automatic compatibility patches for common issues like `PytorchGELUTanh`.

### Garbage Transcription Output
Long audio files are automatically chunked at silence boundaries to prevent garbage output from ASR models with duration limits.

## Hardware Requirements

| Script | Minimum VRAM | Recommended |
|--------|--------------|-------------|
| gemini_only | None (cloud) | Any |
| gemini_finisher | 3-5GB | 8GB+ |
| local_only | 20GB+ | 24GB (RTX 3090) |

## License

MIT
