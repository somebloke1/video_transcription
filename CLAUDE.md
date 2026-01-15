# Video Transcription Pipeline

## Project Overview

A professional video transcription system for educational content (primarily GCP networking videos), designed for learning engineers who need technical documentation from video courses.

## Repository

- **GitHub**: https://github.com/somebloke1/video_transcription (private)
- **Branch**: main

## Hardware Environment

- **GPU**: NVIDIA RTX 3090 (24GB VRAM)
- **OS**: WSL2 Linux (Debian)
- **Python**: 3.12+ with virtual environment at `.venv`
- **Package Manager**: uv (preferred) or pip

## Architecture

The project uses three independent processing scripts, named by how language model processing is distributed between local GPU and cloud APIs:

### Active Scripts

| Script | ASR | Visual Analysis | Synthesis | Best For |
|--------|-----|-----------------|-----------|----------|
| `process_video_gemini_finisher.py` | Local (multiple options) | Gemini 2.5 Flash | Gemini 2.5 Flash | Technical jargon with cloud polish |
| `process_video_gemini_only.py` | Gemini 2.5 Flash | Gemini 2.5 Flash | Gemini 2.5 Flash | Simplest setup, cloud-only |
| `process_video_local_only.py` | Local (multiple options) | Qwen2.5-VL-7B | Qwen2.5-14B-AWQ | Fully offline, no API costs |

## Key Features

### Stable Keyframe Extraction
- Uses perceptual hashing (imagehash/phash) to detect frame stability
- Only captures frames stable for configurable duration (default: 3.0s)
- Filters out transition/animation frames automatically
- MD5 deduplication removes exact duplicates

### VAD-Based Audio Chunking
- Canary-Qwen has a 40-second maximum audio duration limit
- Silero VAD detects speech/silence boundaries
- Audio is chunked at natural silence points (max 35s chunks)
- Prevents garbage output from oversized audio

### Degenerate Output Detection
- Detects garbage transcriptions (repetitive characters like "2222222" or ",,,,,")
- Skips chunks that produce degenerate output
- Prevents garbage from contaminating final document

### Timestamp Correlation
- Keyframes include timestamps (e.g., [05:32])
- Prompts instruct LLMs to correlate visual content with transcript segments
- Enables accurate integration of spoken and visual content

### Batch Processing
- Process single files, directories, or text files with paths
- Resilient to failures - continues processing remaining videos
- Error tracking with `batch_errors.txt` report
- Summary statistics at completion

### Output Formats
- **clean**: Readable Markdown prose without timestamps or image references
- **html**: HTML document with embedded base64 keyframe images

### Keyframe Rendering Modes (`--keyframe-rendering`, `-k`)

Controls how keyframe images are represented in the output:

| Mode | Description | Output | Default For |
|------|-------------|--------|-------------|
| `embedded` | Base64 images embedded in HTML | Actual images viewable in browser | `--format html` |
| `markup` | Structured diagram notation | Mermaid diagrams, ASCII boxes, markdown tables | - |
| `brief` | Short text description | 1-2 sentences per keyframe | - |
| `detailed` | Full visual analysis | Complete paragraph describing all visual content | `--format clean` |

**Script Support:**

| Script | embedded | markup | brief | detailed |
|--------|----------|--------|-------|----------|
| gemini_only | ✓ | ✓ | ✓ | ✓ |
| gemini_finisher | ✓ | ✓ | ✓ | ✓ |
| local_only | ✓ | ✗ | ✓ | ✓ |

Note: `markup` mode requires Gemini for reliable Mermaid diagram generation.

## ASR Model Options (gemini_finisher and local_only)

| Model | VRAM | Speed | Best For |
|-------|------|-------|----------|
| `whisperx` (default for gemini_finisher) | ~3GB | 70x faster | General use, word-level timestamps |
| `canary` (default for local_only) | ~5GB | Normal | Technical jargon (BGP, OSPF, VXLAN) |
| `whisper` | ~3GB | Normal | Original Whisper, reliable |
| `parakeet` | ~2GB | 6x faster | Fast processing |
| `granite` | ~16GB | Slow | Best accuracy, no timestamps* |

*\*Granite limitation: Processes entire audio as one segment without internal timestamps. Output documents will have all images placed at the end rather than correlated with the timeline. Use WhisperX or Canary if timeline-correlated image placement is important.*

## Usage Examples

### Single Video (Local File)
```bash
python process_video_gemini_finisher.py video.mp4
python process_video_gemini_only.py video.mp4
python process_video_local_only.py video.mp4
```

### Batch Processing
```bash
# From directory
python process_video_gemini_finisher.py ./videos/

# From list file (one path per line)
python process_video_gemini_finisher.py video_list.txt
```

### Options
```bash
# Use Canary ASR for technical content (gemini_finisher)
python process_video_gemini_finisher.py --asr canary video.mp4

# Use WhisperX for faster processing (local_only)
python process_video_local_only.py --asr whisperx video.mp4

# HTML output with embedded images
python process_video_gemini_finisher.py --format html video.mp4

# Adjust keyframe stability threshold
python process_video_gemini_finisher.py --min-stable 5.0 video.mp4

# Keyframe rendering modes
python process_video_gemini_finisher.py --format html -k embedded video.mp4  # Default: images in HTML
python process_video_gemini_finisher.py --format html -k markup video.mp4    # Mermaid diagrams
python process_video_gemini_finisher.py --format clean -k brief video.mp4    # Short descriptions
python process_video_gemini_finisher.py --format clean -k detailed video.mp4 # Full descriptions (default for clean)
```

## Output Structure

```
transcriptions/
└── video_name_analysis/
    ├── transcript.md          # or transcript.html
    ├── audio_transcript.txt   # Raw ASR output with timestamps
    ├── audio.wav              # Extracted audio
    └── keyframes/
        ├── frame_0001.png
        ├── frame_0002.png
        └── ...
```

## Dependencies

Managed via `pyproject.toml` with pinned versions for reproducibility.

### Installation Groups

```bash
# Activate virtual environment first
source .venv/bin/activate

# Install by use case:
uv pip install -e ".[gemini]"     # Gemini scripts
uv pip install -e ".[local]"      # Local-only processing
uv pip install -e ".[full]"       # Everything (except special installs)

# Special installs (not on PyPI):
uv pip install whisperx           # WhisperX ASR
uv pip install "nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git"  # Canary/Parakeet
```

### Pinned Versions (from working environment)

| Package | Version | Notes |
|---------|---------|-------|
| torch | 2.6.0+cu124 | CUDA 12.4 |
| transformers | 4.57.5 | Granite requires >=4.52.4 |
| google-genai | 1.57.0 | Gemini API |
| openai-whisper | 20250625 | Whisper ASR |
| autoawq | 0.2.9 | Deprecated but functional |
| peft | 0.18.1 | Granite requirement |
| pillow | 12.0.0 | Image processing |
| imagehash | 4.3.2 | Keyframe deduplication |

### System Dependencies
- FFmpeg (required for audio/video extraction)

## Environment Variables

- `GOOGLE_API_KEY` - Required for Gemini scripts (get at https://aistudio.google.com/app/apikey)

## System Administration

For commands requiring sudo, use heredoc to pass the password:

```bash
cat << 'PASS' | sudo -S <command>
Fjdksl!23
PASS
```

This avoids issues with special characters (like `!`) in the password.

## Known Issues & Solutions

### PytorchGELUTanh Import Error
NeMo/transformers version conflict. Scripts include automatic compatibility patch.

### Canary Garbage Output
Audio >40s produces garbage. Fixed with VAD-based chunking.

### OOM Errors
- Qwen2.5-14B cannot run alongside other models on 24GB
- Use AWQ quantized version (Qwen2.5-14B-Instruct-AWQ) for local_only
- Clear orphaned GPU memory: `python clear_gpu.py`

### AutoAWQ Deprecation Warning
- AutoAWQ is deprecated but still functional with torch 2.6.0
- Shows deprecation warning on import (can be ignored)
- Compatibility patch in `compat.py` fixes PytorchGELUTanh import error

## Technical Notes

### Prompt Engineering
- Filler slides (greetings, transitions, Q&A) are explicitly filtered
- Technical terminology preservation is emphasized
- Timestamp correlation instructions help LLMs integrate audio/visual

### Memory Management
- Models loaded/unloaded between stages
- `cleanup_gpu()` called after each processing step
- Smart ASR preloading based on VRAM constraints:
  - `should_preload_asr()` checks if ASR + next stage fit in VRAM
  - Small models (WhisperX ~3GB, Canary ~5GB): preloaded for batch efficiency
  - Large models (Granite ~16GB): loaded per-video to avoid OOM with vision stage (~16GB)

## Shared Library Architecture

The project uses a modular `video_transcription/` package that contains shared functionality used by all three scripts. This eliminates code duplication and provides a clean, maintainable architecture.

### Package Structure

```
video_transcription/
├── __init__.py        # Public API exports
├── video_utils.py     # FFmpeg operations (extract audio, duration, timestamps)
├── keyframe.py        # Stable keyframe extraction with perceptual hashing
├── vad.py             # VAD-based audio chunking (Silero VAD)
├── asr.py             # Automatic Speech Recognition (multiple backends)
├── output.py          # HTML assembly and formatting utilities
├── batch.py           # Video collection and batch processing
├── gpu.py             # GPU memory management
├── compat.py          # Compatibility patches (PytorchGELUTanh)
└── prompts.py         # System prompts for LLM-based analysis
```

### Key Modules

| Module | Purpose |
|--------|---------|
| `video_utils` | `check_ffmpeg()`, `get_video_duration()`, `format_timestamp()`, `extract_audio_wav()`, `extract_audio_mp3()` |
| `keyframe` | `extract_keyframes()`, `dedupe_keyframes()` - stable frame detection with perceptual hashing |
| `vad` | `load_silero_vad()`, `detect_speech_segments()`, `group_segments_into_chunks()`, `is_degenerate_output()` |
| `asr` | `load_asr_model()`, `unload_asr_model()`, `transcribe_audio()` - supports canary, whisperx, whisper, parakeet, granite |
| `output` | `assemble_html()`, `format_transcript_segments()`, HTML styling |
| `batch` | `collect_video_files()`, `BatchResults` class, `VIDEO_EXTENSIONS`, `DEFAULT_OUTPUT_DIR` |
| `gpu` | `cleanup_gpu()`, `get_device()`, `get_available_vram()` |
| `compat` | `apply_nemo_transformers_patch()` - fixes NeMo/transformers version conflicts |
| `prompts` | System prompts for Gemini and local LLM synthesis (clean/html formats) |

### Usage in Scripts

```python
from video_transcription import (
    check_ffmpeg,
    extract_keyframes,
    dedupe_keyframes,
    collect_video_files,
    assemble_html,
    cleanup_gpu,
    BatchResults,
    # ASR functions
    load_asr_model,
    unload_asr_model,
    transcribe_audio,
    ASR_MODELS,
)
```

## File Inventory

```
video_transcription/                   # Project root
├── .gitignore                         # Git ignore patterns
├── CLAUDE.md                          # This file (agent instructions)
├── README.md                          # User-facing documentation
├── pyproject.toml                     # Package config with pinned deps
├── requirements.txt                   # Legacy requirements (reference)
├── process_video_gemini_finisher.py   # Hybrid: local ASR + Gemini
├── process_video_gemini_only.py       # Cloud-only: all Gemini
├── process_video_local_only.py        # Offline: all local models
├── clear_gpu.py                       # GPU memory cleanup utility
├── test_gemini_key.py                 # API key validation
├── video_transcription/               # Shared library package
│   ├── __init__.py                    # Public API exports
│   ├── video_utils.py                 # FFmpeg operations
│   ├── keyframe.py                    # Stable keyframe extraction
│   ├── vad.py                         # VAD-based audio chunking
│   ├── asr.py                         # ASR backends (5 models)
│   ├── output.py                      # HTML assembly
│   ├── batch.py                       # Batch processing
│   ├── gpu.py                         # GPU memory management
│   ├── compat.py                      # Compatibility patches
│   └── prompts.py                     # LLM system prompts
└── transcriptions/                    # Output directory (gitignored)
```

## Git Workflow

```bash
# Check status
git status

# Commit changes
git add -A && git commit -m "Description"

# Push to GitHub
git push

# The repo uses gh as credential helper (configured)
```
