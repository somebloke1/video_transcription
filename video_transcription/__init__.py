"""
Video Transcription Library
===========================

A modular library for video transcription pipelines, supporting both
local GPU processing and cloud API integration.

Modules:
    video_utils: FFmpeg operations (extract audio, duration, timestamps)
    keyframe: Stable keyframe extraction with perceptual hashing
    vad: Voice Activity Detection and audio chunking
    asr: Automatic Speech Recognition with multiple backends
    output: HTML assembly and formatting utilities
    batch: Video collection and batch processing
    gpu: GPU memory management
    compat: Compatibility patches for library conflicts
    prompts: System prompts for LLM-based analysis

Example usage:
    from video_transcription import (
        check_ffmpeg,
        extract_keyframes,
        dedupe_keyframes,
        collect_video_files,
        assemble_html,
    )
"""

# Video utilities
from .video_utils import (
    check_ffmpeg,
    get_video_duration,
    format_timestamp,
    extract_audio_wav,
    extract_audio_mp3,
    extract_audio_chunk,
)

# Keyframe extraction
from .keyframe import (
    extract_keyframes,
    dedupe_keyframes,
)

# VAD and audio chunking
from .vad import (
    load_silero_vad,
    detect_speech_segments,
    group_segments_into_chunks,
    is_degenerate_output,
    chunk_audio_with_vad,
)

# Output formatting
from .output import (
    assemble_html,
    assemble_html_with_descriptions,
    format_transcript_segments,
    extract_title_from_html,
    clean_video_name_for_title,
    HTML_STYLES,
)

# Batch processing
from .batch import (
    collect_video_files,
    BatchResults,
    print_batch_header,
    copy_to_all_transcripts,
    VIDEO_EXTENSIONS,
    DEFAULT_OUTPUT_DIR,
)

# GPU utilities
from .gpu import (
    cleanup_gpu,
    get_device,
    get_available_vram,
    get_gpu_info,
    print_gpu_status,
)

# Compatibility patches
from .compat import (
    apply_nemo_transformers_patch,
)

# Prompts
from .prompts import (
    get_clean_prompt_for_gemini_with_transcript,
    get_html_prompt_for_gemini_with_transcript,
    get_clean_prompt_for_gemini_with_audio,
    get_html_prompt_for_gemini_with_audio,
    get_clean_prompt_for_local_synthesis,
    get_html_prompt_for_local_synthesis,
    get_vision_analysis_prompt,
)

# ASR (Automatic Speech Recognition)
from .asr import (
    load_asr_model,
    unload_asr_model,
    transcribe_audio,
    transcribe_with_canary,
    transcribe_with_whisperx,
    transcribe_with_whisper,
    transcribe_with_parakeet,
    transcribe_with_granite,
    should_preload_asr,
    ASR_MODELS,
    ASR_MODEL_INFO,
    ASR_VRAM_GB,
)

__version__ = "1.0.0"

__all__ = [
    # video_utils
    "check_ffmpeg",
    "get_video_duration",
    "format_timestamp",
    "extract_audio_wav",
    "extract_audio_mp3",
    "extract_audio_chunk",
    # keyframe
    "extract_keyframes",
    "dedupe_keyframes",
    # vad
    "load_silero_vad",
    "detect_speech_segments",
    "group_segments_into_chunks",
    "is_degenerate_output",
    "chunk_audio_with_vad",
    # output
    "assemble_html",
    "assemble_html_with_descriptions",
    "format_transcript_segments",
    "extract_title_from_html",
    "clean_video_name_for_title",
    "HTML_STYLES",
    # batch
    "collect_video_files",
    "BatchResults",
    "print_batch_header",
    "copy_to_all_transcripts",
    "VIDEO_EXTENSIONS",
    "DEFAULT_OUTPUT_DIR",
    # gpu
    "cleanup_gpu",
    "get_device",
    "get_available_vram",
    "get_gpu_info",
    "print_gpu_status",
    # compat
    "apply_nemo_transformers_patch",
    # prompts
    "get_clean_prompt_for_gemini_with_transcript",
    "get_html_prompt_for_gemini_with_transcript",
    "get_clean_prompt_for_gemini_with_audio",
    "get_html_prompt_for_gemini_with_audio",
    "get_clean_prompt_for_local_synthesis",
    "get_html_prompt_for_local_synthesis",
    "get_vision_analysis_prompt",
    # asr
    "load_asr_model",
    "unload_asr_model",
    "transcribe_audio",
    "transcribe_with_canary",
    "transcribe_with_whisperx",
    "transcribe_with_whisper",
    "transcribe_with_parakeet",
    "transcribe_with_granite",
    "should_preload_asr",
    "ASR_MODELS",
    "ASR_MODEL_INFO",
    "ASR_VRAM_GB",
]
