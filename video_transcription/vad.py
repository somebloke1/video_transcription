"""
Voice Activity Detection (VAD) based audio chunking.

This module provides functions for detecting speech segments in audio
and grouping them into chunks suitable for ASR models with duration limits.
Uses Silero VAD for speech/silence detection.
"""

from pathlib import Path
from typing import List, Tuple

import torch

from .video_utils import extract_audio_chunk


def load_silero_vad():
    """
    Load Silero VAD model for speech/silence detection.

    Returns:
        Tuple of (vad_model, utils) from torch.hub.
        utils[0] is the get_speech_timestamps function.
    """
    vad_model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False,
        onnx=False
    )
    return vad_model, utils


def detect_speech_segments(
    audio_path: Path,
    vad_model,
    get_speech_timestamps,
    threshold: float = 0.5,
    min_speech_duration_ms: int = 250,
    min_silence_duration_ms: int = 100
) -> List[Tuple[float, float]]:
    """
    Detect speech segments in audio using Silero VAD.

    Args:
        audio_path: Path to the audio file.
        vad_model: Loaded Silero VAD model.
        get_speech_timestamps: Function from Silero utils.
        threshold: VAD threshold (default: 0.5).
        min_speech_duration_ms: Minimum speech duration in ms.
        min_silence_duration_ms: Minimum silence duration in ms.

    Returns:
        List of (start_sec, end_sec) tuples for speech regions.
    """
    import torchaudio

    # Load audio
    waveform, sample_rate = torchaudio.load(str(audio_path))

    # Resample to 16kHz if needed (Silero VAD expects 16kHz)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
        sample_rate = 16000

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Get speech timestamps
    speech_timestamps = get_speech_timestamps(
        waveform.squeeze(),
        vad_model,
        sampling_rate=sample_rate,
        threshold=threshold,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms
    )

    # Convert sample indices to seconds
    segments = []
    for ts in speech_timestamps:
        start_sec = ts['start'] / sample_rate
        end_sec = ts['end'] / sample_rate
        segments.append((start_sec, end_sec))

    return segments


def group_segments_into_chunks(
    speech_segments: List[Tuple[float, float]],
    max_chunk_duration: float = 35.0
) -> List[Tuple[float, float]]:
    """
    Group speech segments into chunks that respect silence boundaries.

    Each chunk will be <= max_chunk_duration seconds, cutting only at
    silence points between speech segments.

    Args:
        speech_segments: List of (start_sec, end_sec) tuples from VAD.
        max_chunk_duration: Maximum duration for each chunk in seconds.

    Returns:
        List of (chunk_start, chunk_end) tuples.
    """
    if not speech_segments:
        return []

    chunks = []
    chunk_start = speech_segments[0][0]
    chunk_end = speech_segments[0][1]

    for i in range(1, len(speech_segments)):
        seg_start, seg_end = speech_segments[i]

        # Check if adding this segment would exceed max duration
        potential_duration = seg_end - chunk_start

        if potential_duration <= max_chunk_duration:
            # Extend current chunk to include this segment
            chunk_end = seg_end
        else:
            # Save current chunk and start a new one
            chunks.append((chunk_start, chunk_end))
            chunk_start = seg_start
            chunk_end = seg_end

    # Don't forget the last chunk
    chunks.append((chunk_start, chunk_end))

    return chunks


def is_degenerate_output(text: str) -> bool:
    """
    Check if transcription output is degenerate (repetitive garbage).

    This detects common ASR failure modes like repeated characters
    or extremely low character diversity.

    Args:
        text: Transcription text to check.

    Returns:
        True if the output appears to be garbage, False otherwise.
    """
    if len(text) < 10:
        return False

    # Check for excessive repetition of single characters
    char_counts = {}
    for char in text:
        char_counts[char] = char_counts.get(char, 0) + 1

    # If any single character is more than 50% of the text, it's degenerate
    max_char_ratio = max(char_counts.values()) / len(text)
    if max_char_ratio > 0.5:
        return True

    # Check for repeated patterns (e.g., "2222222" or ",,,,,,")
    if len(set(text)) < 5 and len(text) > 20:
        return True

    return False


def chunk_audio_with_vad(
    audio_path: Path,
    max_chunk_duration: float = 35.0,
    cleanup_vad: bool = True
) -> Tuple[List[Tuple[float, float]], any]:
    """
    High-level function to chunk audio using VAD.

    Loads VAD model, detects speech segments, groups into chunks,
    and optionally cleans up the VAD model.

    Args:
        audio_path: Path to the audio file.
        max_chunk_duration: Maximum duration for each chunk.
        cleanup_vad: If True, delete VAD model after use.

    Returns:
        Tuple of (chunks, vad_model) where chunks is a list of
        (start_sec, end_sec) tuples. vad_model is None if cleanup_vad=True.
    """
    from .gpu import cleanup_gpu

    print(f"   Loading Silero VAD...")
    vad_model, utils = load_silero_vad()
    get_speech_timestamps = utils[0]

    print(f"   Detecting speech segments...")
    speech_segments = detect_speech_segments(audio_path, vad_model, get_speech_timestamps)
    print(f"   Found {len(speech_segments)} speech segments")

    if not speech_segments:
        if cleanup_vad:
            del vad_model
            cleanup_gpu()
        return [], None

    chunks = group_segments_into_chunks(speech_segments, max_chunk_duration)
    print(f"   Grouped into {len(chunks)} chunks for transcription")

    if cleanup_vad:
        del vad_model
        cleanup_gpu()
        return chunks, None

    return chunks, vad_model
