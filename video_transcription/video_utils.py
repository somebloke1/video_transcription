"""
Video utility functions for FFmpeg operations.

This module provides common video/audio operations using FFmpeg.
"""

import subprocess
import shutil
import sys
from pathlib import Path
from typing import Optional


def check_ffmpeg() -> None:
    """
    Verify FFmpeg is installed and available.

    Exits the program with an error message if FFmpeg or ffprobe
    is not found in the system PATH.
    """
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        print("ERROR: FFmpeg not found. Please install it:")
        print("  Ubuntu/WSL: sudo apt install ffmpeg")
        sys.exit(1)


def get_video_duration(video_path: str) -> float:
    """
    Get video duration in seconds using ffprobe.

    Args:
        video_path: Path to the video file.

    Returns:
        Duration in seconds as a float.
    """
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", video_path],
        capture_output=True, text=True
    )
    return float(result.stdout.strip())


def format_timestamp(seconds: float) -> str:
    """
    Format seconds as HH:MM:SS or MM:SS timestamp.

    Args:
        seconds: Time in seconds.

    Returns:
        Formatted timestamp string. Returns HH:MM:SS if hours > 0,
        otherwise MM:SS.
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def extract_audio_wav(video_path: str, output_dir: Path) -> Optional[Path]:
    """
    Extract audio track as WAV file for local ASR models.

    Extracts audio at 16kHz mono, which is the expected format
    for most speech recognition models (Whisper, Canary, etc.).

    Args:
        video_path: Path to the video file.
        output_dir: Directory to save the extracted audio.

    Returns:
        Path to the extracted audio file, or None if extraction failed.
    """
    print("\n🎵 Extracting audio...")

    audio_path = output_dir / "audio.wav"

    cmd = [
        "ffmpeg", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1",
        "-y", str(audio_path)
    ]
    subprocess.run(cmd, capture_output=True)

    if audio_path.exists():
        print(f"   ✓ Extracted audio ({audio_path.stat().st_size / 1024 / 1024:.1f} MB)")
        return audio_path
    return None


def extract_audio_mp3(video_path: str, output_dir: Path) -> Optional[Path]:
    """
    Extract audio track as MP3 file for cloud API upload.

    Extracts audio as MP3 at 192kbps, suitable for uploading
    to cloud services like Gemini.

    Args:
        video_path: Path to the video file.
        output_dir: Directory to save the extracted audio.

    Returns:
        Path to the extracted audio file, or None if extraction failed.
    """
    print("\n🎵 Extracting audio track...")

    audio_path = output_dir / "audio.mp3"

    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn",  # No video
        "-acodec", "libmp3lame",
        "-ab", "192k",
        str(audio_path)
    ]

    subprocess.run(cmd, capture_output=True)

    if audio_path.exists():
        size_mb = audio_path.stat().st_size / (1024 * 1024)
        print(f"   ✓ Extracted audio ({size_mb:.1f} MB)")
        return audio_path
    else:
        print("   ⚠ No audio track found")
        return None


def extract_audio_chunk(
    audio_path: Path,
    start_sec: float,
    end_sec: float,
    output_path: Path
) -> Path:
    """
    Extract a chunk of audio using FFmpeg.

    Args:
        audio_path: Path to the source audio file.
        start_sec: Start time in seconds.
        end_sec: End time in seconds.
        output_path: Path to save the audio chunk.

    Returns:
        Path to the extracted audio chunk.
    """
    duration = end_sec - start_sec
    cmd = [
        "ffmpeg", "-y",
        "-i", str(audio_path),
        "-ss", str(start_sec),
        "-t", str(duration),
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(output_path)
    ]
    subprocess.run(cmd, capture_output=True)
    return output_path
