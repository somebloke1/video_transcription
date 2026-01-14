"""
Keyframe extraction and deduplication utilities.

This module provides stable keyframe extraction using perceptual hashing
to detect frames that remain unchanged for a configurable duration.
"""

import hashlib
import shutil
import subprocess
from pathlib import Path
from typing import List, Tuple

from .video_utils import get_video_duration

# Type alias for keyframe tuples: (timestamp_seconds, frame_path)
Keyframe = Tuple[float, Path]


def dedupe_keyframes(keyframes: List[Keyframe]) -> List[Keyframe]:
    """
    Remove duplicate keyframes based on image content hash.

    Uses MD5 hash of the raw image bytes to identify exact duplicates.

    Args:
        keyframes: List of (timestamp, image_path) tuples.

    Returns:
        Deduplicated list of (timestamp, image_path) tuples.
    """
    seen_hashes = set()
    unique_keyframes = []

    for ts, frame_path in keyframes:
        with open(frame_path, 'rb') as f:
            img_hash = hashlib.md5(f.read()).hexdigest()

        if img_hash not in seen_hashes:
            seen_hashes.add(img_hash)
            unique_keyframes.append((ts, frame_path))

    removed = len(keyframes) - len(unique_keyframes)
    if removed > 0:
        print(f"   Removed {removed} duplicate keyframes")

    return unique_keyframes


def extract_keyframes(
    video_path: str,
    output_dir: Path,
    min_stable_seconds: float = 3.0,
    sampling_fps: int = 2,
    hash_threshold: int = 8
) -> List[Keyframe]:
    """
    Extract stable keyframes - frames that remain unchanged for min_stable_seconds.

    This avoids capturing mid-transition/animation frames by:
    1. Extracting frames at high frequency (sampling_fps) for stability detection
    2. Comparing consecutive frames using perceptual hash
    3. Grouping similar consecutive frames
    4. Only keeping frames where group duration >= min_stable_seconds

    Args:
        video_path: Path to the video file.
        output_dir: Directory to save extracted keyframes.
        min_stable_seconds: Minimum seconds a frame must be stable to be captured.
        sampling_fps: Frames per second for stability sampling (default: 2).
        hash_threshold: Perceptual hash difference threshold (lower = stricter).

    Returns:
        List of (timestamp, frame_path) tuples for stable keyframes.
    """
    try:
        import imagehash
        from PIL import Image
    except ImportError:
        print("ERROR: imagehash not installed")
        print("Run: uv pip install imagehash Pillow")
        import sys
        sys.exit(1)

    print(f"\n📸 Extracting stable keyframes (min stable: {min_stable_seconds}s)...")

    keyframes_dir = output_dir / "keyframes"
    keyframes_dir.mkdir(parents=True, exist_ok=True)

    # Temp directory for high-frequency sampling
    temp_dir = output_dir / "temp_frames"
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Clear any existing frames
    for old_frame in keyframes_dir.glob("frame_*.png"):
        old_frame.unlink()
    for old_frame in temp_dir.glob("sample_*.png"):
        old_frame.unlink()

    # Get video duration
    duration = get_video_duration(video_path)
    print(f"   Video duration: {duration:.1f}s")

    # Extract frames at sampling_fps for stability analysis
    print(f"   Sampling frames at {sampling_fps}fps for stability analysis...")
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vf", f"fps={sampling_fps}",
        str(temp_dir / "sample_%05d.png")
    ]
    subprocess.run(cmd, capture_output=True)

    sample_frames = sorted(temp_dir.glob("sample_*.png"))
    print(f"   Extracted {len(sample_frames)} sample frames")

    if not sample_frames:
        print("   ⚠ No frames extracted, falling back to simple extraction")
        # Fallback to simple interval extraction
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-vf", "fps=0.2",  # 1 frame per 5 seconds
            str(keyframes_dir / "frame_%04d.png")
        ]
        subprocess.run(cmd, capture_output=True)
        frames = sorted(keyframes_dir.glob("frame_*.png"))
        return [(i * 5, f) for i, f in enumerate(frames)]

    # Compute perceptual hash for each frame
    print(f"   Computing perceptual hashes...")
    frame_hashes = []
    for frame_path in sample_frames:
        try:
            img = Image.open(frame_path)
            h = imagehash.phash(img)
            frame_hashes.append((frame_path, h))
        except Exception as e:
            print(f"   ⚠ Error hashing {frame_path}: {e}")
            continue

    if not frame_hashes:
        print("   ⚠ No frames could be hashed")
        shutil.rmtree(temp_dir)
        return []

    # Group consecutive similar frames
    print(f"   Grouping stable frames...")
    groups = []
    current_group = [frame_hashes[0]]

    for i in range(1, len(frame_hashes)):
        prev_path, prev_hash = frame_hashes[i-1]
        curr_path, curr_hash = frame_hashes[i]

        # Compare hashes - lower difference = more similar
        hash_diff = prev_hash - curr_hash

        if hash_diff <= hash_threshold:
            # Similar enough - add to current group
            current_group.append(frame_hashes[i])
        else:
            # Different - save current group and start new one
            groups.append(current_group)
            current_group = [frame_hashes[i]]

    # Don't forget the last group
    groups.append(current_group)

    print(f"   Found {len(groups)} distinct frame groups")

    # Filter groups by minimum stable duration
    min_frames_needed = int(min_stable_seconds * sampling_fps)
    stable_keyframes = []
    frame_idx = 0

    for group in groups:
        group_duration = len(group) / sampling_fps
        group_start_time = frame_idx / sampling_fps

        if len(group) >= min_frames_needed:
            # Take the middle frame of the stable period (most representative)
            middle_idx = len(group) // 2
            src_frame_path, _ = group[middle_idx]

            # Calculate timestamp for this frame
            timestamp = group_start_time + (middle_idx / sampling_fps)

            # Copy to keyframes directory with new name
            dest_frame_path = keyframes_dir / f"frame_{len(stable_keyframes)+1:04d}.png"
            shutil.copy(src_frame_path, dest_frame_path)

            stable_keyframes.append((timestamp, dest_frame_path))
            print(f"   ✓ Stable frame at {timestamp:.1f}s (stable for {group_duration:.1f}s)")

        frame_idx += len(group)

    # Cleanup temp directory
    shutil.rmtree(temp_dir)

    print(f"   ✓ Extracted {len(stable_keyframes)} stable keyframes")
    return stable_keyframes
