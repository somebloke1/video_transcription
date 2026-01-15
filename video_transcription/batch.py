"""
Batch processing utilities for video transcription.

This module provides functions for collecting video files from various
input sources and tracking batch processing results.
"""

import shutil
import sys
from pathlib import Path
from typing import List, Tuple, Optional

# Supported video file extensions
VIDEO_EXTENSIONS = ['.mp4', '.mkv', '.webm', '.avi', '.mov']

# Default output directory
DEFAULT_OUTPUT_DIR = "./transcriptions"


def collect_video_files(input_path: Path) -> List[Path]:
    """
    Collect video files from input path.

    Supports:
    - Single video file
    - Directory of videos (non-recursive)
    - Text file with paths (one per line, # for comments)

    Args:
        input_path: Path to video file, directory, or list file.

    Returns:
        List of Path objects for video files to process.

    Exits:
        With error if input path doesn't exist or is unsupported.
    """
    video_files = []

    if input_path.is_file():
        # Check if it's a video file or a list file
        if input_path.suffix.lower() in VIDEO_EXTENSIONS:
            video_files.append(input_path)
        elif input_path.suffix.lower() in ['.txt', '.list']:
            # Read list of paths from file
            print(f"📋 Reading video list from: {input_path}")
            with open(input_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        path = Path(line)
                        if path.exists() and path.suffix.lower() in VIDEO_EXTENSIONS:
                            video_files.append(path)
                        elif path.exists():
                            print(f"   ⚠ Skipping non-video: {path}")
                        else:
                            print(f"   ⚠ File not found: {path}")
        else:
            print(f"ERROR: Unsupported input file type: {input_path.suffix}")
            sys.exit(1)

    elif input_path.is_dir():
        # Collect all video files in directory
        print(f"📁 Scanning directory: {input_path}")
        for ext in VIDEO_EXTENSIONS:
            video_files.extend(input_path.glob(f"*{ext}"))
            video_files.extend(input_path.glob(f"*{ext.upper()}"))
        video_files = sorted(set(video_files))  # Dedupe and sort

    else:
        print(f"ERROR: Path not found: {input_path}")
        sys.exit(1)

    return video_files


class BatchResults:
    """
    Track batch processing results.

    Provides methods for recording successes/failures and
    generating reports.
    """

    def __init__(self):
        self.success: List[Path] = []
        self.failed: List[Tuple[Path, str]] = []

    def add_success(self, path: Path) -> None:
        """Record a successful processing."""
        self.success.append(path)

    def add_failure(self, path: Path, error: str) -> None:
        """Record a failed processing with error message."""
        self.failed.append((path, error))

    @property
    def total(self) -> int:
        """Total number of processed files."""
        return len(self.success) + len(self.failed)

    def print_summary(self) -> None:
        """Print batch processing summary to console."""
        print(f"\n{'='*60}")
        print("📊 BATCH PROCESSING SUMMARY")
        print(f"{'='*60}")
        print(f"   ✓ Successful: {len(self.success)}")
        print(f"   ✗ Failed: {len(self.failed)}")
        print(f"   Total: {self.total}")

        if self.success:
            print(f"\n✓ Successfully processed:")
            for path in self.success:
                print(f"   - {path.name}")

        if self.failed:
            print(f"\n✗ Failed with errors:")
            for path, error in self.failed:
                print(f"   - {path.name}")
                print(f"     Error: {error}")

    def write_error_report(self, output_dir: Path) -> Path:
        """
        Write error report to file.

        Args:
            output_dir: Directory to write the report.

        Returns:
            Path to the error report file.
        """
        report_path = output_dir / "batch_errors.txt"
        with open(report_path, "w") as f:
            f.write(f"Batch Processing Error Report\n")
            f.write(f"{'='*60}\n\n")
            for path, error in self.failed:
                f.write(f"File: {path}\n")
                f.write(f"Error: {error}\n\n")
        print(f"\n📄 Error report saved to: {report_path}")
        return report_path

    def get_exit_code(self) -> int:
        """
        Get appropriate exit code based on results.

        Returns:
            0 if all succeeded, 1 if all failed, 2 if partial failure.
        """
        if not self.failed:
            return 0
        elif not self.success:
            return 1
        else:
            return 2


def print_batch_header(video_path: Path, index: int, total: int) -> None:
    """
    Print header for batch processing iteration.

    Args:
        video_path: Path to current video.
        index: Current video index (1-based).
        total: Total number of videos.
    """
    print(f"\n{'='*60}")
    print(f"📹 [{index}/{total}] {video_path.name}")
    print(f"{'='*60}")


def copy_to_all_transcripts(
    transcript_path: Path,
    video_name: str,
    output_dir: Path,
    audio_transcript_path: Optional[Path] = None,
    prefix: str = ""
) -> Optional[Path]:
    """
    Copy transcripts to the all_transcripts directory with standardized naming.

    Creates a flat directory structure with all transcripts in one place
    for easy access after batch processing. Full and audio-only transcripts
    are placed in separate subdirectories.

    Args:
        transcript_path: Path to the transcript file (html or md).
        video_name: Original video filename (without extension).
        output_dir: Base output directory (e.g., ./transcriptions).
        audio_transcript_path: Optional path to raw audio transcript.
        prefix: Optional string to prepend to all filenames.

    Returns:
        Path to the copied full transcript, or None if copy failed.

    Output structure:
        all_transcripts/
        ├── full/
        │   └── <prefix><video_name>__transcript.<ext>
        └── audio_only/
            └── <prefix><video_name>__transcript.txt
    """
    all_transcripts_dir = output_dir / "all_transcripts"
    full_dir = all_transcripts_dir / "full"
    audio_dir = all_transcripts_dir / "audio_only"

    full_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Copy full transcript
        ext = transcript_path.suffix
        dest_name = f"{prefix}{video_name}__transcript{ext}"
        dest_path = full_dir / dest_name
        shutil.copy2(transcript_path, dest_path)

        # Copy audio-only transcript if provided
        if audio_transcript_path and audio_transcript_path.exists():
            audio_dest_name = f"{prefix}{video_name}__transcript.txt"
            audio_dest_path = audio_dir / audio_dest_name
            shutil.copy2(audio_transcript_path, audio_dest_path)

        return dest_path

    except Exception as e:
        print(f"   ⚠ Failed to copy to all_transcripts: {e}")
        return None
