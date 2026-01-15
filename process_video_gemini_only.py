#!/usr/bin/env python3
"""
Video Transcription - Gemini Only
=================================
All processing done via Google Gemini 2.5 Flash (cloud).
No local GPU required - just FFmpeg for extraction.

Pipeline:
  Video → FFmpeg → stable keyframes (timestamped PNGs) + audio track
        → Gemini 2.5 Flash → transcription + visual analysis + synthesis

Output Format Options (--format flag):
  - clean: Readable Markdown prose without timestamps (default)
  - html: HTML with embedded keyframe images

Keyframe Extraction (--min-stable flag):
  - Frames must be stable for at least N seconds to be captured (default: 3.0)
  - Automatically filters out transition/animation frames
  - Uses perceptual hashing to detect frame stability

Batch Processing:
  - Single file:  python process_video_gemini_only.py video.mp4
  - Directory:    python process_video_gemini_only.py ./videos/
  - List file:    python process_video_gemini_only.py videos.txt

Requirements:
  - FFmpeg installed
  - google-genai package: uv pip install google-genai
  - imagehash package: uv pip install imagehash
  - GOOGLE_API_KEY environment variable set
"""

import argparse
import sys
import os
from pathlib import Path

try:
    from google import genai
except ImportError:
    print("ERROR: google-genai not installed")
    print("Run: uv pip install google-genai")
    sys.exit(1)

# Import from shared library
from video_transcription import (
    check_ffmpeg,
    format_timestamp,
    extract_keyframes,
    dedupe_keyframes,
    extract_audio_mp3,
    collect_video_files,
    assemble_html,
    BatchResults,
    print_batch_header,
    copy_to_all_transcripts,
    DEFAULT_OUTPUT_DIR,
    get_clean_prompt_for_gemini_with_audio,
    get_html_prompt_for_gemini_with_audio,
)

# Check for API key
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("ERROR: GOOGLE_API_KEY environment variable not set!")
    print("Get your key at: https://aistudio.google.com/app/apikey")
    sys.exit(1)

# Initialize Gemini client
gemini_client = genai.Client(api_key=GOOGLE_API_KEY)

# Config (mutable)
CONFIG = {
    "output_format": "clean",
    "min_stable_seconds": 3.0
}


def analyze_with_gemini(keyframes: list, audio_path: Path, video_name: str) -> str:
    """
    Send keyframes and audio to Gemini 2.5 Flash for analysis.
    Gemini handles both transcription and visual synthesis.
    """
    output_format = CONFIG["output_format"]
    print(f"\n🧠 Analyzing with Gemini 2.5 Flash (format: {output_format})...")

    if output_format == "clean":
        system_prompt = get_clean_prompt_for_gemini_with_audio()
    else:
        system_prompt = get_html_prompt_for_gemini_with_audio()

    contents = []

    # Add audio file first (Gemini will transcribe this)
    if audio_path and audio_path.exists():
        print(f"   📤 Uploading audio ({audio_path.stat().st_size / 1024 / 1024:.1f} MB)...")
        audio_file = gemini_client.files.upload(file=audio_path)
        contents.append("AUDIO TRACK (transcribe this completely):\n")
        contents.append(audio_file)

    # Add keyframes with timestamps
    contents.append("\n\nKEYFRAMES (timestamped screenshots from video):\n")

    print(f"   📤 Uploading {len(keyframes)} keyframes...")
    for i, (ts, frame_path) in enumerate(keyframes):
        ts_str = format_timestamp(ts)
        contents.append(f"\n[{ts_str}] Screenshot {i+1} (use {{{{IMAGE_{i+1}}}}} to reference):")
        image_file = gemini_client.files.upload(file=frame_path)
        contents.append(image_file)

    if output_format == "clean":
        contents.append("\n\nPlease transcribe the audio and synthesize with visual content into a clean, readable document without timestamps or visual markers.")
    else:
        contents.append("\n\nPlease transcribe the audio and synthesize with visual content into an HTML document with image placeholders.")

    print("   🔄 Processing...")

    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=genai.types.GenerateContentConfig(
                system_instruction=system_prompt,
                max_output_tokens=65536,
                temperature=0.1,
            )
        )

        result = response.text
        print(f"   ✓ Generated {len(result)} characters")
        return result

    except Exception as e:
        print(f"   ❌ Error: {e}")
        raise


def process_video(video_path: Path, output_dir: Path) -> bool:
    """Process a single video file."""
    try:
        video_name = video_path.stem
        work_dir = output_dir / f"{video_name}_analysis"
        work_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"📹 Processing: {video_path.name}")
        print(f"📂 Output: {work_dir}")
        print(f"{'='*60}")

        # Step 1: Extract keyframes
        keyframes = extract_keyframes(
            str(video_path), work_dir,
            min_stable_seconds=CONFIG["min_stable_seconds"]
        )
        keyframes = dedupe_keyframes(keyframes)

        # Step 2: Extract audio
        audio_path = extract_audio_mp3(str(video_path), work_dir)

        # Step 3: Analyze with Gemini (handles both transcription and synthesis)
        result = analyze_with_gemini(keyframes, audio_path, video_name)

        # Step 4: Save results based on output format
        output_format = CONFIG["output_format"]

        if output_format == "html":
            html_content = assemble_html(result, keyframes, video_name)
            output_file = work_dir / "transcript.html"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(html_content)
        else:
            output_file = work_dir / "transcript.md"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(result)

        # Copy to all_transcripts directory
        copy_to_all_transcripts(output_file, video_name, output_dir)

        print(f"\n🎉 Done!")
        print(f"   📄 Transcript: {output_file}")

        return True

    except Exception as e:
        print(f"\n❌ Error processing {video_path}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Video transcription using Gemini 2.5 Flash (cloud-only, no GPU required)",
        epilog="""Examples:
  Single file:   python process_video_gemini_only.py video.mp4
  Directory:     python process_video_gemini_only.py ./videos/
  Batch file:    python process_video_gemini_only.py video_list.txt
  HTML output:   python process_video_gemini_only.py --format html video.mp4
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input", help="Video file, directory of videos, or .txt file with video paths")
    parser.add_argument("--output", "-o", default=DEFAULT_OUTPUT_DIR,
                        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--format", "-f", choices=["clean", "html"], default="clean",
                        help="Output format: clean (readable prose) or html (with embedded images)")
    parser.add_argument("--min-stable", "-s", type=float, default=3.0,
                        help="Minimum seconds a frame must be stable to be captured (default: 3.0)")

    args = parser.parse_args()

    check_ffmpeg()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Path not found: {input_path}")
        sys.exit(1)

    video_files = collect_video_files(input_path)

    if not video_files:
        print("ERROR: No video files found to process")
        sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    CONFIG["output_format"] = args.format
    CONFIG["min_stable_seconds"] = args.min_stable

    format_info = {
        "clean": "Markdown (readable prose, no timestamps)",
        "html": "HTML (with embedded images)"
    }

    print("🚀 Gemini-Only Video Transcription")
    print(f"   - Keyframes: Stable frame detection (min {args.min_stable}s)")
    print("   - Audio extraction: FFmpeg (local)")
    print("   - Transcription: Gemini 2.5 Flash (cloud)")
    print("   - Visual Analysis: Gemini 2.5 Flash (cloud)")
    print("   - Synthesis: Gemini 2.5 Flash (cloud)")
    print(f"   - Output Format: {format_info[args.format]}")
    print(f"   - Videos to process: {len(video_files)}")
    print("   - GPU Required: None")

    results = BatchResults()

    for i, video_path in enumerate(video_files, 1):
        print_batch_header(video_path, i, len(video_files))

        try:
            success = process_video(video_path, output_dir)
            if success:
                results.add_success(video_path)
            else:
                results.add_failure(video_path, "Processing returned False")
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            results.add_failure(video_path, error_msg)
            print(f"\n❌ Exception: {error_msg}")

    results.print_summary()

    if results.failed:
        results.write_error_report(output_dir)

    sys.exit(results.get_exit_code())


if __name__ == "__main__":
    main()
