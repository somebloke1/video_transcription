#!/usr/bin/env python3
"""
Video Transcription - Gemini Finisher (Hybrid)
==============================================
Local extraction + transcription, Gemini for synthesis.
Best of both worlds: local GPU for audio, cloud for visual analysis.

Pipeline:
  Video → FFmpeg → keyframes (timestamped)
        → ASR Model (local) → timestamped transcript
        → Gemini 2.5 Flash → visual analysis + synthesis

ASR Model Options (--asr flag):
  - canary: Canary-Qwen 2.5B (~5GB VRAM) - Best for technical jargon (BGP, OSPF, VXLAN)
  - whisperx: WhisperX Large V3 (~3GB VRAM) - 70x faster, word-level timestamps [DEFAULT]
  - whisper: Whisper Large V3 (~3GB VRAM) - Original, good timestamps
  - parakeet: Parakeet CTC 1.1B (~2GB VRAM) - Fast
  - granite: Granite Speech 3.3 8B (~16GB VRAM) - Best accuracy, NO TIMESTAMPS
            (images will be grouped at end of output, not placed inline)

Output Format Options (--format flag):
  - clean: Readable Markdown prose without timestamps (default)
  - html: HTML with embedded keyframe images

Keyframe Extraction (--min-stable flag):
  - Frames must be stable for at least N seconds to be captured (default: 3.0)
  - Automatically filters out transition/animation frames
  - Uses perceptual hashing to detect frame stability

Batch Processing:
  - Single file:  python process_video_gemini_finisher.py video.mp4
  - Directory:    python process_video_gemini_finisher.py ./videos/
  - List file:    python process_video_gemini_finisher.py videos.txt

Requirements:
  - FFmpeg installed
  - NVIDIA GPU
  - google-genai package: uv pip install google-genai
  - GOOGLE_API_KEY environment variable set
"""

import argparse
import sys
import os
from pathlib import Path

import torch

# Apply NeMo/transformers compatibility patch before importing NeMo
from video_transcription import apply_nemo_transformers_patch
apply_nemo_transformers_patch()

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
    extract_audio_wav,
    collect_video_files,
    assemble_html,
    format_transcript_segments,
    cleanup_gpu,
    BatchResults,
    print_batch_header,
    copy_to_all_transcripts,
    DEFAULT_OUTPUT_DIR,
    get_clean_prompt_for_gemini_with_transcript,
    get_html_prompt_for_gemini_with_transcript,
    # ASR functions
    load_asr_model,
    unload_asr_model,
    transcribe_audio,
    ASR_MODELS,
    ASR_MODEL_INFO,
)

# Check for API key
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("ERROR: GOOGLE_API_KEY environment variable not set!")
    print("Get your key at: https://aistudio.google.com/app/apikey")
    sys.exit(1)

# Initialize
gemini_client = genai.Client(api_key=GOOGLE_API_KEY)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Config (mutable)
CONFIG = {
    "asr_model": "whisperx",
    "output_format": "clean",
    "min_stable_seconds": 3.0,
    "prefix": ""
}


# =============================================================================
# Gemini Analysis
# =============================================================================

def analyze_with_gemini(keyframes: list, transcript_segments: list, output_dir: Path) -> str:
    """Send keyframes + transcript to Gemini 2.5 Flash for synthesis."""
    output_format = CONFIG["output_format"]
    print(f"\n🧠 [STAGE 2] Analyzing with Gemini 2.5 Flash (format: {output_format})...")

    if output_format == "clean":
        system_prompt = get_clean_prompt_for_gemini_with_transcript()
    else:
        system_prompt = get_html_prompt_for_gemini_with_transcript()

    contents = []

    formatted_transcript = format_transcript_segments(transcript_segments)
    contents.append(f"AUDIO TRANSCRIPT (timestamped):\n\n{formatted_transcript}\n\n")

    contents.append("KEYFRAMES (timestamped screenshots):\n")

    print(f"   📤 Uploading {len(keyframes)} keyframes...")
    for i, (ts, frame_path) in enumerate(keyframes):
        ts_str = format_timestamp(ts)
        contents.append(f"\n[{ts_str}] Screenshot {i+1} (use {{{{IMAGE_{i+1}}}}} to reference):")
        image_file = gemini_client.files.upload(file=frame_path)
        contents.append(image_file)

    if output_format == "clean":
        contents.append("\n\nPlease synthesize the transcript and visual content into a clean, readable document without timestamps or visual markers.")
    else:
        contents.append("\n\nPlease synthesize the transcript and visual content into an HTML document with image placeholders.")

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


# =============================================================================
# Main Processing
# =============================================================================

def process_video(video_path: Path, output_dir: Path, preloaded_model=None) -> bool:
    """Process a single video file."""
    try:
        video_name = video_path.stem
        work_dir = output_dir / f"{video_name}_analysis"
        work_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"📹 Processing: {video_path.name}")
        print(f"📂 Output: {work_dir}")
        print(f"🖥️ Device: {device}")
        if torch.cuda.is_available():
            print(f"🎮 GPU: {torch.cuda.get_device_name()}")
        print(f"{'='*60}")

        # Step 1: Extract keyframes
        keyframes = extract_keyframes(
            str(video_path), work_dir,
            min_stable_seconds=CONFIG["min_stable_seconds"]
        )
        keyframes = dedupe_keyframes(keyframes)

        # Step 2: Extract and transcribe audio
        audio_path = extract_audio_wav(str(video_path), work_dir)
        if audio_path:
            transcript_segments = transcribe_audio(
                audio_path,
                asr_model=CONFIG["asr_model"],
                preloaded_model=preloaded_model
            )
        else:
            print("   ⚠ No audio track, proceeding with visuals only")
            transcript_segments = []

        # Step 3: Analyze with Gemini
        result = analyze_with_gemini(keyframes, transcript_segments, work_dir)

        # Step 4: Save results
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

        # Save raw transcript
        raw_transcript = work_dir / "audio_transcript.txt"
        with open(raw_transcript, "w", encoding="utf-8") as f:
            f.write(format_transcript_segments(transcript_segments))

        # Copy to all_transcripts directory
        copy_to_all_transcripts(output_file, video_name, output_dir, raw_transcript, CONFIG["prefix"])

        print(f"\n🎉 Done!")
        print(f"   📄 Full transcript: {output_file}")
        print(f"   🎤 Audio-only transcript: {raw_transcript}")

        return True

    except Exception as e:
        print(f"\n❌ Error processing {video_path}: {e}")
        import traceback
        traceback.print_exc()
        cleanup_gpu()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Video transcription: local ASR + Gemini synthesis",
        epilog="""Examples:
  Single file:   python process_video_gemini_finisher.py video.mp4
  Directory:     python process_video_gemini_finisher.py ./videos/
  Batch file:    python process_video_gemini_finisher.py video_list.txt
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input", help="Video file, directory of videos, or .txt file with video paths")
    parser.add_argument("--output", "-o", default=DEFAULT_OUTPUT_DIR,
                        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--asr", choices=ASR_MODELS, default="whisperx",
                        help="ASR model: canary (technical jargon), whisperx (fast), whisper, parakeet, granite. "
                             "Note: granite has best accuracy but no timestamps - images will be grouped at end of output")
    parser.add_argument("--format", "-f", choices=["clean", "html"], default="clean",
                        help="Output format: clean (readable prose) or html (with embedded images)")
    parser.add_argument("--min-stable", "-s", type=float, default=3.0,
                        help="Minimum seconds a frame must be stable to be captured (default: 3.0)")
    parser.add_argument("--prefix", "-p", default="",
                        help="String to prepend to transcript filenames in all_transcripts/")

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

    CONFIG["asr_model"] = args.asr
    CONFIG["output_format"] = args.format
    CONFIG["min_stable_seconds"] = args.min_stable
    CONFIG["prefix"] = args.prefix

    format_info = {
        "clean": "Markdown (readable prose, no timestamps)",
        "html": "HTML (with embedded images)"
    }

    print("🚀 Gemini Finisher - Hybrid Pipeline")
    print(f"   - Keyframes: Stable frame detection (min {args.min_stable}s)")
    print(f"   - Transcription: {ASR_MODEL_INFO[args.asr]}")
    print("   - Visual Analysis: Gemini 2.5 Flash (cloud)")
    print("   - Synthesis: Gemini 2.5 Flash (cloud)")
    print(f"   - Output Format: {format_info[args.format]}")
    print(f"   - Videos to process: {len(video_files)}")

    results = BatchResults()

    # For batch processing (>1 video), preload ASR model
    preloaded_model = None
    if len(video_files) > 1:
        print(f"\n📦 Batch mode: Pre-loading ASR model for {len(video_files)} videos...")
        preloaded_model = load_asr_model(args.asr)
        if preloaded_model is None:
            print("   ⚠ Failed to preload model, will load per-video instead")

    try:
        for i, video_path in enumerate(video_files, 1):
            print_batch_header(video_path, i, len(video_files))

            try:
                success = process_video(video_path, output_dir, preloaded_model)
                if success:
                    results.add_success(video_path)
                else:
                    results.add_failure(video_path, "Processing returned False")
            except Exception as e:
                error_msg = f"{type(e).__name__}: {str(e)}"
                results.add_failure(video_path, error_msg)
                print(f"\n❌ Exception: {error_msg}")
                cleanup_gpu()
    finally:
        if preloaded_model is not None:
            unload_asr_model(preloaded_model)

    results.print_summary()

    if results.failed:
        results.write_error_report(output_dir)

    sys.exit(results.get_exit_code())


if __name__ == "__main__":
    main()
