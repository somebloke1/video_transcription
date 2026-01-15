#!/usr/bin/env python3
"""
Video Transcription - Local Only (Optimized)
=============================================
All processing done locally on GPU (RTX 3090 24GB optimized).
No cloud APIs required - fully offline capable.

Pipeline:
  Video → Stable Keyframes → ASR Model → Qwen2.5-VL-7B (vision) → Qwen2.5-14B-AWQ (synthesis)

Models loaded sequentially to fit in 24GB VRAM:
  - ASR Model: Varies by selection (see --asr flag)
  - Qwen2.5-VL-7B: ~16GB (vision-language model)
  - Qwen2.5-14B-AWQ: ~10GB (4-bit quantized, better reasoning)

ASR Model Options (--asr flag):
  - canary: Canary-Qwen 2.5B (~5GB VRAM) - Best for technical jargon [DEFAULT]
  - whisperx: WhisperX Large V3 (~3GB VRAM) - 70x faster, word-level timestamps
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

Batch Processing:
  - Single file:  python process_video_local_only.py video.mp4
  - Directory:    python process_video_local_only.py ./videos/
  - List file:    python process_video_local_only.py videos.txt

Requirements:
  - NVIDIA GPU with 24GB VRAM
  - PyTorch 2.6+ with CUDA
  - transformers, autoawq, imagehash
  - For Canary: nemo_toolkit
  - For WhisperX: whisperx
"""

import argparse
import sys
import os
from pathlib import Path

# Force use of primary GPU only (for multi-GPU systems)
# Must be set BEFORE importing torch
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch

# Apply NeMo/transformers compatibility patch before importing NeMo
from video_transcription import apply_nemo_transformers_patch
apply_nemo_transformers_patch()

# Import from shared library
from video_transcription import (
    check_ffmpeg,
    format_timestamp,
    extract_keyframes,
    dedupe_keyframes,
    extract_audio_wav,
    collect_video_files,
    assemble_html_with_descriptions,
    cleanup_gpu,
    BatchResults,
    print_batch_header,
    copy_to_all_transcripts,
    DEFAULT_OUTPUT_DIR,
    get_clean_prompt_for_local_synthesis,
    get_html_prompt_for_local_synthesis,
    get_vision_analysis_prompt,
    # ASR functions
    load_asr_model,
    unload_asr_model,
    transcribe_audio,
    should_preload_asr,
    ASR_MODELS,
    ASR_MODEL_INFO,
    ASR_VRAM_GB,
)

# Check CUDA availability
if not torch.cuda.is_available():
    print("WARNING: CUDA not available. This script requires an NVIDIA GPU.")
    print("Processing will be extremely slow on CPU.")

device = "cuda" if torch.cuda.is_available() else "cpu"

# Config (mutable)
CONFIG = {
    "min_stable_seconds": 3.0,
    "output_format": "clean",
    "asr_model": "canary",  # Default to canary for technical content
}


# =============================================================================
# Vision Analysis with Qwen2.5-VL
# =============================================================================

def analyze_keyframes(keyframes: list) -> list:
    """Analyze keyframes using Qwen2.5-VL-7B."""
    print("\n👁️ [STAGE 2] Analyzing keyframes with Qwen2.5-VL-7B...")

    if torch.cuda.is_available():
        free_mem = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
        print(f"   Available VRAM: {free_mem / 1e9:.1f} GB")

    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from PIL import Image

    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        trust_remote_code=True
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="eager"
    ).to(device)
    print(f"   Model loaded on {device}")

    vision_prompt = get_vision_analysis_prompt()

    descriptions = []
    for i, (ts, frame_path) in enumerate(keyframes):
        print(f"   Analyzing keyframe {i+1}/{len(keyframes)} [{format_timestamp(ts)}]...")

        image = Image.open(frame_path)

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": vision_prompt}
            ]
        }]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=1024, do_sample=False)

        desc = processor.batch_decode(output, skip_special_tokens=True)[0]
        if "assistant\n" in desc:
            desc = desc.split("assistant\n", 1)[1]

        descriptions.append((ts, frame_path, desc.strip()))

    del model, processor
    cleanup_gpu(aggressive=True)

    # Filter out SKIP responses
    filtered = [(ts, fp, desc) for ts, fp, desc in descriptions if not desc.startswith("SKIP")]
    skipped = len(descriptions) - len(filtered)
    if skipped > 0:
        print(f"   Skipped {skipped} filler slides")

    print(f"   ✓ Analyzed {len(filtered)} content keyframes")
    return filtered


# =============================================================================
# Document Synthesis with Qwen2.5-14B-AWQ
# =============================================================================

def synthesize_document(keyframe_descriptions: list, transcript_segments: list) -> str:
    """Synthesize final document using Qwen2.5-14B-AWQ (4-bit quantized)."""
    print("\n📝 [STAGE 3] Synthesizing with Qwen2.5-14B-AWQ...")

    if torch.cuda.is_available():
        free_mem = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
        print(f"   Available VRAM: {free_mem / 1e9:.1f} GB")

    from transformers import AutoTokenizer
    from awq import AutoAWQForCausalLM

    model_id = "Qwen/Qwen2.5-14B-Instruct-AWQ"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoAWQForCausalLM.from_quantized(
        model_id,
        fuse_layers=True,
        trust_remote_code=True,
    )
    print(f"   Model loaded (4-bit quantized) on cuda:0")

    output_format = CONFIG["output_format"]

    if output_format == "clean":
        system_prompt = get_clean_prompt_for_local_synthesis()
    else:
        system_prompt = get_html_prompt_for_local_synthesis()

    # Build prompt with transcript and visual descriptions
    prompt = "## Audio Transcript:\n\n"
    for start, end, text in transcript_segments:
        prompt += f"[{format_timestamp(start)}] {text}\n"

    prompt += "\n## Visual Content (Keyframes):\n\n"
    for i, (ts, frame_path, desc) in enumerate(keyframe_descriptions, 1):
        prompt += f"### Keyframe {i} [{format_timestamp(ts)}]:\n{desc}\n\n"

    prompt += "\nSynthesize the above into a comprehensive document."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        output = model.generate(
            inputs,
            max_new_tokens=8192,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    result = tokenizer.decode(output[0], skip_special_tokens=True)

    # Extract assistant response
    if "assistant\n" in result.lower():
        result = result.split("assistant")[-1].strip()
        if result.startswith("\n"):
            result = result[1:]

    del model, tokenizer
    cleanup_gpu(aggressive=True)

    print(f"   ✓ Generated {len(result)} characters")
    return result


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
            print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"{'='*60}")

        # Step 1: Extract stable keyframes
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
            print("   ⚠ No audio track")
            transcript_segments = []

        # Step 3: Analyze keyframes with vision model
        keyframe_descriptions = analyze_keyframes(keyframes)

        # Step 4: Synthesize document
        result = synthesize_document(keyframe_descriptions, transcript_segments)

        # Step 5: Save results
        output_format = CONFIG["output_format"]

        if output_format == "html":
            html_content = assemble_html_with_descriptions(result, keyframe_descriptions, video_name)
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
            for start, end, text in transcript_segments:
                f.write(f"[{format_timestamp(start)}] {text}\n")

        # Copy to all_transcripts directory
        copy_to_all_transcripts(output_file, video_name, output_dir)

        print(f"\n🎉 Done!")
        print(f"   📄 Transcript: {output_file}")
        print(f"   🎤 Audio transcript: {raw_transcript}")

        return True

    except Exception as e:
        print(f"\n❌ Error processing {video_path}: {e}")
        import traceback
        traceback.print_exc()
        cleanup_gpu(aggressive=True)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Local-only video transcription (no cloud APIs)",
        epilog="""Examples:
  Single file:   python process_video_local_only.py video.mp4
  Directory:     python process_video_local_only.py ./videos/
  Batch file:    python process_video_local_only.py video_list.txt
  Use WhisperX:  python process_video_local_only.py --asr whisperx video.mp4
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input", help="Video file, directory, or .txt file with paths")
    parser.add_argument("--output", "-o", default=DEFAULT_OUTPUT_DIR,
                        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--asr", choices=ASR_MODELS, default="canary",
                        help="ASR model: canary (technical jargon), whisperx (fast), whisper, parakeet, granite. "
                             "Note: granite has best accuracy but no timestamps - images will be grouped at end of output")
    parser.add_argument("--format", "-f", choices=["clean", "html"], default="clean",
                        help="Output format: clean (markdown) or html (with images)")
    parser.add_argument("--min-stable", "-s", type=float, default=3.0,
                        help="Minimum seconds a frame must be stable (default: 3.0)")

    args = parser.parse_args()

    check_ffmpeg()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Path not found: {input_path}")
        sys.exit(1)

    video_files = collect_video_files(input_path)

    if not video_files:
        print("ERROR: No video files found")
        sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    CONFIG["output_format"] = args.format
    CONFIG["min_stable_seconds"] = args.min_stable
    CONFIG["asr_model"] = args.asr

    format_info = {
        "clean": "Markdown (readable prose, no timestamps)",
        "html": "HTML (with embedded images)"
    }

    print("🚀 Local-Only Video Transcription (Optimized)")
    print(f"   - ASR: {ASR_MODEL_INFO[args.asr]}")
    print("   - Vision: Qwen2.5-VL-7B (~16GB VRAM)")
    print("   - Synthesis: Qwen2.5-14B-AWQ 4-bit (~10GB VRAM)")
    print(f"   - Keyframes: Stable detection (min {args.min_stable}s)")
    print(f"   - Output: {format_info[args.format]}")
    print(f"   - Videos: {len(video_files)}")
    print("   - Cloud APIs: None (fully offline)")

    results = BatchResults()

    # For batch processing (>1 video), preload ASR model if it fits with vision model
    # Qwen2.5-VL-7B needs ~16GB VRAM for vision analysis
    preloaded_model = None
    if len(video_files) > 1:
        if should_preload_asr(args.asr, next_stage_vram_gb=16.0):
            print(f"\n📦 Batch mode: Pre-loading ASR model for {len(video_files)} videos...")
            preloaded_model = load_asr_model(args.asr)
            if preloaded_model is None:
                print("   ⚠ Failed to preload model, will load per-video instead")
        else:
            asr_vram = ASR_VRAM_GB.get(args.asr, 5.0)
            print(f"\n📦 Batch mode: {args.asr} ({asr_vram:.0f}GB) too large to coexist with vision model (~16GB)")
            print(f"   Will load/unload ASR per-video to avoid OOM")

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
                cleanup_gpu(aggressive=True)
    finally:
        if preloaded_model is not None:
            unload_asr_model(preloaded_model)

    results.print_summary()

    if results.failed:
        results.write_error_report(output_dir)

    sys.exit(results.get_exit_code())


if __name__ == "__main__":
    main()
