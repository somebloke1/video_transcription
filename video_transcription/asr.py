"""
Automatic Speech Recognition (ASR) module.

This module provides a unified interface for multiple ASR backends:
  - canary: Canary-Qwen 2.5B (~5GB VRAM) - Best for technical jargon
  - whisperx: WhisperX Large V3 (~3GB VRAM) - 70x faster, word-level timestamps
  - whisper: Whisper Large V3 (~3GB VRAM) - Original OpenAI model
  - parakeet: Parakeet CTC 1.1B (~2GB VRAM) - Fast, NVIDIA
  - granite: Granite Speech 3.3 8B (~16GB VRAM) - Best accuracy, IBM

All functions support optional model preloading for batch processing efficiency.
"""

import shutil
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Any

import torch

from .gpu import cleanup_gpu
from .vad import (
    load_silero_vad,
    detect_speech_segments,
    group_segments_into_chunks,
    is_degenerate_output,
)
from .video_utils import extract_audio_chunk

# Type alias for transcript segments: (start_sec, end_sec, text)
TranscriptSegment = Tuple[float, float, str]

# Available ASR models
ASR_MODELS = ["canary", "whisperx", "whisper", "parakeet", "granite"]

# Model info for display
ASR_MODEL_INFO = {
    "canary": "Canary-Qwen 2.5B (~5GB VRAM, best for technical jargon)",
    "whisperx": "WhisperX Large V3 (~3GB VRAM, 70x faster, word-level timestamps)",
    "whisper": "Whisper Large V3 (~3GB VRAM, segment timestamps)",
    "parakeet": "Parakeet CTC 1.1B (~2GB VRAM, fast)",
    "granite": "Granite Speech 3.3 8B (~16GB VRAM, best accuracy, no timestamps)",
}

# VRAM requirements in GB (approximate)
ASR_VRAM_GB = {
    "canary": 5.0,
    "whisperx": 3.0,
    "whisper": 3.0,
    "parakeet": 2.0,
    "granite": 16.0,
}


def get_device() -> str:
    """Get the compute device."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def should_preload_asr(asr_model: str, next_stage_vram_gb: float = 16.0) -> bool:
    """
    Determine if ASR model should be preloaded for batch processing.

    Preloading keeps the ASR model in VRAM between videos, saving load time.
    However, it's only beneficial if the ASR model can coexist with the next
    processing stage (typically vision model) in VRAM.

    Args:
        asr_model: The ASR model name.
        next_stage_vram_gb: VRAM required by next stage (default: 16GB for Qwen2.5-VL-7B).

    Returns:
        True if preloading is safe and beneficial, False otherwise.
    """
    asr_vram = ASR_VRAM_GB.get(asr_model, 5.0)  # Default to 5GB if unknown
    total_needed = asr_vram + next_stage_vram_gb

    # Get total VRAM (not just available, since we're planning ahead)
    if not torch.cuda.is_available():
        return False

    try:
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    except Exception:
        return False

    # Add 2GB buffer for safety
    can_preload = (total_needed + 2.0) <= total_vram

    return can_preload


# =============================================================================
# Model Loading/Unloading
# =============================================================================

def load_asr_model(asr_type: str) -> Optional[Any]:
    """
    Pre-load ASR model for persistent use across batch processing.

    Args:
        asr_type: One of 'canary', 'whisperx', 'whisper', 'parakeet', 'granite'.

    Returns:
        Model object (or tuple with processor for some models), or None on failure.
    """
    device = get_device()

    if asr_type == "canary":
        print("🔄 Pre-loading Canary-Qwen 2.5B model...")
        try:
            from nemo.collections.speechlm2.models import SALM
        except ImportError:
            try:
                from nemo.collections.asr.models import EncDecMultiTaskModel as SALM
            except ImportError:
                print("ERROR: Could not import Canary model")
                print('Run: uv pip install "nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git"')
                return None

        try:
            model = SALM.from_pretrained('nvidia/canary-qwen-2.5b').bfloat16().eval().to(device)
            print(f"   ✓ Canary model loaded on {device}")
            return model
        except Exception as e:
            print(f"   from_pretrained failed: {e}")
            print(f"   ❌ Failed to load Canary model")
            return None

    elif asr_type == "whisperx":
        print("🔄 Pre-loading WhisperX Large V3 model...")
        try:
            import whisperx
        except ImportError:
            print("ERROR: whisperx not installed")
            print("Run: uv pip install whisperx")
            return None

        compute_type = "float16" if device == "cuda" else "int8"
        model = whisperx.load_model("large-v3", device, compute_type=compute_type)
        print(f"   ✓ WhisperX model loaded on {device}")
        return model

    elif asr_type == "whisper":
        print("🔄 Pre-loading Whisper Large V3 model...")
        try:
            import whisper
        except ImportError:
            print("ERROR: openai-whisper not installed")
            print("Run: uv pip install openai-whisper")
            return None

        model = whisper.load_model("large-v3").to(device)
        print(f"   ✓ Whisper model loaded on {device}")
        return model

    elif asr_type == "parakeet":
        print("🔄 Pre-loading Parakeet CTC 1.1B model...")
        try:
            from transformers import pipeline
        except ImportError:
            print("ERROR: transformers not installed")
            print("Run: uv pip install transformers torchaudio")
            return None

        pipe = pipeline(
            "automatic-speech-recognition",
            model="nvidia/parakeet-ctc-1.1b",
            device=device
        )
        print(f"   ✓ Parakeet model loaded on {device}")
        return pipe

    elif asr_type == "granite":
        print("🔄 Pre-loading Granite Speech 3.3 8B model...")
        try:
            from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
        except ImportError:
            print("ERROR: transformers not installed")
            print("Run: uv pip install transformers>=4.52.4 torchaudio peft soundfile")
            return None

        model_id = "ibm-granite/granite-speech-3.3-8b"
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        print(f"   ✓ Granite model loaded")
        return (model, processor)

    else:
        print(f"ERROR: Unknown ASR model: {asr_type}")
        return None


def unload_asr_model(model: Any) -> None:
    """
    Unload ASR model and free GPU memory.

    Args:
        model: The model object to unload.
    """
    if model is not None:
        if isinstance(model, tuple):
            for m in model:
                del m
        else:
            del model
    cleanup_gpu()
    print("🧹 ASR model unloaded")


# =============================================================================
# Transcription Functions
# =============================================================================

def transcribe_with_canary(
    audio_path: Path,
    preloaded_model: Optional[Any] = None
) -> List[TranscriptSegment]:
    """
    Transcribe with Canary-Qwen 2.5B (~5GB VRAM, best for technical jargon).

    Uses VAD-based chunking for audio longer than 35 seconds.

    Args:
        audio_path: Path to the audio file.
        preloaded_model: Optional pre-loaded model for batch efficiency.

    Returns:
        List of (start_sec, end_sec, text) tuples.
    """
    device = get_device()
    print("\n🎤 [ASR] Transcribing with Canary-Qwen 2.5B (optimized for technical terminology)...")

    model_was_preloaded = preloaded_model is not None

    if preloaded_model is not None:
        model = preloaded_model
        print(f"   Using preloaded model on {device}")
    else:
        try:
            from nemo.collections.speechlm2.models import SALM
        except ImportError:
            try:
                from nemo.collections.asr.models import EncDecMultiTaskModel as SALM
            except ImportError:
                print("ERROR: Could not import Canary model")
                print('Try: uv pip install "nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git"')
                sys.exit(1)

        try:
            model = SALM.from_pretrained('nvidia/canary-qwen-2.5b').bfloat16().eval().to(device)
        except Exception as e:
            print(f"   Failed to load Canary: {e}")
            print("   Falling back to Whisper...")
            return transcribe_with_whisper(audio_path)

        print(f"   Model loaded on {device} (SALM architecture with Qwen LLM)")

    import torchaudio
    waveform, sample_rate = torchaudio.load(str(audio_path))
    audio_duration = waveform.shape[1] / sample_rate
    print(f"   Audio duration: {audio_duration:.1f}s")

    # For short audio (<= 35s), transcribe directly without chunking
    if audio_duration <= 35.0:
        print(f"   Short audio, transcribing directly...")
        answer_ids = model.generate(
            prompts=[[{
                "role": "user",
                "content": f"Transcribe the following: {model.audio_locator_tag}",
                "audio": [str(audio_path)]
            }]],
            max_new_tokens=256,
        )
        full_text = model.tokenizer.ids_to_text(answer_ids[0].cpu()).strip()

        if not model_was_preloaded:
            del model
            cleanup_gpu(aggressive=True)

        if full_text and not is_degenerate_output(full_text):
            return [(0.0, audio_duration, full_text)]
        return []

    # For long audio, use VAD-based chunking
    print(f"   Long audio detected, using VAD-based chunking...")

    print(f"   Loading Silero VAD...")
    vad_model, utils = load_silero_vad()
    get_speech_timestamps = utils[0]

    print(f"   Detecting speech segments...")
    speech_segments = detect_speech_segments(audio_path, vad_model, get_speech_timestamps)
    print(f"   Found {len(speech_segments)} speech segments")

    if not speech_segments:
        print(f"   ⚠ No speech detected in audio")
        if not model_was_preloaded:
            del model
        del vad_model
        cleanup_gpu(aggressive=True)
        return []

    chunks = group_segments_into_chunks(speech_segments, max_chunk_duration=35.0)
    print(f"   Grouped into {len(chunks)} chunks for transcription")

    del vad_model
    cleanup_gpu()

    temp_dir = audio_path.parent / "temp_chunks"
    temp_dir.mkdir(exist_ok=True)

    segments = []
    for i, (chunk_start, chunk_end) in enumerate(chunks):
        chunk_duration = chunk_end - chunk_start
        print(f"   Transcribing chunk {i+1}/{len(chunks)} ({chunk_start:.1f}s - {chunk_end:.1f}s, {chunk_duration:.1f}s)...")

        chunk_path = temp_dir / f"chunk_{i:04d}.wav"
        extract_audio_chunk(audio_path, chunk_start, chunk_end, chunk_path)

        try:
            answer_ids = model.generate(
                prompts=[[{
                    "role": "user",
                    "content": f"Transcribe the following: {model.audio_locator_tag}",
                    "audio": [str(chunk_path)]
                }]],
                max_new_tokens=256,
            )
            chunk_text = model.tokenizer.ids_to_text(answer_ids[0].cpu()).strip()

            if chunk_text and not is_degenerate_output(chunk_text):
                segments.append((chunk_start, chunk_end, chunk_text))
            else:
                print(f"      ⚠ Chunk {i+1} produced empty or degenerate output, skipping")
        except Exception as e:
            print(f"      ⚠ Error transcribing chunk {i+1}: {e}")

        if chunk_path.exists():
            chunk_path.unlink()

    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    if not model_was_preloaded:
        del model
        cleanup_gpu(aggressive=True)

    print(f"   ✓ Transcribed {len(segments)} segments (technical terminology optimized)")
    return segments


def transcribe_with_whisperx(
    audio_path: Path,
    preloaded_model: Optional[Any] = None
) -> List[TranscriptSegment]:
    """
    Transcribe with WhisperX Large V3 (~3GB VRAM, 70x faster, word-level timestamps).

    Args:
        audio_path: Path to the audio file.
        preloaded_model: Optional pre-loaded model for batch efficiency.

    Returns:
        List of (start_sec, end_sec, text) tuples.
    """
    device = get_device()
    print("\n🎤 [ASR] Transcribing with WhisperX Large V3 (70x faster)...")

    try:
        import whisperx
    except ImportError:
        print("ERROR: whisperx not installed")
        print("Run: uv pip install whisperx")
        sys.exit(1)

    model_was_preloaded = preloaded_model is not None

    if preloaded_model is not None:
        model = preloaded_model
        print(f"   Using preloaded model on {device}")
    else:
        compute_type = "float16" if device == "cuda" else "int8"
        model = whisperx.load_model("large-v3", device, compute_type=compute_type)
        print(f"   Model loaded on {device} (compute_type={compute_type})")

    audio = whisperx.load_audio(str(audio_path))

    print("   Transcribing (batched)...")
    result = model.transcribe(audio, batch_size=16)

    print("   Aligning for word-level timestamps...")
    model_a, metadata = whisperx.load_align_model(
        language_code=result["language"],
        device=device
    )
    result = whisperx.align(
        result["segments"],
        model_a,
        metadata,
        audio,
        device,
        return_char_alignments=False
    )

    segments = []
    for seg in result.get("segments", []):
        segments.append((seg["start"], seg["end"], seg["text"].strip()))

    del model_a
    if not model_was_preloaded:
        del model
    cleanup_gpu()

    print(f"   ✓ Transcribed {len(segments)} segments with word-level alignment")
    return segments


def transcribe_with_whisper(
    audio_path: Path,
    preloaded_model: Optional[Any] = None
) -> List[TranscriptSegment]:
    """
    Transcribe with Whisper Large V3 (~3GB VRAM, original).

    Args:
        audio_path: Path to the audio file.
        preloaded_model: Optional pre-loaded model for batch efficiency.

    Returns:
        List of (start_sec, end_sec, text) tuples.
    """
    device = get_device()
    print("\n🎤 [ASR] Transcribing with Whisper Large V3...")

    try:
        import whisper
    except ImportError:
        print("ERROR: openai-whisper not installed")
        print("Run: uv pip install openai-whisper")
        sys.exit(1)

    model_was_preloaded = preloaded_model is not None

    if preloaded_model is not None:
        model = preloaded_model
        print(f"   Using preloaded model on {device}")
    else:
        model = whisper.load_model("large-v3").to(device)
        print(f"   Model loaded on {device}")

    result = model.transcribe(
        str(audio_path),
        fp16=(device == "cuda"),
        language="en",
        verbose=False
    )

    segments = []
    for seg in result.get("segments", []):
        segments.append((seg["start"], seg["end"], seg["text"].strip()))

    if not model_was_preloaded:
        del model
        cleanup_gpu()

    print(f"   ✓ Transcribed {len(segments)} segments")
    return segments


def transcribe_with_parakeet(
    audio_path: Path,
    preloaded_model: Optional[Any] = None
) -> List[TranscriptSegment]:
    """
    Transcribe with Parakeet CTC 1.1B (~2GB VRAM, 6x faster).

    Args:
        audio_path: Path to the audio file.
        preloaded_model: Optional pre-loaded model for batch efficiency.

    Returns:
        List of (start_sec, end_sec, text) tuples.
    """
    device = get_device()
    print("\n🎤 [ASR] Transcribing with Parakeet CTC 1.1B...")

    try:
        from transformers import pipeline
        import torchaudio
    except ImportError:
        print("ERROR: Required packages not installed")
        print("Run: uv pip install transformers torchaudio")
        sys.exit(1)

    model_was_preloaded = preloaded_model is not None

    waveform, sample_rate = torchaudio.load(str(audio_path))
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)

    if preloaded_model is not None:
        pipe = preloaded_model
        print(f"   Using preloaded model on {device}")
    else:
        pipe = pipeline(
            "automatic-speech-recognition",
            model="nvidia/parakeet-ctc-1.1b",
            device=device
        )
        print(f"   Model loaded on {device}")

    audio_duration = waveform.shape[1] / 16000
    chunk_duration = 30

    segments = []
    for start in range(0, int(audio_duration), chunk_duration):
        end = min(start + chunk_duration, audio_duration)
        start_sample = int(start * 16000)
        end_sample = int(end * 16000)

        chunk = waveform[:, start_sample:end_sample].squeeze().numpy()
        result = pipe(chunk)
        text = result.get("text", "").strip()

        if text:
            segments.append((float(start), float(end), text))

    if not model_was_preloaded:
        del pipe
        cleanup_gpu()

    print(f"   ✓ Transcribed {len(segments)} segments")
    return segments


def transcribe_with_granite(
    audio_path: Path,
    preloaded_model: Optional[Any] = None
) -> List[TranscriptSegment]:
    """
    Transcribe with IBM Granite Speech 3.3 8B (~16GB VRAM, best accuracy).

    Granite Speech uses a chat template with <|audio|> token placeholder.
    It can handle arbitrary length audio without chunking.

    Args:
        audio_path: Path to the audio file.
        preloaded_model: Optional pre-loaded tuple of (model, processor).

    Returns:
        List of (start_sec, end_sec, text) tuples.
    """
    device = get_device()
    print("\n🎤 [ASR] Transcribing with Granite Speech 3.3 8B...")

    try:
        from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
        import torchaudio
    except ImportError:
        print("ERROR: Required packages not installed")
        print("Run: uv pip install transformers>=4.52.4 torchaudio peft soundfile")
        sys.exit(1)

    model_was_preloaded = preloaded_model is not None

    if preloaded_model is not None:
        model, processor = preloaded_model
        print(f"   Using preloaded model")
    else:
        model_id = "ibm-granite/granite-speech-3.3-8b"
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        print(f"   Model loaded")

    # Load audio - must be mono 16kHz
    waveform, sample_rate = torchaudio.load(str(audio_path), normalize=True)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    audio_duration = waveform.shape[1] / 16000
    print(f"   Audio duration: {audio_duration:.1f}s")

    # Granite Speech uses chat template with <|audio|> placeholder
    tokenizer = processor.tokenizer
    system_prompt = "You are a helpful AI assistant that transcribes speech accurately."
    user_prompt = "<|audio|>Transcribe the speech into a written format."

    chat = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

    # Process audio and prompt together
    model_inputs = processor(prompt, waveform, device=device, return_tensors="pt").to(device)

    print(f"   Generating transcription...")
    with torch.no_grad():
        model_outputs = model.generate(
            **model_inputs,
            max_new_tokens=2048,
            do_sample=False,
            num_beams=1
        )

    # Extract only the new tokens (skip input tokens)
    num_input_tokens = model_inputs["input_ids"].shape[-1]
    new_tokens = model_outputs[0, num_input_tokens:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    if not model_was_preloaded:
        del model, processor
        cleanup_gpu()

    if text:
        print(f"   ✓ Transcribed {len(text)} characters")
        return [(0.0, audio_duration, text)]
    else:
        print(f"   ⚠ No transcription generated")
        return []


# =============================================================================
# Dispatcher
# =============================================================================

def transcribe_audio(
    audio_path: Path,
    asr_model: str = "whisperx",
    preloaded_model: Optional[Any] = None
) -> List[TranscriptSegment]:
    """
    Transcribe audio using the specified ASR model.

    Args:
        audio_path: Path to the audio file.
        asr_model: One of 'canary', 'whisperx', 'whisper', 'parakeet', 'granite'.
        preloaded_model: Optional pre-loaded model for batch efficiency.

    Returns:
        List of (start_sec, end_sec, text) tuples.

    Raises:
        SystemExit: If an unknown ASR model is specified.
    """
    if asr_model == "canary":
        return transcribe_with_canary(audio_path, preloaded_model)
    elif asr_model == "whisperx":
        return transcribe_with_whisperx(audio_path, preloaded_model)
    elif asr_model == "whisper":
        return transcribe_with_whisper(audio_path, preloaded_model)
    elif asr_model == "parakeet":
        return transcribe_with_parakeet(audio_path, preloaded_model)
    elif asr_model == "granite":
        return transcribe_with_granite(audio_path, preloaded_model)
    else:
        print(f"ERROR: Unknown ASR model: {asr_model}")
        print(f"Available models: {', '.join(ASR_MODELS)}")
        sys.exit(1)
