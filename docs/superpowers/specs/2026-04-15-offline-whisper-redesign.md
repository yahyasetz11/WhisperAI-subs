# Offline Whisper Transcription Redesign

**Date:** 2026-04-15
**Status:** Approved

## Problem

The current whisper.py has three issues:

1. **Inaccurate transcription** — OpenAI's Whisper API (`whisper-1`) produces poor Japanese transcription, especially for conversational audio (e.g., Sakurazaka46 variety shows).
2. **Timestamps rounded to 2 seconds** — Every segment in the output SRT is exactly 2 seconds with zero millisecond precision, regardless of actual speech duration.
3. **Overly formal Indonesian translation** — GPT output is stiff ("Saya juga pernah mengalami hal seperti itu") and the `make_natural_indonesian()` find-replace dictionary is too crude to fix it.

## Solution

### 1. Replace OpenAI Whisper API with Local kotoba-whisper-v2.0

**Model:** `kotoba-tech/kotoba-whisper-v2.0` from HuggingFace — a Japanese-specific fine-tuned Whisper model.

**Library:** `faster-whisper` — runs Whisper models via CTranslate2, optimized for GPU inference.

**Configuration:**
- Device: CUDA (NVIDIA GPU)
- Compute type: float16
- VAD filter: enabled (Silero VAD) — detects speech boundaries for accurate sentence-level timestamps
- Language: `ja` (Japanese)

**Transcription call:**
```python
from faster_whisper import WhisperModel

model = WhisperModel("kotoba-tech/kotoba-whisper-v2.0", device="cuda", compute_type="float16")
segments, info = model.transcribe(audio_path, language="ja", vad_filter=True)
```

On first run, the model downloads from HuggingFace and converts to CTranslate2 format. Cached locally after that.

### 2. Translation Prompt Overhaul

**Remove:** `make_natural_indonesian()` function entirely.

**New GPT system prompt:**
```
Kamu adalah penerjemah subtitle dari bahasa Jepang ke bahasa Indonesia.
Terjemahkan setiap dialog dalam tanda [Dialog X] ke bahasa Indonesia.

Aturan gaya bahasa:
- Gunakan "aku/kamu" bukan "saya/Anda"
- Pakai bahasa sehari-hari yang santai seperti ngobrol sama teman
- Hindari bahasa baku/formal (jangan pakai "telah", "namun", "dapat", dll)
- Hindari slang berat Jakarta (jangan pakai "gue/lu", "anjir", dll)
- Singkat dan natural, seperti subtitle anime fansub
- Jaga konteks antar dialog agar cerita tetap nyambung

Pertahankan format [Dialog X] agar bisa dicocokkan kembali.
```

**Unchanged:** Batch processing (5 dialogs/batch), `[Dialog X]` parsing, GPT-4o model (configurable), temperature 0.6.

### 3. Method Consolidation

**Before (4 methods):** `transcribe`, `translate`, `transcribe-only`, `translate-srt`

**After (3 methods):**

| Method | Transcription | Translation | Use case |
|---|---|---|---|
| `transcribe` | Local kotoba-whisper | GPT → Indonesian | Main workflow (default) |
| `transcribe-only` | Local kotoba-whisper | None | Japanese SRT only |
| `translate-srt` | None (reads SRT) | GPT → Indonesian | Translate existing SRT |

**Removed:** `translate` method — used OpenAI's `audio.translations` endpoint which didn't work well and is incompatible with local transcription.

### 4. CLI Argument Changes

| Argument | Before | After |
|---|---|---|
| `--whisper-model` | `"whisper-1"` | `"kotoba-tech/kotoba-whisper-v2.0"` (HuggingFace name or local path) |
| `--method` | 4 choices | 3 choices (remove `translate`) |
| `--api_key` | Required for all | Only for `transcribe` and `translate-srt` |
| `--device` | N/A (new) | `"cuda"` / `"cpu"` (default: `"cuda"`) |
| `--compute-type` | N/A (new) | `"float16"` / `"int8"` / `"float32"` (default: `"float16"`) |

OpenAI client is only initialized when GPT translation is needed (not for `transcribe-only`).

### 5. Dependencies

**Add:**
- `faster-whisper` (includes CTranslate2, Silero VAD)
- `torch` + `torchaudio` (CUDA support)

**Keep:**
- `openai` (for GPT translation)
- `configparser`

### 6. README.md Update

Update to document:
- New offline transcription setup
- CUDA/GPU requirements
- Updated installation instructions (pip install with CUDA torch)
- Updated CLI usage examples for all 3 methods
- Dependency list

## Out of Scope

- Speaker diarization (identifying who is speaking)
- Audio splitting (handled by existing `split_audio.py`)
- Changing the SRT format
- Replacing GPT with a local translation model
