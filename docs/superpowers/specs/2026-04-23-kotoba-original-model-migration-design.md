# Design: Migrate kotoba notebook from CT2 to original kotoba-whisper-v2.0

**Date:** 2026-04-23
**Branch:** feat/colab-notebook
**File:** `whispersubs_colab_kotoba.ipynb`

## Problem

The current kotoba notebook uses `jctv-tech/kotoba-whisper-v21-ct2` loaded via `faster-whisper`. This is a CTranslate2-quantized version of kotoba-whisper. Transcription accuracy on Japanese variety show audio is poor — the quantization trades accuracy for speed in a way that is not acceptable for this use case.

## Goal

Replace the CT2 model with the original `kotoba-tech/kotoba-whisper-v2.0` HuggingFace Transformers model to recover transcription accuracy. Accept the trade-off of slower inference on T4 GPU.

## Target environment

- Google Colab, T4 GPU (15GB VRAM), float16
- Runtime: single session (model cached in memory between cells)

## Cell structure

Only Cells 0, 1, 2, and 4 change. All other cells are untouched.

| Cell | Status | Summary of change |
|------|--------|-------------------|
| Cell 0 | Changed | Update header model name |
| Cell 1 | Changed | Replace `faster-whisper` with `transformers torch torchaudio silero-vad soundfile` |
| Cell 2 | Changed | Add `USE_VAD` toggle; remove `WHISPER_MODEL` variable |
| Cell 3 | Unchanged | Audio upload |
| Cell 4 | Rewritten | New transcription engine (see below) |
| Cell 4b | Unchanged | Raw Japanese SRT export |
| Cell 5 | Unchanged | GPT-4o batch translation |
| Cell 6 | Unchanged | SRT generation and download |

## Cell 1: Dependencies

```python
!pip install -q transformers torch torchaudio silero-vad soundfile openai
```

Removes `faster-whisper`. Adds `transformers`, `torchaudio`, `silero-vad`, `soundfile`.

## Cell 2: Configuration

New `USE_VAD` boolean toggle added. `WHISPER_MODEL` variable removed (model identity is fixed in Cell 4).

```python
USE_VAD = True   # Set False to skip VAD and feed full audio directly
```

Print summary shows VAD state:

```
Config set.
  GPT model  : gpt-4o
  Whisper    : kotoba-tech/kotoba-whisper-v2.0
  VAD filter : True (silero-vad)   # or: False (disabled)
  Batch size : 5 segments per GPT call
```

## Cell 4: Transcription logic

### Step 1 — Load audio

Load audio file with `soundfile` into a float32 numpy array at 16kHz mono. This is the format both silero-vad and the transformers pipeline expect.

### Step 2 — VAD (when USE_VAD=True)

Run silero-vad on the full audio to produce a list of `(start_s, end_s)` speech spans. Merge any spans separated by less than `VAD_MERGE_GAP_MS = 200` ms into a single span to avoid over-segmentation. Parameters are defined as constants at the top of Cell 4:

```python
VAD_THRESHOLD       = 0.2
VAD_MIN_SPEECH_MS   = 100
VAD_MIN_SILENCE_MS  = 200
VAD_MERGE_GAP_MS    = 200
```

When `USE_VAD=False`, the entire audio is treated as one span `(0, duration)`.

### Step 3 — Transcription

Load the pipeline once (cached for the session):

```python
pipe = pipeline(
    "automatic-speech-recognition",
    model="kotoba-tech/kotoba-whisper-v2.0",
    torch_dtype=torch.float16,
    device="cuda",
)
```

For each speech span, extract the audio slice and run:

```python
result = pipe(
    audio_slice,
    chunk_length_s=30,
    batch_size=8,
    return_timestamps=True,
    generate_kwargs={"language": "japanese", "task": "transcribe"},
)
```

Offset each chunk's timestamps by the span's real-world start time. Append to `segments` list.

### Step 4 — Output

`segments` is a list of dicts with `start`, `end`, `text` — identical shape to the current notebook. Cells 4b, 5, and 6 require no changes.

## Trade-offs accepted

| Factor | CT2 (current) | Original model (new) |
|--------|--------------|----------------------|
| Accuracy | Lower | Higher |
| Speed on T4 | ~3-5 min | Slower (accepted) |
| VRAM | ~2GB | ~4-6GB float16 (fits T4) |
| VAD | Built-in | silero-vad (separate) |
| Dependencies | `faster-whisper` | `transformers`, `silero-vad` |

## Out of scope

- Translation cell (Cell 5) — no changes
- SRT generation (Cell 6) — no changes
- The OpenAI notebook (`whispersubs_colab_openai.ipynb`) — not touched
