# Colab Whisper Redesign — JP Audio to Casual Indonesian SRT

**Date:** 2026-04-17  
**Status:** Approved

---

## Problem Statement

The current `whisper.py` runs kotoba-whisper-v2.0 locally on a GTX 1050 (3GB VRAM), which forces int8 quantization. This causes three issues:

1. **Low-voice / muffled audio accuracy degrades** — int8 loses subtle audio detail that full-precision would catch
2. **Hallucinated dialog** — phantom text appears during silence or low-energy regions because VAD was disabled to fix prior under-segmentation
3. **Early subtitle timing** — subtitles appear before the speaker talks, also a consequence of VAD being off

---

## Solution

Move the pipeline to a **Google Colab notebook** running on a free T4 GPU (16GB VRAM). This allows:

- Full **float16** precision for kotoba-whisper-v2.0 — no quantization loss
- VAD re-enabled with well-tuned parameters — fixes timing and hallucination
- Same GPT-4o casual Indonesian translation pipeline, unchanged
- Simple upload-run-download workflow suitable for occasional use (once every few days)

---

## Architecture

A single notebook file: `whispersubs_colab.ipynb`

### Notebook Cell Structure

| Cell | Purpose |
|------|---------|
| 1 | Install dependencies (`faster-whisper`, `openai`) |
| 2 | Configuration form (API key, GPT model, source language) |
| 3 | Audio file upload widget |
| 4 | Transcription (kotoba-whisper-v2.0, float16, fixed VAD) |
| 5 | Translation (GPT-4o, casual Indonesian prompt) |
| 6 | SRT download (auto-triggers browser download) |

No CLI args, no config files, no Google Drive required.

---

## Model Loading

```python
from faster_whisper import WhisperModel

model = WhisperModel(
    "kotoba-tech/kotoba-whisper-v2.0",
    device="cuda",
    compute_type="float16"
)
```

- First run per session: downloads + CT2-converts the model (~3GB, ~3-5 min)
- Subsequent cells in the same session: already cached in memory
- T4 GPU has 16GB VRAM — float16 fits comfortably

---

## Transcription Parameters (VAD + Hallucination Fix)

```python
segments, info = model.transcribe(
    audio_path,
    language="ja",
    beam_size=5,
    condition_on_previous_text=False,
    no_speech_threshold=0.6,
    compression_ratio_threshold=2.4,
    log_prob_threshold=-1.0,
    vad_filter=True,
    vad_parameters={
        "threshold": 0.5,
        "min_silence_duration_ms": 1000,
        "speech_pad_ms": 200
    }
)
```

**Why each parameter:**
- `vad_filter=True` — re-enabled to suppress silent regions (root fix for timing + phantom dialog)
- `threshold=0.5` — lenient enough to not merge natural sentence pauses, strict enough to catch real silence
- `min_silence_duration_ms=1000` — requires 1s of silence before splitting (prevents over-segmentation)
- `speech_pad_ms=200` — small padding around detected speech edges for natural-feeling cuts
- `no_speech_threshold=0.6` — drops segments Whisper isn't confident contain speech (kills phantom dialog)
- `compression_ratio_threshold=2.4` — drops repetitive/looped hallucinated text
- `condition_on_previous_text=False` — prevents context bleeding between segments
- `beam_size=5` — better accuracy over default beam_size=1

All values are starting points. Adjust if results are imperfect on specific audio.

---

## Translation Pipeline

Unchanged from current `whisper.py`. GPT-4o with the existing casual Indonesian system prompt:
- `aku/kamu` register (not `saya/Anda`)
- Fansub-style brevity
- No formal particles
- Batch translation: segments grouped into batches of 5, one GPT-4o API call per batch (balances cost and context quality)

---

## Security

- OpenAI API key entered via Colab form field (`getpass` or text input) each session
- Key is never written to the notebook file — safe to share `.ipynb` without leaking credentials

---

## User Workflow

1. Open notebook in Colab
2. **Runtime → Run all**
3. Enter OpenAI API key when prompted
4. Upload audio file via file picker
5. Wait (~3-5 min for a 10-20 min vlog on T4)
6. SRT file auto-downloads to browser downloads folder

---

## What Is Out of Scope

- Speaker diarization
- Google Drive integration (not needed for this workflow)
- Local fallback / offline mode
- Real-time transcription
- Any changes to the SRT format or translation style

---

## Dependencies

- `faster-whisper` — transcription via CTranslate2
- `openai` — GPT-4o translation
- `torch` (pre-installed on Colab T4 runtime)
- Source model: `kotoba-tech/kotoba-whisper-v2.0` (HuggingFace, auto-downloaded)
