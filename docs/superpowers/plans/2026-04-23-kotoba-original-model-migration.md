# Kotoba CT2 → Original Model Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `jctv-tech/kotoba-whisper-v21-ct2` (faster-whisper) with `kotoba-tech/kotoba-whisper-v2.0` (HuggingFace Transformers) in `whispersubs_colab_kotoba.ipynb` to improve transcription accuracy.

**Architecture:** Load audio with soundfile → optionally filter speech spans with silero-vad → run each span through the transformers ASR pipeline → collect segments with real-world timestamps. The `segments` list shape is unchanged so Cells 3, 4b, 5, and 6 require no edits.

**Tech Stack:** `transformers`, `torch`, `torchaudio`, `silero-vad`, `soundfile`, `openai`

---

## Files

- Modify: `whispersubs_colab_kotoba.ipynb` — cells 0, 1, 2, 4 only

Cell IDs (needed for NotebookEdit):
- Cell 0 header markdown: `2qemDwPeMSv2`
- Cell 1 deps: `zxKK5-kBMSv4`
- Cell 2 config: `yI9PNlmMMSv4`
- Cell 4 transcription: `f1e2488f`

---

## Task 1: Update Cell 0 header

**Files:**
- Modify: `whispersubs_colab_kotoba.ipynb` cell `2qemDwPeMSv2`

- [ ] **Step 1: Edit the markdown header cell**

Use NotebookEdit to replace the source of cell `2qemDwPeMSv2` with:

```markdown
# WhisperSubs Colab
Japanese audio → casual Indonesian SRT using kotoba-whisper-v2.0 (float16) + GPT-4o

> **Before running:** Runtime → Change runtime type → T4 GPU
>
> **Workflow:** Run all cells top to bottom. Enter API key when prompted. Upload audio. Download SRT.
```

- [ ] **Step 2: Commit**

```bash
git add whispersubs_colab_kotoba.ipynb
git commit -m "chore: update kotoba notebook header to reflect model migration"
```

---

## Task 2: Update Cell 1 dependencies

**Files:**
- Modify: `whispersubs_colab_kotoba.ipynb` cell `zxKK5-kBMSv4`

- [ ] **Step 1: Replace the deps cell source**

Use NotebookEdit to replace the source of cell `zxKK5-kBMSv4` with:

```python
# Cell 1: Install dependencies
!pip install -q transformers torch torchaudio silero-vad soundfile openai
print('Dependencies installed.')
```

`faster-whisper` is removed. `transformers`, `torchaudio`, `silero-vad`, and `soundfile` are added.

- [ ] **Step 2: Commit**

```bash
git add whispersubs_colab_kotoba.ipynb
git commit -m "feat: swap faster-whisper for transformers + silero-vad deps"
```

---

## Task 3: Update Cell 2 configuration

**Files:**
- Modify: `whispersubs_colab_kotoba.ipynb` cell `yI9PNlmMMSv4`

- [ ] **Step 1: Replace the config cell source**

Use NotebookEdit to replace the source of cell `yI9PNlmMMSv4` with:

```python
# Cell 2: Configuration — run this once per session
import getpass
import os

OPENAI_API_KEY = getpass.getpass("OpenAI API Key: ")
GPT_MODEL = "gpt-4o"
BATCH_SIZE = 5
USE_VAD = True   # Set False to skip VAD and feed full audio directly

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
vad_label = f"True (silero-vad)" if USE_VAD else "False (disabled)"
print(f"Config set.")
print(f"  GPT model  : {GPT_MODEL}")
print(f"  Whisper    : kotoba-tech/kotoba-whisper-v2.0")
print(f"  VAD filter : {vad_label}")
print(f"  Batch size : {BATCH_SIZE} segments per GPT call")
```

Changes from original:
- `WHISPER_MODEL` variable removed (model is hardcoded in Cell 4)
- `USE_VAD = True` added
- Print summary updated to show VAD state and fixed model name

- [ ] **Step 2: Commit**

```bash
git add whispersubs_colab_kotoba.ipynb
git commit -m "feat: add USE_VAD toggle to kotoba notebook config"
```

---

## Task 4: Rewrite Cell 4 transcription engine

**Files:**
- Modify: `whispersubs_colab_kotoba.ipynb` cell `f1e2488f`

- [ ] **Step 1: Replace Cell 4 source with the new transcription engine**

Use NotebookEdit to replace the source of cell `f1e2488f` with:

```python
# Cell 4: Transcribe audio
# First run: downloads model to Colab (~3GB, 3-5 min). Cached for this session.
import torch
import soundfile as sf
import numpy as np
from transformers import pipeline

# VAD parameters — tune here without touching Cell 2
VAD_THRESHOLD      = 0.2
VAD_MIN_SPEECH_MS  = 100
VAD_MIN_SILENCE_MS = 200
VAD_MERGE_GAP_MS   = 200
SAMPLE_RATE        = 16000
MODEL_ID           = "kotoba-tech/kotoba-whisper-v2.0"

print(f"Loading model: {MODEL_ID} on T4 GPU (float16)...")
print("(First run downloads ~3GB — takes 3-5 min. Cached for this session.)")

pipe = pipeline(
    "automatic-speech-recognition",
    model=MODEL_ID,
    torch_dtype=torch.float16,
    device="cuda",
)

# Load audio as float32 mono at 16kHz
print("Loading audio...")
audio_data, sr = sf.read(audio_filename, dtype="float32")
if audio_data.ndim > 1:
    audio_data = audio_data.mean(axis=1)
if sr != SAMPLE_RATE:
    import torchaudio
    audio_tensor = torch.tensor(audio_data).unsqueeze(0)
    audio_data = torchaudio.functional.resample(audio_tensor, sr, SAMPLE_RATE).squeeze(0).numpy()

duration = len(audio_data) / SAMPLE_RATE
print(f"Audio loaded: {duration:.1f}s")

# Determine speech spans
if USE_VAD:
    print("Running silero-vad...")
    vad_model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        verbose=False,
    )
    get_speech_timestamps = utils[0]

    audio_tensor = torch.tensor(audio_data)
    raw_spans = get_speech_timestamps(
        audio_tensor,
        vad_model,
        threshold=VAD_THRESHOLD,
        min_speech_duration_ms=VAD_MIN_SPEECH_MS,
        min_silence_duration_ms=VAD_MIN_SILENCE_MS,
        sampling_rate=SAMPLE_RATE,
        return_seconds=True,
    )

    # Merge spans closer than VAD_MERGE_GAP_MS
    spans = []
    for span in raw_spans:
        if spans and (span["start"] - spans[-1]["end"]) < VAD_MERGE_GAP_MS / 1000:
            spans[-1]["end"] = span["end"]
        else:
            spans.append({"start": span["start"], "end": span["end"]})

    print(f"VAD found {len(spans)} speech spans (merged from {len(raw_spans)} raw).")
else:
    spans = [{"start": 0.0, "end": duration}]
    print("VAD disabled — transcribing full audio as one span.")

# Transcribe each span
print("Transcribing...")
segments = []
for i, span in enumerate(spans):
    start_sample = int(span["start"] * SAMPLE_RATE)
    end_sample   = int(span["end"]   * SAMPLE_RATE)
    audio_slice  = audio_data[start_sample:end_sample]

    result = pipe(
        {"array": audio_slice, "sampling_rate": SAMPLE_RATE},
        chunk_length_s=30,
        batch_size=8,
        return_timestamps=True,
        generate_kwargs={"language": "japanese", "task": "transcribe"},
    )

    for chunk in result["chunks"]:
        ts = chunk["timestamp"]
        if ts[0] is None or ts[1] is None:
            continue
        text = chunk["text"].strip()
        if not text:
            continue
        segments.append({
            "start": span["start"] + ts[0],
            "end":   span["start"] + ts[1],
            "text":  text,
        })

    if (i + 1) % 10 == 0 or (i + 1) == len(spans):
        print(f"  Processed {i + 1}/{len(spans)} spans...")

print(f"\nTranscription complete! Total segments: {len(segments)}")
if segments:
    print(f"  First: [{segments[0]['start']:.2f}s → {segments[0]['end']:.2f}s] {segments[0]['text'][:80]}")
    print(f"  Last:  [{segments[-1]['start']:.2f}s → {segments[-1]['end']:.2f}s] {segments[-1]['text'][:80]}")
```

- [ ] **Step 2: Commit**

```bash
git add whispersubs_colab_kotoba.ipynb
git commit -m "feat: rewrite Cell 4 to use kotoba-whisper-v2.0 via transformers + silero-vad"
```

---

## Task 5: Clear stale cell outputs

Cell outputs from previous CT2 runs reference the old model name and output counts. Clear them so the notebook looks clean when shared.

**Files:**
- Modify: `whispersubs_colab_kotoba.ipynb` — clear outputs on cells `zxKK5-kBMSv4`, `yI9PNlmMMSv4`, `EOtkfXZOMSv5`, `f1e2488f`, `OzjWCP11pvKP`, `64f51549`, `VgTpIPQ2fIwO`

- [ ] **Step 1: Clear outputs for each changed/stale cell**

Use NotebookEdit with `clear_outputs=True` on each of these cell IDs:
- `zxKK5-kBMSv4` (Cell 1 — old pip install output)
- `yI9PNlmMMSv4` (Cell 2 — old config output showing CT2 model name)
- `EOtkfXZOMSv5` (Cell 3 — old upload output)
- `f1e2488f` (Cell 4 — old transcription output)
- `OzjWCP11pvKP` (Cell 4b — old Japanese SRT output)
- `64f51549` (Cell 5 — old translation batch output)
- `VgTpIPQ2fIwO` (Cell 6 — old SRT download output)

- [ ] **Step 2: Commit**

```bash
git add whispersubs_colab_kotoba.ipynb
git commit -m "chore: clear stale CT2 cell outputs from kotoba notebook"
```

---

## Task 6: Manual verification in Colab

This is a notebook — correctness is verified by running it end-to-end in Colab, not by unit tests.

- [ ] **Step 1: Open notebook in Colab**

Upload `whispersubs_colab_kotoba.ipynb` to Google Colab. Set runtime to T4 GPU.

- [ ] **Step 2: Run Cell 1 — verify deps install cleanly**

Expected output ends with:
```
Dependencies installed.
```
No errors about `faster-whisper` or conflicting packages.

- [ ] **Step 3: Run Cell 2 — verify config output**

Enter any valid OpenAI API key. Expected output:
```
Config set.
  GPT model  : gpt-4o
  Whisper    : kotoba-tech/kotoba-whisper-v2.0
  VAD filter : True (silero-vad)
  Batch size : 5 segments per GPT call
```

- [ ] **Step 4: Run Cell 3 — upload audio**

Upload a short test MP3/WAV (even 30 seconds of Japanese speech is enough). Verify upload succeeds and `audio_filename` is set.

- [ ] **Step 5: Run Cell 4 with USE_VAD=True — verify transcription**

Expected output pattern:
```
Loading model: kotoba-tech/kotoba-whisper-v2.0 on T4 GPU (float16)...
(First run downloads ~3GB — takes 3-5 min. Cached for this session.)
Audio loaded: XXX.Xs
Running silero-vad...
VAD found N speech spans (merged from M raw).
Transcribing...
  Processed 10/N spans...
  ...
Transcription complete! Total segments: NNN
  First: [X.XXs → X.XXs] <Japanese text>
  Last:  [X.XXs → X.XXs] <Japanese text>
```

Check that Japanese text is present and readable. Compare to the CT2 output quality.

- [ ] **Step 6: Run Cell 4b — verify Japanese SRT exports**

Expected: file downloads, SRT contains Japanese text with correct timestamps.

- [ ] **Step 7: Re-run with USE_VAD=False — verify toggle works**

In Cell 2, change `USE_VAD = True` to `USE_VAD = False`. Re-run Cells 2 and 4.

Expected Cell 2 output:
```
  VAD filter : False (disabled)
```

Expected Cell 4 output:
```
VAD disabled — transcribing full audio as one span.
```

Verify `segments` is still populated and Cell 4b still produces an SRT.

- [ ] **Step 8: Final commit after successful verification**

```bash
git add whispersubs_colab_kotoba.ipynb
git commit -m "feat: complete kotoba CT2 → kotoba-whisper-v2.0 migration"
```
