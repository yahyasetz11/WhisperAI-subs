# Colab Whisper Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a single Google Colab notebook (`whispersubs_colab.ipynb`) that transcribes Japanese audio at full float16 precision, fixes hallucination and timing, and translates to casual Indonesian SRT via GPT-4o.

**Architecture:** Six-cell notebook built incrementally — install → config → upload → transcribe (float16 + fixed VAD) → translate (GPT-4o batch) → SRT download. Core logic ported directly from `whisper.py` with minimal changes; the only material differences are `compute_type="float16"`, re-enabled VAD with tuned parameters, and Colab-specific file I/O widgets.

**Tech Stack:** `faster-whisper` (CTranslate2), `openai` Python SDK, `google.colab` widgets, Python 3.10+, Colab T4 GPU (16GB VRAM)

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `whispersubs_colab.ipynb` | Create | Complete Colab notebook — all 6 cells |

No other files are created or modified.

> **Notebook editing workflow:** Task 1 creates the initial `.ipynb` JSON. For Tasks 2–5, add each new cell using **Colab's UI** (click `+ Code` below the last cell, paste the code block). After each task's verify step, save the notebook: **File → Download → Download .ipynb**, overwrite the local `whispersubs_colab.ipynb`, then commit. Do not hand-edit the raw JSON after Task 1.

---

### Task 1: Create notebook skeleton with install cell

**Files:**
- Create: `whispersubs_colab.ipynb`

- [ ] **Step 1: Create the notebook file**

Create `whispersubs_colab.ipynb` with the following content (valid Jupyter notebook JSON):

```json
{
 "nbformat": 4,
 "nbformat_minor": 5,
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.10.0"
  },
  "accelerator": "GPU",
  "colab": {
   "provenance": [],
   "gpuType": "T4"
  }
 },
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# WhisperSubs Colab\n",
    "Japanese audio → casual Indonesian SRT using kotoba-whisper-v2.0 (float16) + GPT-4o\n",
    "\n",
    "> **Before running:** Runtime → Change runtime type → T4 GPU\n",
    ">\n",
    "> **Workflow:** Run all cells top to bottom. Enter API key when prompted. Upload audio. Download SRT."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Cell 1: Install dependencies\n",
    "!pip install -q faster-whisper openai\n",
    "print('Dependencies installed.')"
   ]
  }
 ]
}
```

- [ ] **Step 2: Verify install cell on Colab**

Upload `whispersubs_colab.ipynb` to Google Colab (colab.research.google.com → File → Upload notebook).
Set runtime to T4 GPU: Runtime → Change runtime type → T4 GPU.
Run Cell 1.

Expected output:
```
Dependencies installed.
```

- [ ] **Step 3: Commit**

```bash
git add whispersubs_colab.ipynb
git commit -m "feat: add Colab notebook skeleton with dependency install cell"
```

---

### Task 2: Add configuration and audio upload cells

**Files:**
- Modify: `whispersubs_colab.ipynb`

- [ ] **Step 1: Add Cell 2 (configuration) to the notebook**

In `whispersubs_colab.ipynb`, append this cell to the `"cells"` array (after Cell 1):

```python
# Cell 2: Configuration — run this once per session
import getpass
import os

OPENAI_API_KEY = getpass.getpass("OpenAI API Key: ")
GPT_MODEL = "gpt-4o"
WHISPER_MODEL = "jctv-tech/kotoba-whisper-v21-ct2"
BATCH_SIZE = 5

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
print(f"Config set.")
print(f"  GPT model   : {GPT_MODEL}")
print(f"  Whisper model: {WHISPER_MODEL}")
print(f"  Batch size  : {BATCH_SIZE} segments per GPT call")
```

- [ ] **Step 2: Add Cell 3 (audio upload) to the notebook**

Append this cell to the `"cells"` array (after Cell 2):

```python
# Cell 3: Upload audio file
from google.colab import files

print("Select your audio file (MP3, WAV, M4A, FLAC, etc.):")
uploaded = files.upload()

if not uploaded:
    raise ValueError("No file uploaded. Re-run this cell and select a file.")

audio_filename = list(uploaded.keys())[0]
file_size_mb = len(uploaded[audio_filename]) / 1024 / 1024
print(f"\nReady: {audio_filename} ({file_size_mb:.1f} MB)")
```

- [ ] **Step 3: Verify on Colab**

Run Cell 2 — enter your real API key when prompted. Expected:
```
Config set.
  GPT model   : gpt-4o
  Whisper model: jctv-tech/kotoba-whisper-v21-ct2
  Batch size  : 5 segments per GPT call
```

Run Cell 3 — file picker appears. Upload any audio file. Expected:
```
Ready: my_audio.mp3 (25.3 MB)
```

- [ ] **Step 4: Commit**

```bash
git add whispersubs_colab.ipynb
git commit -m "feat: add configuration and audio upload cells to Colab notebook"
```

---

### Task 3: Add transcription cell (float16, fixed VAD + hallucination guards)

**Files:**
- Modify: `whispersubs_colab.ipynb`

This is the core fix — same model as local, but `compute_type="float16"` and VAD re-enabled with tuned parameters. Ported from `transcribe_local()` in `whisper.py`.

- [ ] **Step 1: Add Cell 4 (transcription) to the notebook**

Append this cell to the `"cells"` array (after Cell 3):

```python
# Cell 4: Transcribe audio
# First run: downloads model to Colab (~3-5 min). Subsequent cells in same session use cached model.
from faster_whisper import WhisperModel

print(f"Loading model: {WHISPER_MODEL} on T4 GPU (float16)...")
print("(First run downloads ~3GB — takes 3-5 min. Cached for this session.)")

model = WhisperModel(WHISPER_MODEL, device="cuda", compute_type="float16")

print("Transcribing...")
segments_iter, info = model.transcribe(
    audio_filename,
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
        "speech_pad_ms": 200,
    },
)

print(f"Detected language: {info.language} (confidence: {info.language_probability:.2f})")

segments = []
for seg in segments_iter:
    segments.append({
        "start": seg.start,
        "end": seg.end,
        "text": seg.text.strip(),
    })

print(f"\nTranscription complete! Total segments: {len(segments)}")
if segments:
    print(f"  First: [{segments[0]['start']:.2f}s → {segments[0]['end']:.2f}s] {segments[0]['text'][:80]}")
    print(f"  Last:  [{segments[-1]['start']:.2f}s → {segments[-1]['end']:.2f}s] {segments[-1]['text'][:80]}")
```

- [ ] **Step 2: Verify on Colab**

Run Cell 4 with a Japanese audio file. Expected output:
```
Loading model: jctv-tech/kotoba-whisper-v21-ct2 on T4 GPU (float16)...
(First run downloads ~3GB — takes 3-5 min. Cached for this session.)
Transcribing...
Detected language: ja (confidence: 0.99)

Transcription complete! Total segments: 47
  First: [1.02s → 3.84s] こんにちは、今日は...
  Last:  [540.10s → 543.60s] ありがとうございました
```

Verify:
- `segments` list is populated (not empty, not just 6 entries)
- First segment start time is > 0 (not 0.00s, meaning VAD detected speech onset correctly)
- No segment spans more than ~10 seconds (sign of over-merging)

- [ ] **Step 3: Commit**

```bash
git add whispersubs_colab.ipynb
git commit -m "feat: add transcription cell with float16 and fixed VAD + hallucination guards"
```

---

### Task 4: Add translation cell (GPT-4o batch, casual Indonesian)

**Files:**
- Modify: `whispersubs_colab.ipynb`

Ported directly from `process_transcribe_method()` in `whisper.py` — same `TRANSLATION_SYSTEM_PROMPT`, same batch parsing logic.

- [ ] **Step 1: Add Cell 5 (translation) to the notebook**

Append this cell to the `"cells"` array (after Cell 4):

```python
# Cell 5: Translate segments to casual Indonesian via GPT-4o
import re
import time
from openai import OpenAI

TRANSLATION_SYSTEM_PROMPT = (
    "Kamu adalah penerjemah subtitle dari bahasa Jepang ke bahasa Indonesia. "
    "Terjemahkan setiap dialog dalam tanda [Dialog X] ke bahasa Indonesia.\n\n"
    "Aturan gaya bahasa:\n"
    "- Gunakan \"aku/kamu\" bukan \"saya/Anda\"\n"
    "- Pakai bahasa sehari-hari yang santai seperti ngobrol sama teman\n"
    "- Hindari bahasa baku/formal (jangan pakai \"telah\", \"namun\", \"dapat\", dll)\n"
    "- Hindari slang berat Jakarta (jangan pakai \"gue/lu\", \"anjir\", dll)\n"
    "- Singkat dan natural, seperti subtitle anime fansub\n"
    "- Jaga konteks antar dialog agar cerita tetap nyambung\n\n"
    "Pertahankan format [Dialog X] agar bisa dicocokkan kembali."
)

client = OpenAI(api_key=OPENAI_API_KEY)
translated_segments = []
total_batches = (len(segments) + BATCH_SIZE - 1) // BATCH_SIZE

for batch_start in range(0, len(segments), BATCH_SIZE):
    batch_end = min(batch_start + BATCH_SIZE, len(segments))
    batch_segments = segments[batch_start:batch_end]
    batch_num = batch_start // BATCH_SIZE + 1

    print(f"Batch {batch_num}/{total_batches} — segments {batch_start + 1}-{batch_end}...")

    batch_texts = []
    for i, seg in enumerate(batch_segments):
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', seg['text'])
        batch_texts.append(f"[Dialog {i + 1}] {text}")

    combined_text = "\n".join(batch_texts)

    try:
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": TRANSLATION_SYSTEM_PROMPT},
                {"role": "user", "content": combined_text},
            ],
            temperature=0.6,
        )

        result_text = response.choices[0].message.content.strip()

        # Parse [Dialog N] markers out of GPT response
        translated_dialogs = {}
        current_dialog = None
        current_text = []

        for line in result_text.split('\n'):
            if line.strip().startswith('[Dialog'):
                if current_dialog is not None:
                    translated_dialogs[current_dialog] = ' '.join(current_text).strip()
                try:
                    dialog_num = int(line.split(']')[0].split()[-1])
                    current_dialog = dialog_num
                    text_after = line.split(']', 1)[1].strip() if ']' in line else ''
                    current_text = [text_after] if text_after else []
                except Exception:
                    continue
            elif current_dialog is not None and line.strip():
                current_text.append(line.strip())

        if current_dialog is not None:
            translated_dialogs[current_dialog] = ' '.join(current_text).strip()

        for i, seg in enumerate(batch_segments):
            translated = translated_dialogs.get(i + 1, '')
            if not translated:
                print(f"  Warning: Dialog {batch_start + i + 1} failed — using original JP text")
                translated = seg['text']
            translated_segments.append({
                'start': seg['start'],
                'end': seg['end'],
                'text': translated,
            })

        if batch_end < len(segments):
            time.sleep(1)

    except Exception as e:
        print(f"  Error in batch {batch_num}: {e}")
        for seg in batch_segments:
            translated_segments.append(seg)

print(f"\nTranslation complete! {len(translated_segments)} segments.")
if translated_segments:
    print(f"Sample translation: {translated_segments[0]['text']}")
```

- [ ] **Step 2: Verify on Colab**

Run Cell 5. Expected output:
```
Batch 1/10 — segments 1-5...
Batch 2/10 — segments 6-10...
...
Translation complete! 47 segments.
Sample translation: Hei, hari ini kita mau pergi ke...
```

Check:
- `translated_segments` has same count as `segments`
- Sample uses `aku/kamu` (not `saya/Anda`)
- No `[Dialog X]` markers leak into the output text
- No "Warning: Dialog N failed" lines (if any appear, the batch parse logic hit an edge case — investigate that batch's GPT response)

- [ ] **Step 3: Commit**

```bash
git add whispersubs_colab.ipynb
git commit -m "feat: add GPT-4o batch translation cell with casual Indonesian prompt"
```

---

### Task 5: Add SRT generation and download cell

**Files:**
- Modify: `whispersubs_colab.ipynb`

Ports `format_time()` and `create_srt()` from `whisper.py`. Adds Colab auto-download.

- [ ] **Step 1: Add Cell 6 (SRT + download) to the notebook**

Append this cell to the `"cells"` array (after Cell 5):

```python
# Cell 6: Generate SRT file and download
from datetime import timedelta
from google.colab import files

def format_time(seconds):
    td = timedelta(seconds=float(seconds))
    hours, remainder = divmod(td.seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    milliseconds = int(td.microseconds / 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"

srt_content = ""
for i, segment in enumerate(translated_segments):
    start_time = format_time(segment['start'])
    end_time = format_time(segment['end'])
    srt_content += f"{i + 1}\n{start_time} --> {end_time}\n{segment['text'].strip()}\n\n"

# Output filename derived from uploaded audio filename
base_name = audio_filename.rsplit('.', 1)[0]
output_filename = f"{base_name}_id.srt"

with open(output_filename, 'w', encoding='utf-8') as f:
    f.write(srt_content)

print(f"SRT saved: {output_filename} ({len(translated_segments)} segments)")
print("\nSample (first 3 entries):")
print(srt_content[:400])

files.download(output_filename)
print("Download triggered — check your browser downloads.")
```

- [ ] **Step 2: Verify on Colab**

Run Cell 6. Expected output:
```
SRT saved: my_audio_id.srt (47 segments)

Sample (first 3 entries):
1
00:00:01,020 --> 00:00:03,840
Hei, hari ini kita mau pergi ke...

2
00:00:04,100 --> 00:00:06,500
Aku udah nyiapin semua barang-barangnya.

3
...
Download triggered — check your browser downloads.
```

Verify:
- Timestamps are NOT all 2-second intervals (e.g., `00:00:00,000 --> 00:00:02,000` pattern should NOT repeat)
- Millisecond precision is present (e.g. `,020`, `,840` — not `,000`)
- SRT file lands in your browser downloads folder
- Open the file in a text editor: no `[Dialog X]` markers, no stray JP characters mixed in

- [ ] **Step 3: Commit**

```bash
git add whispersubs_colab.ipynb
git commit -m "feat: add SRT generation and auto-download cell to Colab notebook"
```

---

### Task 6: End-to-end test on Colab with real vlog audio

**Files:**
- No new files

- [ ] **Step 1: Pick a test clip**

Use a 3-5 minute clip from a real Japanese vlog. The clip should include at least one of: mask/muffled speech, quiet background voice, and a silence of 3+ seconds (e.g., a scene transition).

- [ ] **Step 2: Fresh session — Runtime → Run all**

Open the notebook in Colab in a fresh runtime. Runtime → Run all.
Enter API key when prompted in Cell 2. Upload the test clip in Cell 3.
Watch all cells complete.

- [ ] **Step 3: Check hallucination fix**

After Cell 4, inspect `segments` manually:

```python
# Run in a scratch cell
for i, s in enumerate(segments):
    print(f"{i+1:3}. [{s['start']:7.2f}s → {s['end']:7.2f}s] {s['text'][:60]}")
```

Expected: No subtitle entries during known silent sections. Segments align with actual speech.

- [ ] **Step 4: Check translation register**

Scan `translated_segments` for formality violations:

```python
# Run in a scratch cell
formal_words = ["saya", "Anda", "telah", "namun", "dapat", "merupakan", "tersebut"]
for i, s in enumerate(translated_segments):
    hits = [w for w in formal_words if w in s['text']]
    if hits:
        print(f"Segment {i+1}: {hits} → {s['text']}")
```

Expected: Zero or very few hits (some words like "dapat" may appear as part of informal compound expressions — review manually if flagged).

- [ ] **Step 5: Sync-check the SRT**

Load the downloaded `.srt` in VLC (Subtitle → Add Subtitle File) or any SRT preview tool alongside the audio. Scrub through and verify subtitles appear when the person speaks.

- [ ] **Step 6: Adjust parameters if needed and commit**

If results need tuning, modify only the transcription parameters in Cell 4 (the `vad_parameters` dict and threshold values are safe to adjust). See parameter guidance in the spec: `docs/superpowers/specs/2026-04-17-colab-whisper-redesign.md`.

If any tweaks are made:
```bash
git add whispersubs_colab.ipynb
git commit -m "fix: tune VAD/hallucination parameters based on test results"
```
