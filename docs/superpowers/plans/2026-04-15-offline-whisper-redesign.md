# Offline Whisper Transcription Redesign — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace OpenAI Whisper API with local kotoba-whisper-v2.0 via faster-whisper for accurate Japanese transcription and timestamps, improve Indonesian translation tone, and clean up unused methods.

**Architecture:** Local GPU-based transcription (faster-whisper + CTranslate2 + Silero VAD) replaces OpenAI Whisper API calls. GPT-4o remains for Japanese→Indonesian translation with a redesigned prompt for casual tone. The `translate` method is removed; 3 methods remain.

**Tech Stack:** faster-whisper, torch (CUDA), openai (GPT translation), Python 3.x

---

### Task 1: Update Dependencies

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Update requirements.txt**

Replace the full contents of `requirements.txt` with:

```
openai>=1.0.0
configparser>=5.0.0
faster-whisper>=1.1.0
```

Note: `torch` with CUDA must be installed separately via:
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```
This is not in requirements.txt because the CUDA-specific index URL makes it a manual install step.

- [ ] **Step 2: Install dependencies**

Run:
```bash
pip install -r requirements.txt
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Expected: All packages install successfully. `faster-whisper` pulls in `ctranslate2` and `silero-vad` as transitive dependencies.

- [ ] **Step 3: Verify imports work**

Run:
```bash
python -c "from faster_whisper import WhisperModel; print('faster-whisper OK')"
python -c "import torch; print(f'torch OK, CUDA: {torch.cuda.is_available()}')"
```

Expected:
```
faster-whisper OK
torch OK, CUDA: True
```

- [ ] **Step 4: Commit**

```bash
git add requirements.txt
git commit -m "update dependencies: add faster-whisper for local transcription"
```

---

### Task 2: Add Local Transcription Function

**Files:**
- Modify: `whisper.py:1-8` (imports)
- Modify: `whisper.py` (add new function after `get_mime_type`, before `process_transcribe_only_method`)

- [ ] **Step 1: Update imports**

Replace the imports at the top of `whisper.py` (lines 1-8):

```python
import os
import json
import time
import re
from datetime import timedelta
import argparse
from openai import OpenAI
import configparser
```

With:

```python
import os
import json
import time
import re
from datetime import timedelta
import argparse
from openai import OpenAI
import configparser
from faster_whisper import WhisperModel
```

- [ ] **Step 2: Add local transcription function**

Add this function after `get_mime_type()` (after line 204) and before `process_transcribe_only_method()`:

```python
def transcribe_local(audio_path, model_name="kotoba-tech/kotoba-whisper-v2.0", device="cuda", compute_type="float16"):
    """Transcribe Japanese audio using local faster-whisper model.
    
    Returns list of segments with 'start', 'end', 'text' keys.
    """
    print(f"Loading model: {model_name} (device={device}, compute_type={compute_type})")
    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    
    print("Transcribing with VAD filter...")
    segments_iter, info = model.transcribe(
        audio_path,
        language="ja",
        vad_filter=True,
        vad_parameters=dict(
            min_silence_duration_ms=500,
            speech_pad_ms=200,
        ),
    )
    
    print(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")
    
    segments = []
    for seg in segments_iter:
        segments.append({
            'start': seg.start,
            'end': seg.end,
            'text': seg.text.strip()
        })
    
    print(f"Transcription complete! Total segments: {len(segments)}")
    return segments
```

- [ ] **Step 3: Test the function manually**

Run with one of the existing mp3 files:
```bash
python -c "
from whisper import transcribe_local
segs = transcribe_local('saku278.mp3')
for s in segs[:5]:
    print(f'{s[\"start\"]:.2f} -> {s[\"end\"]:.2f}: {s[\"text\"]}')
"
```

Expected: Japanese text segments with precise timestamps (not rounded to 2 seconds). First run will download the model (~1.5GB).

- [ ] **Step 4: Commit**

```bash
git add whisper.py
git commit -m "add transcribe_local function using faster-whisper + kotoba-whisper-v2.0"
```

---

### Task 3: Remove make_natural_indonesian and Update Translation Prompt

**Files:**
- Modify: `whisper.py:85-163` (remove `make_natural_indonesian`)
- Modify: `whisper.py` (update translation prompts in `process_translate_srt_method` and `process_transcribe_method`)

- [ ] **Step 1: Delete make_natural_indonesian function**

Remove the entire `make_natural_indonesian()` function (lines 85-163, from `def make_natural_indonesian(text):` through the final `return natural_text`).

- [ ] **Step 2: Define the new translation prompt as a module constant**

Add this constant after the `read_srt_file()` function (which will now end around line 66) and before `get_api_key_from_config()`:

```python
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
```

- [ ] **Step 3: Update process_translate_srt_method prompt**

In `process_translate_srt_method()`, replace the system content in the `chat.completions.create` call. Change:

```python
                    {
                        "role": "system",
                        "content": "Kamu adalah penerjemah subtitle profesional dari bahasa Jepang ke bahasa Indonesia. "
                                   "Terjemahkan setiap subtitle dalam tanda [Subtitle X] ke bahasa Indonesia yang natural, sopan, dan mudah dipahami. "
                                   "Pertahankan format [Subtitle X] agar bisa dicocokkan kembali ke segmen aslinya. "
                                   "Jaga konteks antar subtitle agar cerita tetap nyambung. "
                                   "Gunakan bahasa Indonesia sehari-hari yang tidak terlalu formal tapi tetap sopan."
                    },
```

To:

```python
                    {
                        "role": "system",
                        "content": TRANSLATION_SYSTEM_PROMPT.replace("[Dialog X]", "[Subtitle X]")
                    },
```

- [ ] **Step 4: Remove make_natural_indonesian calls in process_translate_srt_method**

In `process_translate_srt_method()`, find this block in the segment creation loop:

```python
                # Apply natural Indonesian
                natural_translated = make_natural_indonesian(translated)
                
                translated_segments.append({
                    'start': seg['start'],
                    'end': seg['end'],
                    'text': natural_translated
                })
```

Replace with:

```python
                translated_segments.append({
                    'start': seg['start'],
                    'end': seg['end'],
                    'text': translated
                })
```

- [ ] **Step 5: Update process_transcribe_method prompt**

In `process_transcribe_method()`, replace the system content in the `chat.completions.create` call. Change:

```python
                {
                    "role": "system",
                    "content": "Kamu adalah penerjemah subtitle dari bahasa Jepang ke bahasa Indonesia. "
                               "Terjemahkan setiap dialog dalam tanda [Dialog X] ke bahasa Indonesia yang natural, sopan, dan mudah dipahami. "
                               "Pertahankan format [Dialog X] agar bisa dicocokkan kembali ke segmen aslinya."
                },
```

To:

```python
                {
                    "role": "system",
                    "content": TRANSLATION_SYSTEM_PROMPT
                },
```

- [ ] **Step 6: Remove make_natural_indonesian call in process_transcribe_method**

In `process_transcribe_method()`, find:

```python
            natural_translated = make_natural_indonesian(translated)
            new_segment = {
                "start": seg.start,
                "end": seg.end,
                "text": natural_translated
            }
```

Replace with:

```python
            new_segment = {
                "start": seg['start'],
                "end": seg['end'],
                "text": translated
            }
```

Note: `seg.start` changes to `seg['start']` because segments will now come from `transcribe_local()` which returns dicts, not OpenAI API objects.

- [ ] **Step 7: Commit**

```bash
git add whisper.py
git commit -m "remove make_natural_indonesian, use unified casual translation prompt"
```

---

### Task 4: Rewire process_transcribe_only_method to Use Local Model

**Files:**
- Modify: `whisper.py` — `process_transcribe_only_method()`

- [ ] **Step 1: Replace the function**

Replace the entire `process_transcribe_only_method()` function. Change from the function that takes `(client, input_file, whisper_model)` and uses `client.audio.transcriptions.create(...)` to:

```python
def process_transcribe_only_method(input_file, whisper_model, device, compute_type):
    """Transcribe Japanese audio without translation using local model."""
    print("Menggunakan metode: Transcribe Only (Japanese)")
    print(f"Model: {whisper_model}")
    
    segments = transcribe_local(input_file, whisper_model, device, compute_type)
    
    if not segments:
        print("Tidak ada segmen ditemukan.")
        return []
    
    return segments
```

- [ ] **Step 2: Test with an audio file**

Run:
```bash
python whisper.py --input saku278.mp3 --output test_transcribe_only.srt --method transcribe-only --device cuda
```

Expected: Produces `test_transcribe_only.srt` with Japanese text and precise timestamps (varying durations, millisecond precision).

- [ ] **Step 3: Verify timestamp precision**

Check the output:
```bash
head -20 test_transcribe_only.srt
```

Expected: Timestamps like `00:00:01,240 --> 00:00:03,780` (not all rounded to whole seconds or 2-second blocks).

- [ ] **Step 4: Commit**

```bash
git add whisper.py
git commit -m "rewire transcribe-only method to use local faster-whisper model"
```

---

### Task 5: Rewire process_transcribe_method to Use Local Model

**Files:**
- Modify: `whisper.py` — `process_transcribe_method()`

- [ ] **Step 1: Replace the function**

Replace the entire `process_transcribe_method()` function. Change from the function that takes `(client, input_file, model, whisper_model, batch_size)` and uses `client.audio.transcriptions.create(...)` to:

```python
def process_transcribe_method(client, input_file, model, whisper_model, batch_size, device, compute_type):
    """Transcribe Japanese audio locally, then translate to Indonesian via GPT."""
    print("Menggunakan metode: Transcribe (Local) + Translate (GPT)")
    print(f"Model Whisper: {whisper_model}")
    print(f"Model translasi: {model}")
    
    # Step 1: Local transcription
    segments = transcribe_local(input_file, whisper_model, device, compute_type)
    
    if not segments:
        print("Tidak ada segmen ditemukan.")
        return []
    
    print(f"\nTotal segmen: {len(segments)}")
    print("Memulai translasi ke bahasa Indonesia...")
    
    # Step 2: Translate in batches via GPT
    translated_segments = []
    
    for batch_start in range(0, len(segments), batch_size):
        batch_end = min(batch_start + batch_size, len(segments))
        batch_segments = segments[batch_start:batch_end]
        
        print(f"Menerjemahkan batch {batch_start//batch_size + 1} (segmen {batch_start+1}-{batch_end})...")
        
        batch_texts = []
        for i, seg in enumerate(batch_segments):
            text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', seg['text'])
            batch_texts.append(f"[Dialog {i+1}] {text}")
        
        combined_text = "\n".join(batch_texts)
        
        try:
            chat_completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": TRANSLATION_SYSTEM_PROMPT},
                    {"role": "user", "content": combined_text}
                ],
                temperature=0.6
            )
            
            result_text = chat_completion.choices[0].message.content.strip()
            
            # Parse translation results
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
                    except:
                        continue
                elif current_dialog is not None and line.strip():
                    current_text.append(line.strip())
            
            if current_dialog is not None:
                translated_dialogs[current_dialog] = ' '.join(current_text).strip()
            
            for i, seg in enumerate(batch_segments):
                dialog_num = i + 1
                translated = translated_dialogs.get(dialog_num, '')
                if not translated:
                    print(f"  Warning: Dialog {batch_start + i + 1} gagal diterjemahkan, menggunakan text original")
                    translated = seg['text']
                
                translated_segments.append({
                    'start': seg['start'],
                    'end': seg['end'],
                    'text': translated
                })
            
            if batch_end < len(segments):
                time.sleep(1)
                
        except Exception as e:
            print(f"  Error saat menerjemahkan batch: {str(e)}")
            for seg in batch_segments:
                translated_segments.append(seg)
    
    return translated_segments
```

- [ ] **Step 2: Test with an audio file**

Run:
```bash
python whisper.py --input saku278.mp3 --output test_transcribe.srt --method transcribe --device cuda
```

Expected: Produces `test_transcribe.srt` with casual Indonesian subtitles and precise timestamps.

- [ ] **Step 3: Verify translation tone**

Check the output:
```bash
head -40 test_transcribe.srt
```

Expected: Uses "aku/kamu", casual language, no stiff formal Indonesian.

- [ ] **Step 4: Commit**

```bash
git add whisper.py
git commit -m "rewire transcribe method to use local model + casual translation prompt"
```

---

### Task 6: Remove translate Method and Update main()

**Files:**
- Modify: `whisper.py` — remove `process_translate_method()`, update `main()`

- [ ] **Step 1: Delete process_translate_method**

Remove the entire `process_translate_method()` function (the one that uses `client.audio.translations.create()`).

- [ ] **Step 2: Update argparse in main()**

In `main()`, replace the argument definitions. Change:

```python
    parser.add_argument("--whisper-model", default="whisper-1", 
                        help="Model Whisper untuk transkrip (default: whisper-1)")
    
    # Method selection - UPDATED with new options
    parser.add_argument("--method", default="transcribe", 
                        choices=["transcribe", "translate", "transcribe-only", "translate-srt"],
                        help="Metode processing:\n"
                             "'transcribe' - transcribe Japanese + translate to Indonesian\n"
                             "'translate' - direct translate + paraphrase\n"
                             "'transcribe-only' - transcribe Japanese only (no translation)\n"
                             "'translate-srt' - translate existing Japanese SRT to Indonesian")
    
    parser.add_argument("--batch-size", type=int, default=5,
                        help="Jumlah dialog/subtitle per batch untuk translasi (default: 5)")
```

To:

```python
    parser.add_argument("--whisper-model", default="kotoba-tech/kotoba-whisper-v2.0", 
                        help="Model Whisper lokal (default: kotoba-tech/kotoba-whisper-v2.0)")
    
    parser.add_argument("--method", default="transcribe", 
                        choices=["transcribe", "transcribe-only", "translate-srt"],
                        help="Metode processing:\n"
                             "'transcribe' - transcribe Japanese + translate to Indonesian\n"
                             "'transcribe-only' - transcribe Japanese only (no translation)\n"
                             "'translate-srt' - translate existing Japanese SRT to Indonesian")
    
    parser.add_argument("--batch-size", type=int, default=5,
                        help="Jumlah dialog/subtitle per batch untuk translasi (default: 5)")
    
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                        help="Device untuk model Whisper (default: cuda)")
    
    parser.add_argument("--compute-type", default="float16",
                        choices=["float16", "int8", "float32"],
                        help="Compute type untuk model Whisper (default: float16)")
```

- [ ] **Step 3: Update variable extraction after parse_args**

After `args = parser.parse_args()`, add the new variables. Find:

```python
    whisper_model = args.whisper_model
    batch_size = args.batch_size
```

Replace with:

```python
    whisper_model = args.whisper_model
    batch_size = args.batch_size
    device = args.device
    compute_type = args.compute_type
```

- [ ] **Step 4: Update API key logic — not required for transcribe-only**

Replace the API key handling block. The current logic (lines ~704-726) is convoluted. Replace everything from `# Get API key for methods that need it` through the `transcribe-only` API key block with:

```python
    # Get API key (only needed for methods that use GPT translation)
    needs_api = method in ("transcribe", "translate-srt")
    api_key = None
    
    if needs_api:
        api_key = args.api_key
        if not api_key:
            api_key = get_api_key_from_config(args.config)
        if not api_key:
            print(f"Error: API key diperlukan untuk metode '{method}'.")
            print(f"Silakan tentukan API key melalui argument --api_key atau di file {args.config}")
            print(f"\nFormat file config.ini:")
            print("[OPENAI]")
            print("api_key = sk-your_openai_api_key_here")
            print("model = gpt-4o  # opsional")
            return
```

- [ ] **Step 5: Update OpenAI client initialization**

Replace the current client initialization block:

```python
    # Initialize OpenAI client
    try:
        client = OpenAI(api_key=api_key)
    except Exception as e:
        print(f"Error inisialisasi OpenAI client: {str(e)}")
        return
```

With:

```python
    # Initialize OpenAI client only if needed
    client = None
    if needs_api:
        try:
            client = OpenAI(api_key=api_key)
        except Exception as e:
            print(f"Error inisialisasi OpenAI client: {str(e)}")
            return
```

- [ ] **Step 6: Update method routing in the try block**

Replace the method routing inside the `try:` block. Replace the entire section from `if method == "transcribe-only":` through `segments = process_translate_method(...)` with:

```python
        if method == "transcribe-only":
            print(f"Model Whisper: {whisper_model}")
            print(f"Device: {device} ({compute_type})")
            
            segments = process_transcribe_only_method(input_file, whisper_model, device, compute_type)
            
        elif method == "translate-srt":
            print(f"Metode: Translate SRT")
            print(f"Model translasi: {model}")
            segments = process_translate_srt_method(client, input_file, model, batch_size)
            
        elif method == "transcribe":
            print(f"Model Whisper: {whisper_model}")
            print(f"Model translasi: {model}")
            print(f"Device: {device} ({compute_type})")
            
            # Check file size
            file_size = os.path.getsize(input_file)
            print(f"Ukuran file: {file_size / 1024 / 1024:.1f} MB")
            
            segments = process_transcribe_method(client, input_file, model, whisper_model, batch_size, device, compute_type)
```

- [ ] **Step 7: Remove the file size check from transcribe-only route**

The `transcribe-only` route no longer needs the 25MB API limit check since we're running locally. The code in Step 6 already handles this — no 25MB check for local methods.

- [ ] **Step 8: Test all three methods**

Test transcribe-only:
```bash
python whisper.py --input saku278.mp3 --output test1.srt --method transcribe-only
```

Test transcribe (main workflow):
```bash
python whisper.py --input saku278.mp3 --output test2.srt --method transcribe
```

Test translate-srt (using the Japanese SRT from test1):
```bash
python whisper.py --input test1.srt --output test3.srt --method translate-srt
```

Expected: All three produce valid SRT files without errors.

- [ ] **Step 9: Commit**

```bash
git add whisper.py
git commit -m "remove translate method, update CLI args for local whisper model"
```

---

### Task 7: Update README.md

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Write the updated README**

Replace the entire contents of `README.md` with:

```markdown
# WhisperAI-subs

Japanese audio to Indonesian subtitle generator. Uses local [kotoba-whisper-v2.0](https://huggingface.co/kotoba-tech/kotoba-whisper-v2.0) for transcription and OpenAI GPT for translation.

## Requirements

- Python 3.8+
- NVIDIA GPU with CUDA support (recommended)
- OpenAI API key (for translation methods)

## Installation

1. Install PyTorch with CUDA:
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

2. Install other dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your OpenAI API key in `config.ini`:
```ini
[OPENAI]
api_key = sk-your_openai_api_key_here
model = gpt-4o
```

## Usage

### Transcribe + Translate (main workflow)

Transcribes Japanese audio locally, then translates to casual Indonesian via GPT:

```bash
python whisper.py --input audio.mp3 --output output.srt
```

### Transcribe Only

Produces Japanese-only subtitles (no translation, no API key needed):

```bash
python whisper.py --input audio.mp3 --output output.srt --method transcribe-only
```

### Translate Existing SRT

Translates an existing Japanese SRT file to Indonesian:

```bash
python whisper.py --input japanese.srt --output indonesian.srt --method translate-srt
```

## Options

| Argument | Default | Description |
|---|---|---|
| `--input` | `audio.wav` | Input audio file or SRT file |
| `--output` | `output.srt` | Output SRT file |
| `--method` | `transcribe` | `transcribe`, `transcribe-only`, or `translate-srt` |
| `--model` | `gpt-3.5-turbo` | OpenAI model for translation |
| `--whisper-model` | `kotoba-tech/kotoba-whisper-v2.0` | Local Whisper model name or path |
| `--device` | `cuda` | `cuda` or `cpu` |
| `--compute-type` | `float16` | `float16`, `int8`, or `float32` |
| `--batch-size` | `5` | Dialogs per translation batch |
| `--api_key` | from config.ini | OpenAI API key |

## Notes

- First run downloads the Whisper model (~1.5GB) — cached locally after that
- CPU mode (`--device cpu`) works but is significantly slower
- Use `split_audio.py` to split large audio files before processing
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "update README for local whisper transcription setup"
```

---

### Task 8: Clean Up Test Files

- [ ] **Step 1: Remove test output files**

```bash
rm -f test_transcribe_only.srt test_transcribe.srt test1.srt test2.srt test3.srt
```

- [ ] **Step 2: Final verification**

Run the main workflow one more time to confirm everything works end-to-end:

```bash
python whisper.py --input saku278.mp3 --output output.srt --method transcribe
```

Verify: output.srt has casual Indonesian subtitles with precise timestamps.
