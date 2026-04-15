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
