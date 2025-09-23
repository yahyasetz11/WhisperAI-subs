# WhisperAI-subs

Transkrip dan terjemahkan audio Jepang ke subtitle bahasa Indonesia natural menggunakan OpenAI Whisper API.

## Update Terbaru

- Menggunakan OpenAI client library (v1.0+) untuk stabilitas lebih baik
- Support pemilihan model GPT untuk translasi
- Bahasa Indonesia yang lebih natural (tidak terlalu formal/gaul)
- **NEW**: Metode Direct Translate dengan batch processing untuk konteks yang lebih baik
- **NEW**: Mode transcribe-only untuk menghasilkan subtitle Jepang tanpa translasi
- **NEW**: Fitur translate SRT untuk menerjemahkan file subtitle Jepang yang sudah ada

## Fitur

- Transkrip audio Jepang menggunakan Whisper API
- Terjemahan otomatis ke bahasa Indonesia yang natural (tidak terlalu formal/gaul)
- **Transcribe-only mode**: Hasilkan subtitle Jepang tanpa terjemahan
- **SRT Translation**: Terjemahkan file SRT Jepang yang sudah ada ke Indonesia
- Output dalam format SRT yang siap digunakan
- Support berbagai format audio (WAV, MP3, MP4, M4A, OGG, FLAC, WEBM)
- Rekomendasi menggunakan WAV untuk kualitas transkrip terbaik
- Empat metode processing:
  - **Transcribe**: Transcribe Jepang → Translate per segmen
  - **Translate**: Direct translate → Batch paraphrase untuk konteks lebih baik
  - **Transcribe-only**: Transcribe Jepang saja (tanpa terjemahan)
  - **Translate-srt**: Terjemahkan file SRT Jepang ke Indonesia

## Requirements

- Python 3.7+
- OpenAI Python library v1.0.0+
- ffmpeg (untuk split_audio.py)

## Instalasi

1. Clone repository ini
2. Install dependencies:

```bash
pip install -r requirements.txt
```

Atau install manual:

```bash
pip install openai configparser
```

## Konfigurasi

Buat file `config.ini` dengan format:

```ini
[OPENAI]
api_key = sk-your_openai_api_key_here
model = gpt-4o  # opsional, default: gpt-3.5-turbo
```

## Penggunaan

### Basic Usage (dengan WAV - recommended)

```bash
python whisper.py --input audio.wav --output subtitle.srt
```

### Transcribe Only (Japanese subtitle tanpa terjemahan)

```bash
python whisper.py --input audio.wav --output japanese.srt --method transcribe-only
```

### Translate Existing SRT File

```bash
python whisper.py --input japanese.srt --output indonesian.srt --method translate-srt
```

### Dengan format lain

```bash
python whisper.py --input audio.mp3 --output subtitle.srt
```

### Dengan API key langsung

```bash
python whisper.py --input audio.wav --api_key YOUR_API_KEY --output subtitle.srt
```

### Parameters

- `--input`: File audio input atau file SRT (default: audio.wav)
- `--output`: File SRT output (default: output.srt)
- `--api_key`: API key OpenAI (opsional jika menggunakan config.ini)
- `--config`: File konfigurasi (default: config.ini)
- `--model`: Model OpenAI untuk translasi (default: gpt-3.5-turbo)
  - Opsi: `gpt-3.5-turbo`, `gpt-4`, `gpt-4-turbo-preview`, `gpt-4o`, `gpt-4o-mini`
- `--whisper-model`: Model Whisper untuk transkrip (default: whisper-1)
- `--method`: Metode processing (default: transcribe)
  - `transcribe`: Transcribe ke Jepang lalu translate per segmen
  - `translate`: Direct translate lalu batch paraphrase
  - `transcribe-only`: Transcribe Jepang saja tanpa terjemahan
  - `translate-srt`: Terjemahkan file SRT Jepang yang sudah ada
- `--batch-size`: Jumlah dialog/subtitle per batch untuk translasi (default: 5)

## Contoh Penggunaan dengan Model

### Menggunakan GPT-4 untuk translasi lebih akurat

```bash
python whisper.py --input audio.wav --output subtitle.srt --model gpt-4
```

### Menggunakan GPT-4o untuk kecepatan dan kualitas

```bash
python whisper.py --input audio.wav --output subtitle.srt --model gpt-4o
```

### Menggunakan GPT-4o-mini untuk efisiensi biaya

```bash
python whisper.py --input audio.wav --output subtitle.srt --model gpt-4o-mini
```

## Metode Processing

### Metode 1: Transcribe + Translate (Default)

Proses 2 tahap tradisional:

1. Whisper transcribe audio Jepang → text Jepang
2. GPT translate text Jepang → Indonesia (per segmen)

```bash
python whisper.py --input audio.wav --method transcribe --model gpt-4o
```

### Metode 2: Direct Translate + Batch Paraphrase (Recommended)

Proses yang lebih efisien dengan konteks lebih baik:

1. Whisper direct translate audio → Indonesia
2. GPT paraphrase dalam batch untuk natural language + konteks

```bash
# Default batch size 5 dialog
python whisper.py --input audio.wav --method translate --model gpt-4o

# Custom batch size untuk dialog panjang
python whisper.py --input audio.wav --method translate --batch-size 10 --model gpt-4o
```

**Keuntungan metode translate:**

- Lebih cepat (1 Whisper call vs banyak)
- Konteks antar dialog terjaga
- Hasil lebih natural dan nyambung
- Cocok untuk dialog yang saling berkaitan

### Metode 3: Transcribe Only (Japanese Subtitle)

Hanya melakukan transcribe tanpa terjemahan:

1. Whisper transcribe audio → Japanese text dengan timing
2. Output langsung ke SRT format

```bash
# Hasilkan subtitle Jepang
python whisper.py --input anime_episode.mp3 --output japanese_subtitle.srt --method transcribe-only
```

**Kegunaan:**

- Untuk membuat subtitle Jepang asli
- Sebagai backup sebelum translasi
- Untuk keperluan pembelajaran bahasa
- Review manual sebelum translasi

### Metode 4: Translate SRT (Japanese → Indonesian)

Terjemahkan file SRT Jepang yang sudah ada:

1. Baca file SRT Jepang dengan timing
2. Translate text dengan batch processing untuk konteks
3. Output SRT Indonesia dengan timing yang sama

```bash
# Terjemahkan SRT yang sudah ada
python whisper.py --input japanese.srt --output indonesian.srt --method translate-srt --model gpt-4o

# Dengan batch size lebih besar untuk film
python whisper.py --input movie_jp.srt --output movie_id.srt --method translate-srt --batch-size 10
```

**Keuntungan:**

- Edit/koreksi subtitle Jepang dulu baru translate
- Reuse subtitle Jepang yang sudah ada
- Kontrol penuh atas timing
- Batch processing untuk konteks yang lebih baik

## Workflow Kombinasi

### Workflow 1: Review & Edit

```bash
# Step 1: Transcribe Japanese
python whisper.py --input video.mp4 --output japanese.srt --method transcribe-only

# Step 2: Manual review/edit japanese.srt (opsional)

# Step 3: Translate to Indonesian
python whisper.py --input japanese.srt --output indonesian.srt --method translate-srt
```

### Workflow 2: Multi-language Output

```bash
# Hasilkan Japanese subtitle
python whisper.py --input anime.wav --output japanese.srt --method transcribe-only

# Hasilkan Indonesian subtitle
python whisper.py --input anime.wav --output indonesian.srt --method translate
```

## Contoh Output

```srt
1
00:00:00,000 --> 00:00:02,500
Halo, apa kabar?

2
00:00:02,500 --> 00:00:05,000
Saya baik-baik saja, terima kasih.
```

## Perbandingan Model

| Model               | Kecepatan  | Kualitas   | Biaya | Rekomendasi                  |
| ------------------- | ---------- | ---------- | ----- | ---------------------------- |
| gpt-3.5-turbo       | ⭐⭐⭐⭐⭐ | ⭐⭐⭐     | $     | Default, cepat dan murah     |
| gpt-4               | ⭐⭐       | ⭐⭐⭐⭐⭐ | $$    | Translasi paling akurat      |
| gpt-4-turbo-preview | ⭐⭐⭐     | ⭐⭐⭐⭐⭐ | $$    | Balance kualitas & kecepatan |
| gpt-4o              | ⭐⭐⭐⭐   | ⭐⭐⭐⭐   | $     | Recommended untuk produksi   |
| gpt-4o-mini         | ⭐⭐⭐⭐⭐ | ⭐⭐⭐     | $     | Efisien untuk volume besar   |

## Perbandingan Metode

| Metode          | Use Case                      | Kecepatan  | Konteks    | API Calls |
| --------------- | ----------------------------- | ---------- | ---------- | --------- |
| transcribe      | Akurasi tinggi, dialog pendek | ⭐⭐⭐     | ⭐⭐⭐     | Banyak    |
| translate       | Dialog panjang & berkaitan    | ⭐⭐⭐⭐   | ⭐⭐⭐⭐⭐ | Sedang    |
| transcribe-only | Subtitle Jepang saja          | ⭐⭐⭐⭐⭐ | N/A        | Minimal   |
| translate-srt   | SRT sudah ada, mau translate  | ⭐⭐⭐⭐   | ⭐⭐⭐⭐⭐ | Sedang    |

## Tips

- Gunakan file WAV untuk kualitas transkrip terbaik (lossless audio)
- Pastikan audio memiliki kualitas yang baik untuk hasil optimal
- API key OpenAI diperlukan untuk menggunakan tool ini (kecuali transcribe-only untuk Japanese)
- **Untuk anime/film**: Gunakan `--method translate` dengan batch size 5-10 untuk dialog yang nyambung
- **Untuk presentasi/monolog**: Gunakan `--method transcribe` untuk akurasi maksimal
- **Untuk review manual**: Gunakan `--method transcribe-only` dulu, edit, baru `--method translate-srt`
- Adjust batch size berdasarkan panjang dialog (lebih panjang = batch size lebih besar)

## Catatan

Tool ini menghasilkan terjemahan bahasa Indonesia yang natural - tidak terlalu formal namun juga tidak terlalu gaul, cocok untuk subtitle yang mudah dipahami berbagai kalangan.

## Contoh Kasus Penggunaan

### 1. Anime Episode (dialog cepat & banyak)

```bash
python whisper.py --input episode1.wav --method translate --batch-size 8 --model gpt-4o
```

### 2. Video Tutorial (penjelasan teknis)

```bash
python whisper.py --input tutorial.wav --method transcribe --model gpt-4
```

### 3. Film dengan Budget Terbatas

```bash
python whisper.py --input movie.mp3 --method translate --model gpt-3.5-turbo
```

### 4. Konten dengan Dialog Penting

```bash
python whisper.py --input important.wav --method translate --batch-size 3 --model gpt-4o
```

### 5. Workflow Professional Subtitling

```bash
# Step 1: Generate Japanese subtitle
python whisper.py --input movie.wav --output movie_jp.srt --method transcribe-only

# Step 2: QC and edit movie_jp.srt manually

# Step 3: Translate to Indonesian
python whisper.py --input movie_jp.srt --output movie_id.srt --method translate-srt --model gpt-4o --batch-size 10
```

### 6. Quick Japanese Learning Material

```bash
# Generate Japanese subtitle for study
python whisper.py --input japanese_podcast.mp3 --output study_material.srt --method transcribe-only
```

## Troubleshooting

### File SRT tidak terbaca dengan benar

- Pastikan format SRT standar (index, timing, text, blank line)
- Encoding harus UTF-8
- Timing format: HH:MM:SS,mmm --> HH:MM:SS,mmm

### Error saat translate-srt

- Pastikan file input berekstensi .srt
- API key diperlukan untuk method translate-srt
- Cek koneksi internet untuk API calls

### Transcribe-only tidak menghasilkan timing

- Beberapa format audio mungkin tidak support verbose_json
- Coba konversi ke WAV terlebih dahulu untuk hasil terbaik
