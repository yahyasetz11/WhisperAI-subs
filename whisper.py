import os
import json
import time
import re
from datetime import timedelta
import argparse
from openai import OpenAI
import configparser
from faster_whisper import WhisperModel

# Fungsi untuk mengubah detik ke format waktu SRT (HH:MM:SS,mmm)
def format_time(seconds):
    # Konversi detik ke timedelta
    td = timedelta(seconds=float(seconds))
    
    # Hitung jam, menit, detik, dan milidetik
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = int((td.microseconds / 1000))
    
    # Format ke format waktu SRT
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

# Fungsi untuk parse waktu SRT ke detik
def parse_srt_time(time_str):
    """Convert SRT time format to seconds"""
    # Format: HH:MM:SS,mmm
    time_parts = time_str.replace(',', '.').split(':')
    hours = int(time_parts[0])
    minutes = int(time_parts[1])
    seconds = float(time_parts[2])
    
    return hours * 3600 + minutes * 60 + seconds

# Fungsi untuk membaca file SRT
def read_srt_file(filepath):
    """Read and parse SRT file"""
    segments = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by double newline to get individual subtitles
    subtitle_blocks = content.strip().split('\n\n')
    
    for block in subtitle_blocks:
        lines = block.strip().split('\n')
        if len(lines) >= 3:  # Must have index, time, and text
            try:
                # Parse time line (second line)
                time_line = lines[1]
                start_time_str, end_time_str = time_line.split(' --> ')
                
                # Get text (remaining lines)
                text = '\n'.join(lines[2:])
                
                segment = {
                    'start': parse_srt_time(start_time_str.strip()),
                    'end': parse_srt_time(end_time_str.strip()),
                    'text': text.strip()
                }
                segments.append(segment)
            except Exception as e:
                print(f"Warning: Skipping malformed subtitle block: {e}")
                continue
    
    return segments

# Fungsi untuk mengubah transkrip ke format SRT
def create_srt(segments):
    srt_content = ""
    for i, segment in enumerate(segments):
        # Ambil waktu mulai dan selesai
        start_time = format_time(segment['start'])
        end_time = format_time(segment['end'])
        
        # Ambil teks
        text = segment['text'].strip()
        
        # Format sebagai SRT
        srt_content += f"{i+1}\n{start_time} --> {end_time}\n{text}\n\n"
    
    return srt_content

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

def get_api_key_from_config(config_file):
    """Membaca API key dari file config.ini"""
    if not os.path.exists(config_file):
        return None
        
    config = configparser.ConfigParser()
    config.read(config_file)
    
    try:
        return config['OPENAI']['api_key']
    except (KeyError, configparser.NoSectionError):
        return None

def get_model_from_config(config_file):
    """Membaca model preference dari file config.ini"""
    if not os.path.exists(config_file):
        return None
        
    config = configparser.ConfigParser()
    config.read(config_file)
    
    try:
        return config['OPENAI'].get('model', None)
    except (KeyError, configparser.NoSectionError):
        return None

def get_mime_type(filename):
    """Menentukan MIME type berdasarkan ekstensi file"""
    ext = os.path.splitext(filename)[1].lower()
    mime_types = {
        '.wav': 'audio/wav',
        '.mp3': 'audio/mpeg',
        '.mp4': 'audio/mp4',
        '.m4a': 'audio/mp4',
        '.ogg': 'audio/ogg',
        '.flac': 'audio/flac',
        '.webm': 'audio/webm'
    }
    return mime_types.get(ext, 'audio/wav')

def transcribe_local(audio_path, model_name="jctv-tech/kotoba-whisper-v21-ct2", device="cuda", compute_type="int8"):
    """Transcribe Japanese audio using local faster-whisper model.

    Returns list of segments with 'start', 'end', 'text' keys.
    """
    print(f"Loading model: {model_name} (device={device}, compute_type={compute_type})")
    model = WhisperModel(model_name, device=device, compute_type=compute_type)

    print("Transcribing with VAD filter...")
    segments_iter, info = model.transcribe(
        audio_path,
        language="ja",
        vad_filter=False,
        condition_on_previous_text=False,
        beam_size=5,
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

# NEW: Transcribe only method (no translation)
def process_transcribe_only_method(input_file, whisper_model, device, compute_type):
    """Transcribe Japanese audio without translation using local model."""
    print("Menggunakan metode: Transcribe Only (Japanese)")
    print(f"Model: {whisper_model}")

    segments = transcribe_local(input_file, whisper_model, device, compute_type)

    if not segments:
        print("Tidak ada segmen ditemukan.")
        return []

    return segments

# NEW: Translate SRT file method
def process_translate_srt_method(client, input_srt, model, batch_size=5):
    """Method to translate existing Japanese SRT file to Indonesian"""
    print(f"Menggunakan metode: Translate SRT File")
    print(f"Model translasi: {model}")
    print(f"Batch size: {batch_size} subtitle per batch")
    print(f"Membaca file SRT: {input_srt}")
    
    # Read SRT file
    segments = read_srt_file(input_srt)
    
    if not segments:
        print("Tidak ada subtitle yang ditemukan dalam file SRT.")
        return []
    
    print(f"Total subtitle ditemukan: {len(segments)}")
    
    translated_segments = []
    
    # Process in batches
    for batch_start in range(0, len(segments), batch_size):
        batch_end = min(batch_start + batch_size, len(segments))
        batch_segments = segments[batch_start:batch_end]
        
        print(f"\nMenerjemahkan batch {batch_start//batch_size + 1} (subtitle {batch_start+1}-{batch_end})...")
        
        # Combine texts with markers
        batch_texts = []
        for i, seg in enumerate(batch_segments):
            text = seg['text']
            # Remove null bytes and control characters that break JSON serialization
            text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
            batch_texts.append(f"[Subtitle {i+1}] {text}")

        combined_text = "\n".join(batch_texts)
        
        # Send to GPT for batch translation
        try:
            chat_completion = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": TRANSLATION_SYSTEM_PROMPT.replace("[Dialog X]", "[Subtitle X]")
                    },
                    {"role": "user", "content": combined_text}
                ],
                temperature=0.6
            )
            
            result_text = chat_completion.choices[0].message.content.strip()
            
            # Parse translation results
            translated_dialogs = {}
            current_subtitle = None
            current_text = []
            
            for line in result_text.split('\n'):
                if line.strip().startswith('[Subtitle'):
                    # Save previous subtitle if exists
                    if current_subtitle is not None:
                        translated_dialogs[current_subtitle] = ' '.join(current_text).strip()
                    
                    # Extract subtitle number
                    try:
                        subtitle_num = int(line.split(']')[0].split()[-1])
                        current_subtitle = subtitle_num
                        # Extract text after [Subtitle X]
                        text_after = line.split(']', 1)[1].strip() if ']' in line else ''
                        current_text = [text_after] if text_after else []
                    except:
                        continue
                elif current_subtitle is not None and line.strip():
                    current_text.append(line.strip())
            
            # Save last subtitle
            if current_subtitle is not None:
                translated_dialogs[current_subtitle] = ' '.join(current_text).strip()
            
            # Create translated segments
            for i, seg in enumerate(batch_segments):
                subtitle_num = i + 1
                translated = translated_dialogs.get(subtitle_num, '')
                
                # Fallback to original if translation failed
                if not translated:
                    print(f"  Warning: Subtitle {batch_start + i + 1} gagal diterjemahkan, menggunakan text original")
                    translated = seg['text']
                
                translated_segments.append({
                    'start': seg['start'],
                    'end': seg['end'],
                    'text': translated
                })
            
            # Delay between batches to avoid rate limits
            if batch_end < len(segments):
                time.sleep(1)
                
        except Exception as e:
            print(f"  Error saat menerjemahkan batch: {str(e)}")
            # Add original segments on error
            for seg in batch_segments:
                translated_segments.append(seg)
    
    return translated_segments

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


def main():
    parser = argparse.ArgumentParser(description="Transcribe/Translate Japanese audio/SRT to Indonesian")
    
    # Input/Output arguments
    parser.add_argument("--input", default="audio.wav", 
                        help="File audio input atau file SRT (default: audio.wav)")
    parser.add_argument("--output", default="output.srt", 
                        help="File SRT output (default: output.srt)")
    
    # API Configuration
    parser.add_argument("--api_key", required=False, 
                        help="API key OpenAI (opsional jika menggunakan config.ini)")
    parser.add_argument("--config", default="config.ini", 
                        help="File konfigurasi (default: config.ini)")
    parser.add_argument("--model", default="gpt-3.5-turbo", 
                        help="Model OpenAI untuk translasi (default: gpt-3.5-turbo). "
                             "Opsi: gpt-3.5-turbo, gpt-4, gpt-4-turbo-preview, gpt-4o, gpt-4o-mini")
    parser.add_argument("--whisper-model", default="jctv-tech/kotoba-whisper-v21-ct2",
                        help="Model Whisper lokal (default: jctv-tech/kotoba-whisper-v21-ct2)")
    
    # Method selection
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

    parser.add_argument("--compute-type", default="int8",
                        choices=["float16", "int8", "float32"],
                        help="Compute type untuk model Whisper (default: int8)")

    args = parser.parse_args()
    
    # Validate input based on method
    method = args.method
    input_file = args.input
    output_srt = args.output
    
    # Check if input is SRT for translate-srt method
    if method == "translate-srt":
        if not input_file.lower().endswith('.srt'):
            print(f"Error: Metode 'translate-srt' memerlukan file SRT sebagai input!")
            print(f"File yang diberikan: {input_file}")
            return
    else:
        # For audio methods, check if it's an audio file
        if input_file.lower().endswith('.srt'):
            print(f"Warning: File input adalah SRT, tetapi metode '{method}' memerlukan file audio.")
            print("Gunakan --method translate-srt untuk menerjemahkan file SRT.")
            return
    
    # Check if file exists
    if not os.path.exists(input_file):
        print(f"Error: File {input_file} tidak ditemukan!")
        return
    
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
    
    # Get model preference
    model = args.model
    if model == "gpt-3.5-turbo":  # If still default
        config_model = get_model_from_config(args.config)
        if config_model:
            model = config_model
    
    whisper_model = args.whisper_model
    batch_size = args.batch_size
    device = args.device
    compute_type = args.compute_type
    
    # Initialize OpenAI client only if needed
    client = None
    if needs_api:
        try:
            client = OpenAI(api_key=api_key)
        except Exception as e:
            print(f"Error inisialisasi OpenAI client: {str(e)}")
            return
    
    # Process based on selected method
    print(f"\n{'='*60}")
    print(f"Memproses file: {input_file}")
    
    try:
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
        
        # Check if we got segments
        if not segments:
            print("\nTidak ada segmen yang berhasil diproses.")
            return
        
        # Create SRT content
        srt_content = create_srt(segments)
        
        # Write to output file
        with open(output_srt, "w", encoding="utf-8") as srt_file:
            srt_file.write(srt_content)
        
        print(f"\n{'='*60}")
        print(f"✓ Proses selesai!")
        print(f"  Total segmen: {len(segments)}")
        print(f"  Output disimpan ke: {output_srt}")
        
        # Show sample of result
        if segments:
            print(f"\n  Sample hasil (segmen pertama):")
            print(f"  {segments[0]['text'][:100]}...")
        
    except FileNotFoundError:
        print(f"Error: File {input_file} tidak dapat dibuka.")
    except Exception as e:
        print(f"Terjadi kesalahan: {str(e)}")
        
        # Error handling for specific OpenAI errors
        if "api_key" in str(e).lower():
            print("\nMasalah API Key:")
            print("- Pastikan API key valid")
            print("- Cek format: sk-...")
            print("- Pastikan tidak ada spasi di awal/akhir")
        elif "rate_limit" in str(e).lower():
            print("\nRate limit tercapai. Tunggu beberapa saat dan coba lagi.")
        elif "maximum" in str(e).lower() and "size" in str(e).lower():
            print("\nFile terlalu besar. Gunakan split_audio.py untuk membagi file.")
        
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()