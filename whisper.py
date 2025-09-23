import os
import json
import time
import re
from datetime import timedelta
import argparse
from openai import OpenAI
import configparser

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

# Fungsi untuk membuat bahasa Indonesia lebih natural/sehari-hari
def make_natural_indonesian(text):
    # Kamus kata formal ke sehari-hari (lebih moderat)
    natural_dict = {
        # Kata ganti - tetap formal untuk kesopanan
        "saya": "saya",
        "anda": "Anda",
        
        # Kata kerja yang umum diganti dalam percakapan sehari-hari
        "telah": "sudah",
        "sedang": "lagi",
        "ingin": "mau",
        "dapat": "bisa",
        "berkata": "bilang",
        "mengatakan": "bilang",
        "melihat": "lihat",
        "mengerti": "ngerti",
        "memberi": "kasih",
        "mengambil": "ambil",
        "melakukan": "lakukan",
        
        # Kata sambung dan keterangan
        "tetapi": "tapi",
        "namun": "tapi",
        "hanya": "cuma",
        "sangat": "sangat",
        "sekali": "sekali",
        "sebentar": "sebentar",
        "sekarang": "sekarang",
        "kemarin": "kemarin",
        
        # Kata tanya
        "mengapa": "kenapa",
        "bagaimana": "gimana",
        
        # Ekspresi umum
        "terima kasih": "terima kasih",
        "maaf": "maaf",
        "baik": "baik",
        "benar": "benar",
        
        # Kata sifat
        "cantik": "cantik",
        "tampan": "tampan",
        "bagus": "bagus",

        # Kata lainnya
        "varietas": "variety",
    }
    
    # Ubah kata formal ke natural
    natural_text = text
    for formal, natural in natural_dict.items():
        # Ganti kata lengkap saja (dengan batas kata)
        natural_text = natural_text.replace(f" {formal} ", f" {natural} ")
        
        # Cek kata di awal teks
        if natural_text.startswith(f"{formal} "):
            natural_text = natural_text.replace(f"{formal} ", f"{natural} ", 1)
        
        # Cek kata di akhir teks
        if natural_text.endswith(f" {formal}"):
            natural_text = natural_text[:-len(f" {formal}")] + f" {natural}"
        
        # Ganti kata jika itu adalah satu-satunya kata dalam teks
        if natural_text == formal:
            natural_text = natural
    
    # Tambahkan partikel yang natural tapi tidak berlebihan
    if "." in natural_text and len(natural_text) > 30:
        sentences = natural_text.split(".")
        for i in range(len(sentences) - 1):
            # Hanya tambahkan partikel pada 20% kalimat untuk terasa natural
            if len(sentences[i]) > 15 and not sentences[i].endswith(("sih", "ya", "kok", "lho")):
                import random
                if random.random() < 0.2:  # 20% peluang
                    suffix = random.choice(["ya", "kok", "lho"])
                    sentences[i] = sentences[i] + " " + suffix
        natural_text = ".".join(sentences)
    
    return natural_text

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

# NEW: Transcribe only method (no translation)
def process_transcribe_only_method(client, input_file, whisper_model):
    """Method to only transcribe Japanese audio without translation"""
    print("Menggunakan metode: Transcribe Only (Japanese)")
    print(f"Model Whisper: {whisper_model}")
    print("Melakukan transkripsi bahasa Jepang...")
    
    with open(input_file, "rb") as audio_file:
        try:
            # Try verbose_json for segments with timing
            transcription = client.audio.transcriptions.create(
                file=audio_file,
                model=whisper_model,
                language="ja",
                response_format="verbose_json"
            )
            
            # Extract segments
            segments = transcription.segments if hasattr(transcription, 'segments') else []
            
            if not segments:
                print("Tidak ada segmen ditemukan, mencoba format text...")
                # Fallback to text
                audio_file.seek(0)
                transcription = client.audio.transcriptions.create(
                    file=audio_file,
                    model=whisper_model,
                    language="ja"
                )
                
                # Create single segment from text
                if hasattr(transcription, 'text') and transcription.text:
                    segments = [{
                        'start': 0,
                        'end': 10,  # Default duration
                        'text': transcription.text
                    }]
                else:
                    print("Tidak ada hasil transkripsi.")
                    return []
            
            print(f"Transkripsi selesai! Total segmen: {len(segments)}")
            
            # Format segments properly
            formatted_segments = []
            for seg in segments:
                if hasattr(seg, 'start'):
                    start_time = seg.start
                    end_time = seg.end
                    text = seg.text
                elif isinstance(seg, dict):
                    start_time = seg.get('start', 0)
                    end_time = seg.get('end', 0)
                    text = seg.get('text', '')
                else:
                    continue
                
                formatted_segments.append({
                    'start': start_time,
                    'end': end_time,
                    'text': text.strip()
                })
            
            return formatted_segments
            
        except Exception as e:
            print(f"Error saat transkripsi: {str(e)}")
            return []

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
            batch_texts.append(f"[Subtitle {i+1}] {text}")
        
        combined_text = "\n".join(batch_texts)
        
        # Send to GPT for batch translation
        try:
            chat_completion = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "Kamu adalah penerjemah subtitle profesional dari bahasa Jepang ke bahasa Indonesia. "
                                   "Terjemahkan setiap subtitle dalam tanda [Subtitle X] ke bahasa Indonesia yang natural, sopan, dan mudah dipahami. "
                                   "Pertahankan format [Subtitle X] agar bisa dicocokkan kembali ke segmen aslinya. "
                                   "Jaga konteks antar subtitle agar cerita tetap nyambung. "
                                   "Gunakan bahasa Indonesia sehari-hari yang tidak terlalu formal tapi tetap sopan."
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
                
                # Apply natural Indonesian
                natural_translated = make_natural_indonesian(translated)
                
                translated_segments.append({
                    'start': seg['start'],
                    'end': seg['end'],
                    'text': natural_translated
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

def process_transcribe_method(client, input_file, model, whisper_model, batch_size=5):
    print("Menggunakan metode: Transcribe (Whisper) + Translate (GPT-4o dengan batching)")
    
    with open(input_file, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            file=audio_file,
            model=whisper_model,
            language="ja",
            response_format="verbose_json"
        )

    segments = transcription.segments if hasattr(transcription, 'segments') else []
    if not segments:
        print("Tidak ada segmen ditemukan.")
        return []

    print(f"Total segmen ditemukan: {len(segments)}")
    translated_segments = []

    for batch_start in range(0, len(segments), batch_size):
        batch_end = min(batch_start + batch_size, len(segments))
        batch_segments = segments[batch_start:batch_end]

        print(f"Menerjemahkan batch {batch_start//batch_size + 1} (segmen {batch_start+1}-{batch_end})...")

        # Gabungkan teks dengan penanda [Dialog X]
        batch_texts = []
        for i, seg in enumerate(batch_segments):
            text = seg.text if hasattr(seg, 'text') else seg.get('text', '')
            batch_texts.append(f"[Dialog {i+1}] {text.strip()}")

        combined_text = "\n".join(batch_texts)

        # Kirim ke GPT untuk batch translate
        chat_completion = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "Kamu adalah penerjemah subtitle dari bahasa Jepang ke bahasa Indonesia. "
                               "Terjemahkan setiap dialog dalam tanda [Dialog X] ke bahasa Indonesia yang natural, sopan, dan mudah dipahami. "
                               "Pertahankan format [Dialog X] agar bisa dicocokkan kembali ke segmen aslinya."
                },
                {"role": "user", "content": combined_text}
            ],
            temperature=0.6
        )

        result_text = chat_completion.choices[0].message.content.strip()

        # Parsing hasil terjemahan
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
            elif current_dialog is not None:
                current_text.append(line.strip())

        if current_dialog is not None:
            translated_dialogs[current_dialog] = ' '.join(current_text).strip()

        # Masukkan hasil ke translated_segments
        for i, seg in enumerate(batch_segments):
            dialog_num = i + 1
            translated = translated_dialogs.get(dialog_num, '')
            if not translated:
                translated = seg.text  # fallback
            natural_translated = make_natural_indonesian(translated)
            new_segment = {
                "start": seg.start,
                "end": seg.end,
                "text": natural_translated
            }
            translated_segments.append(new_segment)

        time.sleep(1)  # Jeda antar batch

    return translated_segments


def process_translate_method(client, input_file, model, whisper_model, batch_size):
    """Metode baru: Whisper direct translate, lalu GPT paraphrase dengan batch"""
    print("Menggunakan metode: Direct Translate + Batch Paraphrase")
    print(f"Batch size: {batch_size} dialog per batch")
    print("Menerjemahkan langsung dengan API Whisper...")
    
    with open(input_file, "rb") as audio_file:
        # Try verbose_json first for segments
        try:
            translation = client.audio.translations.create(
                model=whisper_model,
                file=audio_file,
                response_format="verbose_json"
            )
        except:
            # Fallback to regular text if verbose_json not supported
            print("Verbose format tidak tersedia, menggunakan text format...")
            audio_file.seek(0)  # Reset file pointer
            translation = client.audio.translations.create(
                model=whisper_model,
                file=audio_file
            )
    
    print("Terjemahan Whisper selesai!")
    
    # Ekstrak segmen
    segments = translation.segments if hasattr(translation, 'segments') else []
    
    if not segments:
        # Jika tidak ada segments, coba ambil dari text
        if hasattr(translation, 'text') and translation.text:
            print("Warning: Tidak ada timing segments, menggunakan full text")
            # Buat satu segment dari keseluruhan text
            # Split by sentences untuk simulasi segments
            import re
            sentences = re.split(r'[.!?]+', translation.text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            # Estimate timing (assume 3 seconds per sentence)
            segments = []
            current_time = 0
            for sent in sentences:
                segments.append({
                    'start': current_time,
                    'end': current_time + 3,
                    'text': sent
                })
                current_time += 3.5
        else:
            print("Tidak ada terjemahan yang ditemukan.")
            return []
    
    print(f"Ditemukan {len(segments)} segmen untuk di-paraphrase.")
    
    # Process dalam batch
    paraphrased_segments = []
    
    for batch_start in range(0, len(segments), batch_size):
        batch_end = min(batch_start + batch_size, len(segments))
        batch_segments = segments[batch_start:batch_end]
        
        print(f"\nMemproses batch {batch_start//batch_size + 1} (segmen {batch_start+1}-{batch_end})...")
        
        # Gabungkan text dari batch untuk context
        batch_texts = []
        for i, seg in enumerate(batch_segments):
            if hasattr(seg, 'text'):
                text = seg.text
            elif isinstance(seg, dict) and 'text' in seg:
                text = seg.get('text', '')
            else:
                text = str(seg)  # Fallback
            batch_texts.append(f"[Dialog {i+1}] {text.strip()}")
        
        combined_text = "\n".join(batch_texts)
        
        # Paraphrase batch dengan GPT
        chat_completion = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system", 
                    "content": "Kamu adalah editor subtitle bahasa Indonesia. "
                               "Ubah dialog-dialog berikut menjadi bahasa Indonesia yang natural dan mudah dipahami. "
                               "Gunakan bahasa sehari-hari yang sopan, tidak terlalu formal namun juga tidak terlalu gaul. "
                               "PENTING: Pertahankan format [Dialog X] dan jaga konteks antar dialog agar nyambung. "
                               "Setiap dialog harus tetap terpisah dengan format yang sama."
                },
                {"role": "user", "content": combined_text}
            ],
            temperature=0.6
        )
        
        paraphrased_text = chat_completion.choices[0].message.content.strip()
        
        # Parse hasil paraphrase
        paraphrased_dialogs = {}
        current_dialog = None
        current_text = []
        
        for line in paraphrased_text.split('\n'):
            if line.strip().startswith('[Dialog'):
                # Save previous dialog if exists
                if current_dialog is not None:
                    paraphrased_dialogs[current_dialog] = ' '.join(current_text).strip()
                
                # Extract dialog number
                try:
                    dialog_num = int(line.split(']')[0].split()[-1])
                    current_dialog = dialog_num
                    # Extract text after [Dialog X]
                    text_after = line.split(']', 1)[1].strip() if ']' in line else ''
                    current_text = [text_after] if text_after else []
                except:
                    continue
            elif current_dialog is not None and line.strip():
                current_text.append(line.strip())
        
        # Save last dialog
        if current_dialog is not None:
            paraphrased_dialogs[current_dialog] = ' '.join(current_text).strip()
        
        # Create segments dengan text yang di-paraphrase
        for i, seg in enumerate(batch_segments):
            dialog_num = i + 1
            paraphrased = paraphrased_dialogs.get(dialog_num, '')
            
            # Fallback ke text original jika parsing gagal
            if not paraphrased:
                if hasattr(seg, 'text'):
                    original_text = seg.text
                elif isinstance(seg, dict) and 'text' in seg:
                    original_text = seg.get('text', '')
                else:
                    original_text = str(seg)
                paraphrased = make_natural_indonesian(original_text)
            
            # Extract timing info safely
            if hasattr(seg, 'start'):
                start_time = seg.start
                end_time = seg.end
            elif isinstance(seg, dict):
                start_time = seg.get('start', 0)
                end_time = seg.get('end', 0)
            else:
                start_time = 0
                end_time = 0
            
            new_segment = {
                "start": start_time,
                "end": end_time,
                "text": paraphrased
            }
            
            paraphrased_segments.append(new_segment)
        
        # Delay antar batch
        if batch_end < len(segments):
            time.sleep(1)
    
    return paraphrased_segments

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
    
    # Get API key for methods that need it
    api_key = None
    if method != "transcribe-only" or method == "translate-srt":  # These methods need API key
        api_key = args.api_key
        if not api_key:
            api_key = get_api_key_from_config(args.config)
            if not api_key and method != "transcribe-only":
                print(f"Error: API key diperlukan untuk metode '{method}'.")
                print(f"Silakan tentukan API key melalui argument --api_key atau di file {args.config}")
                print(f"\nFormat file config.ini:")
                print("[OPENAI]")
                print("api_key = sk-your_openai_api_key_here")
                print("model = gpt-4o  # opsional")
                return
    
    # For transcribe-only, we still need API key
    if method == "transcribe-only":
        api_key = args.api_key
        if not api_key:
            api_key = get_api_key_from_config(args.config)
            if not api_key:
                print(f"Error: API key diperlukan untuk metode 'transcribe-only'.")
                print(f"Silakan tentukan API key melalui argument --api_key atau di file {args.config}")
                return
    
    # Get model preference
    model = args.model
    if model == "gpt-3.5-turbo":  # If still default
        config_model = get_model_from_config(args.config)
        if config_model:
            model = config_model
    
    whisper_model = args.whisper_model
    batch_size = args.batch_size
    
    # Initialize OpenAI client
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
            # Transcribe only (no translation)
            mime_type = get_mime_type(input_file)
            print(f"Format audio: {mime_type}")
            
            # Check file size
            file_size = os.path.getsize(input_file)
            print(f"Ukuran file: {file_size / 1024 / 1024:.1f} MB")
            
            if file_size > 25 * 1024 * 1024:
                print("WARNING: File melebihi batas 25MB! API akan menolak file ini.")
                print("Gunakan split_audio.py untuk membagi file terlebih dahulu.")
                return
            
            segments = process_transcribe_only_method(client, input_file, whisper_model)
            
        elif method == "translate-srt":
            # Translate existing SRT file
            print(f"Metode: Translate SRT")
            print(f"Model translasi: {model}")
            segments = process_translate_srt_method(client, input_file, model, batch_size)
            
        elif method == "transcribe":
            # Original transcribe + translate method
            mime_type = get_mime_type(input_file)
            print(f"Format audio: {mime_type}")
            print(f"Model translasi: {model}")
            print(f"Model Whisper: {whisper_model}")
            
            # Check file size
            file_size = os.path.getsize(input_file)
            print(f"Ukuran file: {file_size / 1024 / 1024:.1f} MB")
            
            if file_size > 25 * 1024 * 1024:
                print("WARNING: File melebihi batas 25MB! API akan menolak file ini.")
                return
            
            segments = process_transcribe_method(client, input_file, model, whisper_model, batch_size)
            
        else:  # method == "translate"
            # Direct translate + paraphrase method
            mime_type = get_mime_type(input_file)
            print(f"Format audio: {mime_type}")
            print(f"Model translasi: {model}")
            print(f"Model Whisper: {whisper_model}")
            
            # Check file size
            file_size = os.path.getsize(input_file)
            print(f"Ukuran file: {file_size / 1024 / 1024:.1f} MB")
            
            if file_size > 25 * 1024 * 1024:
                print("WARNING: File melebihi batas 25MB! API akan menolak file ini.")
                return
            
            segments = process_translate_method(client, input_file, model, whisper_model, batch_size)
        
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