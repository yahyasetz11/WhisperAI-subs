import os
import json
import time
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
    parser = argparse.ArgumentParser(description="Transcribe Japanese audio to natural Indonesian SRT")
    parser.add_argument("--input", default="audio.wav", help="File audio input (default: audio.wav)")
    parser.add_argument("--output", default="output.srt", help="File SRT output (default: output.srt)")
    parser.add_argument("--api_key", required=False, help="API key OpenAI (opsional jika menggunakan config.ini)")
    parser.add_argument("--config", default="config.ini", help="File konfigurasi (default: config.ini)")
    parser.add_argument("--model", default="gpt-3.5-turbo", 
                        help="Model OpenAI untuk translasi (default: gpt-3.5-turbo). "
                             "Opsi: gpt-3.5-turbo, gpt-4, gpt-4-turbo-preview, gpt-4o, gpt-4o-mini")
    parser.add_argument("--whisper-model", default="whisper-1", 
                        help="Model Whisper untuk transkrip (default: whisper-1)")
    parser.add_argument("--method", default="transcribe", choices=["transcribe", "translate"],
                        help="Metode processing: 'transcribe' (transcribe+translate) atau 'translate' (direct translate+paraphrase)")
    parser.add_argument("--batch-size", type=int, default=5,
                        help="Jumlah dialog per batch untuk metode translate (default: 5)")
    
    args = parser.parse_args()
    
    # Coba ambil API key dari argument
    api_key = args.api_key
    
    # Jika API key tidak ada di argument, coba ambil dari config.ini
    if not api_key:
        api_key = get_api_key_from_config(args.config)
        if not api_key:
            print(f"Error: API key tidak ditemukan. Silakan tentukan API key melalui argument --api_key atau di file {args.config}")
            print(f"Format file config.ini yang benar:")
            print("[OPENAI]")
            print("api_key = sk-your_openai_api_key_here")
            print("model = gpt-4o  # opsional")
            return
    
    # Coba ambil model dari config jika tidak ada di argument
    model = args.model
    if model == "gpt-3.5-turbo":  # Jika masih default
        config_model = get_model_from_config(args.config)
        if config_model:
            model = config_model
    
    input_file = args.input
    output_srt = args.output
    whisper_model = args.whisper_model
    method = args.method
    batch_size = args.batch_size
    
    # Validasi file input
    if not os.path.exists(input_file):
        print(f"Error: File {input_file} tidak ditemukan!")
        return
    
    # Deteksi MIME type
    mime_type = get_mime_type(input_file)
    
    print(f"Memproses file: {input_file}")
    print(f"Format audio: {mime_type}")
    print(f"Model translasi: {model}")
    print(f"Model Whisper: {whisper_model}")
    print(f"Metode: {method}")
    if method == "translate":
        print(f"Batch size: {batch_size}")
    
    try:
        # Cek ukuran file
        file_size = os.path.getsize(input_file)
        print(f"Ukuran file: {file_size / 1024 / 1024:.1f} MB")
        
        if file_size > 25 * 1024 * 1024:  # 25 MB
            print("WARNING: File melebihi batas 25MB! API akan menolak file ini.")
            print("Gunakan split_audio.py untuk membagi file terlebih dahulu.")
            return
        
        # Inisialisasi OpenAI client
        client = OpenAI(api_key=api_key)
        
        # Process berdasarkan metode yang dipilih
        if method == "transcribe":
            translated_segments = process_transcribe_method(client, input_file, model, whisper_model, batch_size)
        else:  # method == "translate"
            translated_segments = process_translate_method(client, input_file, model, whisper_model, batch_size)
        
        if not translated_segments:
            print("Tidak ada segmen yang berhasil diproses.")
            return
        
        # Langkah 3: Buat konten SRT
        srt_content = create_srt(translated_segments)
        
        # Langkah 4: Tulis ke file
        with open(output_srt, "w", encoding="utf-8") as srt_file:
            srt_file.write(srt_content)
        
        print(f"\nTerjemahan selesai!")
        print(f"Total segmen: {len(translated_segments)}")
        print(f"Output disimpan ke: {output_srt}")
    
    except FileNotFoundError:
        print(f"Error: File {input_file} tidak dapat dibuka.")
    except Exception as e:
        print(f"Terjadi kesalahan: {str(e)}")
        
        # Error handling khusus untuk OpenAI errors
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