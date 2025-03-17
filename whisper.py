import os
import json
import time
from datetime import timedelta
import argparse
import requests
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

# Fungsi untuk membuat bahasa Indonesia lebih santai/gaul
def make_casual_indonesian(text):
    # Kamus kata formal ke gaul
    casual_dict = {
        "saya": "gue",
        "aku": "gue",
        "kamu": "lu",
        "anda": "elu",
        "tidak": "nggak",
        "tidak ada": "gak ada",
        "sedang": "lagi",
        "ingin": "pengen",
        "bagaimana": "gimana",
        "mengapa": "kenapa",
        "begitu": "gitu",
        "seperti": "kayak",
        "telah": "udah",
        "sudah": "udah",
        "hanya": "cuma",
        "sangat": "banget",
        "sekali": "banget",
        "bagus sekali": "keren banget",
        "boleh": "bisa",
        "tetapi": "tapi",
        "namun": "tapi",
        "dapat": "bisa",
        "berkata": "ngomong",
        "mengatakan": "ngomong",
        "berbicara": "ngomong",
        "pergi": "cabut",
        "baik": "oke",
        "teman": "temen",
        "sahabat": "bestie",
        "uang": "duit",
        "cantik": "cakep",
        "tampan": "ganteng",
        "laki-laki": "cowok",
        "perempuan": "cewek",
        "benar": "bener",
        "sungguh": "beneran",
        "serius": "serius",
        "tertawa": "ketawa",
        "terima kasih": "makasih",
        "maaf": "sori",
        "mungkin": "kali",
        "sebentar": "bentar",
        "sekarang": "sekarang",
        "kemarin": "kemaren",
        "besok": "besok",
        "rumah": "rumah",
        "makan": "makan",
        "kenapa": "ngapain",
        "melihat": "ngeliat",
        "mengerti": "ngerti",
        "memberi": "ngasih",
        "melakukan": "ngelakuin",
        "mengambil": "ngambil",
    }
    
    # Ubah kata formal ke gaul
    casual_text = text
    for formal, casual in casual_dict.items():
        # Ganti kata lengkap saja (dengan batas kata)
        casual_text = casual_text.replace(f" {formal} ", f" {casual} ")
        
        # Cek kata di awal teks
        if casual_text.startswith(f"{formal} "):
            casual_text = casual_text.replace(f"{formal} ", f"{casual} ", 1)
        
        # Cek kata di akhir teks
        if casual_text.endswith(f" {formal}"):
            casual_text = casual_text[:-len(f" {formal}")] + f" {casual}"
        
        # Ganti kata jika itu adalah satu-satunya kata dalam teks
        if casual_text == formal:
            casual_text = casual
    
    # Tambahkan ekspresi gaul umum
    casual_expressions = [
        ("!", " banget!"),
        ("Ya", "Iya nih"),
        ("Baiklah", "Oke deh"),
        ("Terima kasih", "Makasih"),
        ("Selamat", "Congrats"),
        ("Maaf", "Sori"),
    ]
    
    for formal_expr, casual_expr in casual_expressions:
        casual_text = casual_text.replace(formal_expr, casual_expr)
    
    # Tambahkan kata santai di akhir kalimat
    if "." in casual_text and len(casual_text) > 20:
        sentences = casual_text.split(".")
        for i in range(len(sentences) - 1):
            if len(sentences[i]) > 10 and not sentences[i].endswith(("sih", "dong", "deh", "lho", "nih")):
                import random
                suffix = random.choice(["sih", "dong", "deh", "lho", "nih"])
                if random.random() < 0.4:  # 40% peluang
                    sentences[i] = sentences[i] + " " + suffix
        casual_text = ".".join(sentences)
    
    return casual_text

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

def main():
    parser = argparse.ArgumentParser(description="Transcribe Japanese audio to casual Indonesian SRT")
    parser.add_argument("--input", default="audio.mp3", help="File audio input (default: audio.mp3)")
    parser.add_argument("--output", default="output.srt", help="File SRT output (default: output.srt)")
    parser.add_argument("--api_key", required=False, help="API key OpenAI kamu (opsional jika menggunakan config.ini)")
    parser.add_argument("--config", default="config.ini", help="File konfigurasi (default: config.ini)")
    
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
            return
    
    input_file = args.input
    output_srt = args.output
    
    print(f"Memproses file: {input_file}")
    
    try:
        # Langkah 1: Transkrip audio Jepang menggunakan API Whisper
        print("Mentranskrip dengan API Whisper...")
        
        headers = {
            "Authorization": f"Bearer {api_key}"
        }
        
        with open(input_file, "rb") as audio_file:
            files = {
                "file": (os.path.basename(input_file), audio_file, "audio/mpeg"),
                "model": (None, "whisper-1"),
                "language": (None, "ja"),
                "response_format": (None, "verbose_json")
            }
            
            response = requests.post(
                "https://api.openai.com/v1/audio/transcriptions",
                headers=headers,
                files=files
            )
            
            if response.status_code != 200:
                print(f"Error dari API OpenAI: {response.text}")
                return
            
            transcription_data = response.json()
        
        # Ekstrak segmen
        segments = transcription_data.get("segments", [])
        
        if not segments:
            print("Tidak ada segmen yang ditemukan dalam transkrip.")
            return
        
        # Langkah 2: Terjemahkan setiap segmen ke bahasa Indonesia gaul
        print("Menerjemahkan ke bahasa Indonesia gaul...")
        translated_segments = []
        
        for segment in segments:
            japanese_text = segment.get("text", "").strip()
            
            if not japanese_text:
                continue
            
            # Gunakan API Chat untuk menerjemahkan ke bahasa Indonesia
            chat_data = {
                "model": "gpt-3.5-turbo",
                "messages": [
                    {"role": "system", "content": "Kamu adalah penerjemah handal dari bahasa Jepang ke bahasa Indonesia. Terjemahkan teks berikut ke dalam bahasa Indonesia yang santai dan percakapan dengan bahasa gaul yang sesuai. Gunakan bahasa gaul anak Jakarta jika memungkinkan."},
                    {"role": "user", "content": japanese_text}
                ],
                "temperature": 0.7
            }
            
            chat_response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json=chat_data
            )
            
            if chat_response.status_code != 200:
                print(f"Error dari API Chat: {chat_response.text}")
                continue
            
            chat_result = chat_response.json()
            translated_text = chat_result["choices"][0]["message"]["content"].strip()
            
            # Buat lebih gaul
            casual_translated_text = make_casual_indonesian(translated_text)
            
            # Buat segment baru
            new_segment = {
                "start": segment.get("start", 0),
                "end": segment.get("end", 0),
                "text": casual_translated_text
            }
            
            translated_segments.append(new_segment)
            
            # Tunggu sebentar untuk menghindari batas rate
            time.sleep(0.5)
        
        # Langkah 3: Buat konten SRT
        srt_content = create_srt(translated_segments)
        
        # Langkah 4: Tulis ke file
        with open(output_srt, "w", encoding="utf-8") as srt_file:
            srt_file.write(srt_content)
        
        print(f"Terjemahan selesai! Output disimpan ke {output_srt}")
    
    except Exception as e:
        print(f"Terjadi kesalahan: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()