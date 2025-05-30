import os
import subprocess
import argparse
from pathlib import Path

def get_audio_duration(file_path):
    """Mendapatkan durasi audio dalam detik menggunakan ffprobe"""
    cmd = [
        'ffprobe', '-v', 'error', '-show_entries',
        'format=duration', '-of',
        'default=noprint_wrappers=1:nokey=1', file_path
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        return float(result.stdout)
    except:
        print("Error: Pastikan ffmpeg terinstall!")
        return None

def check_ffmpeg():
    """Cek apakah ffmpeg terinstall"""
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True)
        return True
    except FileNotFoundError:
        return False

def split_audio(input_file, chunk_duration=300, output_dir="chunks"):
    """
    Split audio menjadi beberapa bagian
    chunk_duration: durasi per bagian dalam detik (default 5 menit)
    """
    # Buat direktori output jika belum ada
    Path(output_dir).mkdir(exist_ok=True)
    
    # Dapatkan durasi total
    total_duration = get_audio_duration(input_file)
    if not total_duration:
        return []
    
    print(f"Durasi total: {total_duration:.1f} detik ({total_duration/60:.1f} menit)")
    
    # Hitung jumlah bagian
    num_chunks = int((total_duration + chunk_duration - 1) / chunk_duration)
    print(f"Akan dibagi menjadi {num_chunks} bagian")
    
    # Dapatkan nama file dan ekstensi
    base_name = Path(input_file).stem
    extension = Path(input_file).suffix
    
    chunk_files = []
    
    for i in range(num_chunks):
        start_time = i * chunk_duration
        output_file = os.path.join(output_dir, f"{base_name}_part{i+1:03d}{extension}")
        
        # Buat command ffmpeg
        cmd = [
            'ffmpeg', '-i', input_file,
            '-ss', str(start_time),
            '-t', str(chunk_duration),
            '-c', 'copy',  # Copy codec untuk proses cepat
            '-y',  # Overwrite jika ada
            output_file
        ]
        
        print(f"Memproses bagian {i+1}/{num_chunks}...")
        subprocess.run(cmd, capture_output=True)
        chunk_files.append(output_file)
        
        # Cek ukuran file
        file_size = os.path.getsize(output_file)
        print(f"  -> {output_file} ({file_size/1024/1024:.1f} MB)")
    
    return chunk_files

def merge_srt_files(srt_files, output_file="merged.srt"):
    """Menggabungkan beberapa file SRT menjadi satu"""
    merged_content = []
    subtitle_counter = 1
    time_offset = 0
    
    for i, srt_file in enumerate(srt_files):
        if not os.path.exists(srt_file):
            print(f"Warning: {srt_file} tidak ditemukan, skip...")
            continue
            
        with open(srt_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            
        if not content:
            continue
            
        # Parse SRT content
        blocks = content.split('\n\n')
        
        for block in blocks:
            lines = block.strip().split('\n')
            if len(lines) >= 3:
                # Update nomor subtitle
                lines[0] = str(subtitle_counter)
                
                # Parse dan update timestamp jika bukan file pertama
                if i > 0:
                    timestamp_line = lines[1]
                    start_time, end_time = timestamp_line.split(' --> ')
                    
                    # Tambahkan offset waktu
                    start_time = add_time_offset(start_time, time_offset)
                    end_time = add_time_offset(end_time, time_offset)
                    
                    lines[1] = f"{start_time} --> {end_time}"
                
                merged_content.append('\n'.join(lines))
                subtitle_counter += 1
        
        # Update time offset untuk file berikutnya
        if i < len(srt_files) - 1:
            time_offset += 300  # 5 menit dalam detik
    
    # Tulis hasil merge
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(merged_content) + '\n')
    
    print(f"SRT files berhasil digabung ke: {output_file}")

def add_time_offset(time_str, offset_seconds):
    """Menambahkan offset waktu ke timestamp SRT"""
    # Parse time format HH:MM:SS,mmm
    hours, minutes, rest = time_str.split(':')
    seconds, milliseconds = rest.split(',')
    
    total_seconds = int(hours) * 3600 + int(minutes) * 60 + int(seconds) + offset_seconds
    
    new_hours = total_seconds // 3600
    new_minutes = (total_seconds % 3600) // 60
    new_seconds = total_seconds % 60
    
    return f"{new_hours:02d}:{new_minutes:02d}:{new_seconds:02d},{milliseconds}"

def main():
    parser = argparse.ArgumentParser(description="Split audio file untuk mengatasi batas ukuran API")
    parser.add_argument("--input", help="File audio input yang akan di-split")
    parser.add_argument("--duration", type=int, default=300, help="Durasi per bagian dalam detik (default: 300 = 5 menit)")
    parser.add_argument("--output-dir", default="chunks", help="Direktori output untuk menyimpan potongan audio")
    parser.add_argument("--merge-srt", action="store_true", help="Gabungkan file SRT dari direktori chunks")
    parser.add_argument("--srt-output", default="merged.srt", help="Nama file output untuk SRT yang digabung")
    
    args = parser.parse_args()
    
    # Mode merge SRT
    if args.merge_srt:
        srt_files = sorted(Path(args.output_dir).glob("*.srt"))
        if not srt_files:
            print(f"Tidak ada file SRT ditemukan di {args.output_dir}")
            return
        
        print(f"Ditemukan {len(srt_files)} file SRT:")
        for srt in srt_files:
            print(f"  - {srt}")
        
        merge_srt_files([str(srt) for srt in srt_files], args.srt_output)
        return
    
    # Mode split audio
    if not args.input:
        parser.error("--input diperlukan untuk mode split audio")
    
    # Cek ffmpeg
    if not check_ffmpeg():
        print("Error: ffmpeg tidak terdeteksi!")
        print("Silakan install ffmpeg terlebih dahulu:")
        print("- Windows: Download dari https://ffmpeg.org/download.html")
        print("- Mac: brew install ffmpeg")
        print("- Linux: sudo apt install ffmpeg")
        return
    
    if not os.path.exists(args.input):
        print(f"Error: File {args.input} tidak ditemukan!")
        return
    
    print(f"Splitting {args.input}...")
    chunk_files = split_audio(args.input, args.duration, args.output_dir)
    
    if chunk_files:
        print(f"\nBerhasil membuat {len(chunk_files)} file:")
        for chunk in chunk_files:
            print(f"  - {chunk}")
        
        print("\nSekarang jalankan whisper.py untuk setiap file:")
        for i, chunk in enumerate(chunk_files):
            srt_name = chunk.replace(Path(chunk).suffix, '.srt')
            print(f"python whisper.py --input {chunk} --output {srt_name} --model gpt-4o")
        
        print("\nSetelah semua selesai, gabungkan SRT dengan:")
        print(f"python split_audio.py --merge-srt --output-dir {args.output_dir}")

if __name__ == "__main__":
    main()