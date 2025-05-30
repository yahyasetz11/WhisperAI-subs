import os
import sys
import subprocess
from pathlib import Path

def get_file_info(filepath):
    """Mendapatkan informasi detail file audio"""
    if not os.path.exists(filepath):
        print(f"Error: File {filepath} tidak ditemukan!")
        return
    
    # Ukuran file
    file_size = os.path.getsize(filepath)
    size_mb = file_size / (1024 * 1024)
    
    print(f"\n📁 File: {filepath}")
    print(f"📊 Ukuran: {size_mb:.1f} MB ({file_size:,} bytes)")
    
    # Status untuk API
    if size_mb > 25:
        print(f"❌ TERLALU BESAR untuk Whisper API (max 25 MB)")
        print(f"   Perlu dikurangi: {size_mb - 25:.1f} MB")
    else:
        print(f"✅ OK untuk Whisper API")
    
    # Info audio menggunakan ffprobe jika tersedia
    try:
        # Durasi
        duration_cmd = ['ffprobe', '-v', 'error', '-show_entries', 
                       'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', 
                       filepath]
        duration = subprocess.check_output(duration_cmd, text=True).strip()
        duration_float = float(duration)
        minutes = duration_float / 60
        
        # Bitrate
        bitrate_cmd = ['ffprobe', '-v', 'error', '-show_entries', 
                      'format=bit_rate', '-of', 'default=noprint_wrappers=1:nokey=1', 
                      filepath]
        bitrate = subprocess.check_output(bitrate_cmd, text=True).strip()
        
        # Format
        format_cmd = ['ffprobe', '-v', 'error', '-show_entries', 
                     'stream=codec_name,sample_rate,channels', '-of', 'json', 
                     filepath]
        import json
        format_info = json.loads(subprocess.check_output(format_cmd, text=True))
        
        print(f"\n📼 Info Audio:")
        print(f"   Durasi: {minutes:.1f} menit ({duration_float:.0f} detik)")
        if bitrate:
            print(f"   Bitrate: {int(bitrate)/1000:.0f} kbps")
        
        if format_info.get('streams'):
            stream = format_info['streams'][0]
            print(f"   Format: {stream.get('codec_name', 'unknown')}")
            print(f"   Sample Rate: {stream.get('sample_rate', 'unknown')} Hz")
            print(f"   Channels: {stream.get('channels', 'unknown')}")
        
        # Rekomendasi
        print(f"\n💡 Rekomendasi:")
        if size_mb > 25:
            # Hitung target bitrate untuk 24 MB
            target_size_mb = 24
            target_bitrate = (target_size_mb * 8 * 1024) / (duration_float)  # kbps
            
            print(f"1. Kompres dengan ffmpeg:")
            print(f"   ffmpeg -i {filepath} -b:a {int(target_bitrate)}k output.mp3")
            
            print(f"\n2. Atau split menjadi {int(minutes/5) + 1} bagian:")
            print(f"   python split_audio.py --input {filepath}")
            
        else:
            print("   File sudah siap untuk diproses!")
            
    except subprocess.CalledProcessError:
        print("\n⚠️  ffmpeg/ffprobe tidak terdeteksi - tidak bisa mendapatkan info detail audio")
    except Exception as e:
        print(f"\n⚠️  Error mendapatkan info audio: {e}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python check_audio.py <audio_file>")
        sys.exit(1)
    
    filepath = sys.argv[1]
    get_file_info(filepath)

if __name__ == "__main__":
    main()