#!/usr/bin/env python3
"""
Script untuk test dan compare kedua metode translasi
"""
import subprocess
import sys
import os
from datetime import datetime

def run_whisper(input_file, output_file, method, model="gpt-3.5-turbo", batch_size=5):
    """Jalankan whisper.py dengan parameter tertentu"""
    cmd = [
        sys.executable, "whisper.py",
        "--input", input_file,
        "--output", output_file,
        "--method", method,
        "--model", model
    ]
    
    if method == "translate":
        cmd.extend(["--batch-size", str(batch_size)])
    
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    start_time = datetime.now()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            elapsed = (datetime.now() - start_time).total_seconds()
            print(f"✅ Success! Elapsed time: {elapsed:.1f} seconds")
            return True, elapsed
        else:
            print(f"❌ Error: {result.stderr}")
            return False, 0
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False, 0

def compare_results(file1, file2):
    """Compare dua file SRT"""
    if not os.path.exists(file1) or not os.path.exists(file2):
        print("Cannot compare - one or both files missing")
        return
    
    with open(file1, 'r', encoding='utf-8') as f1:
        content1 = f1.read()
    
    with open(file2, 'r', encoding='utf-8') as f2:
        content2 = f2.read()
    
    lines1 = len(content1.split('\n'))
    lines2 = len(content2.split('\n'))
    
    print(f"\nFile comparison:")
    print(f"- {file1}: {lines1} lines, {len(content1)} chars")
    print(f"- {file2}: {lines2} lines, {len(content2)} chars")
    
    # Show first few subtitles from each
    print(f"\nFirst subtitle from {file1}:")
    print(content1.split('\n\n')[0] if '\n\n' in content1 else content1[:200])
    
    print(f"\nFirst subtitle from {file2}:")
    print(content2.split('\n\n')[0] if '\n\n' in content2 else content2[:200])

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_methods.py <input_audio> [model] [batch_size]")
        print("Example: python test_methods.py audio.wav gpt-4o 5")
        sys.exit(1)
    
    input_file = sys.argv[1]
    model = sys.argv[2] if len(sys.argv) > 2 else "gpt-3.5-turbo"
    batch_size = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found!")
        sys.exit(1)
    
    base_name = os.path.splitext(input_file)[0]
    
    # Test both methods
    print(f"Testing both methods with {input_file}")
    print(f"Model: {model}")
    print(f"Batch size (for translate method): {batch_size}")
    
    # Method 1: Transcribe
    output1 = f"{base_name}_transcribe.srt"
    success1, time1 = run_whisper(input_file, output1, "transcribe", model)
    
    # Method 2: Translate
    output2 = f"{base_name}_translate.srt"
    success2, time2 = run_whisper(input_file, output2, "translate", model, batch_size)
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    if success1:
        print(f"✅ Transcribe method: {time1:.1f}s → {output1}")
    else:
        print(f"❌ Transcribe method: FAILED")
    
    if success2:
        print(f"✅ Translate method: {time2:.1f}s → {output2}")
    else:
        print(f"❌ Translate method: FAILED")
    
    if success1 and success2:
        if time2 < time1:
            speedup = ((time1 - time2) / time1) * 100
            print(f"\n🚀 Translate method is {speedup:.1f}% faster!")
        else:
            slowdown = ((time2 - time1) / time2) * 100
            print(f"\n🐌 Translate method is {slowdown:.1f}% slower")
        
        compare_results(output1, output2)

if __name__ == "__main__":
    main()