"""
Reorganize WL-BISINDO Dataset
==============================
Script untuk mengorganisasi dataset WL-BISINDO dari struktur flat
ke struktur folder per kelas.

Format nama file asli: signer{X}_label{Y}_sample{Z}.mp4
Struktur output: {class_name}/signer{X}_label{Y}_sample{Z}.mp4

Author: BISINDO Project
"""

import os
import shutil
from pathlib import Path
import re
import argparse

# Mapping label number ke nama kelas (dari gambar sebelumnya)
LABEL_TO_CLASS = {
    0: "Air",
    1: "Belajar",
    2: "Cari",
    3: "Hari",
    4: "Ingat",
    5: "Lagi",
    6: "Maaf",
    7: "Makan",
    8: "Motor",
    9: "Saya",
    10: "Terima_Kasih",
    11: "Tuli",
    12: "Apa",
    13: "Siapa",
    14: "Kapan",
    15: "Di_Mana",
    16: "Mengapa",
    17: "Bagaimana",
    18: "Merah",
    19: "Kuning",
    20: "Hijau",
    21: "Hitam",
    22: "Dengar",
    23: "Berangkat",
    24: "Datang",
    25: "Teman",
    26: "Keluarga",
    27: "Rumah",
    28: "Pagi",
    29: "Siang",
    30: "Sore",
    31: "Malam",
}


def extract_label_from_filename(filename):
    """
    Extract label number dari nama file.
    Format: signer{X}_label{Y}_sample{Z}.mp4
    
    Returns:
        int: Label number, atau None jika tidak ditemukan
    """
    match = re.search(r'label(\d+)', filename)
    if match:
        return int(match.group(1))
    return None


def reorganize_dataset(input_dir, output_dir, copy=True):
    """
    Reorganisasi dataset dari flat ke folder per kelas.
    
    Args:
        input_dir: Folder berisi semua video (flat)
        output_dir: Folder output dengan struktur per kelas
        copy: True = copy file, False = move file
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Cari semua video
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV']
    videos = [f for f in input_path.iterdir() 
              if f.is_file() and f.suffix in video_extensions]
    
    print(f"Found {len(videos)} videos in {input_dir}")
    
    if len(videos) == 0:
        print("No videos found!")
        return
    
    # Organisasi per kelas
    stats = {}
    errors = []
    
    for video in videos:
        label = extract_label_from_filename(video.name)
        
        if label is None:
            errors.append(f"Cannot extract label from: {video.name}")
            continue
        
        if label not in LABEL_TO_CLASS:
            errors.append(f"Unknown label {label} in: {video.name}")
            continue
        
        class_name = LABEL_TO_CLASS[label]
        
        # Buat folder kelas jika belum ada
        class_folder = output_path / class_name
        class_folder.mkdir(parents=True, exist_ok=True)
        
        # Copy atau move file
        dest_file = class_folder / video.name
        
        if copy:
            shutil.copy2(video, dest_file)
        else:
            shutil.move(video, dest_file)
        
        # Update stats
        if class_name not in stats:
            stats[class_name] = 0
        stats[class_name] += 1
    
    # Print summary
    print("\n" + "=" * 50)
    print("REORGANIZATION COMPLETE")
    print("=" * 50)
    print(f"\nOutput directory: {output_path}")
    print(f"Total classes: {len(stats)}")
    print(f"Total videos processed: {sum(stats.values())}")
    
    print("\nPer-class breakdown:")
    for class_name in sorted(stats.keys()):
        print(f"  {class_name}: {stats[class_name]} videos")
    
    if errors:
        print(f"\nErrors ({len(errors)}):")
        for error in errors:
            print(f"  - {error}")
    
    print("\n" + "=" * 50)
    action = "copied" if copy else "moved"
    print(f"Files {action} to: {output_path}")
    print("You can now run extract_landmarks.py on this folder!")


def verify_structure(directory):
    """
    Verifikasi struktur folder sudah benar.
    """
    dir_path = Path(directory)
    
    if not dir_path.exists():
        print(f"Directory not found: {directory}")
        return
    
    print(f"\nVerifying structure of: {directory}")
    print("=" * 50)
    
    # Cek subfolder
    subfolders = [f for f in dir_path.iterdir() if f.is_dir()]
    
    if len(subfolders) == 0:
        print("No subfolders found! Structure is FLAT.")
        
        # Count videos
        videos = list(dir_path.glob("*.mp4")) + list(dir_path.glob("*.MP4"))
        print(f"Videos in root: {len(videos)}")
        
        if len(videos) > 0:
            print("\nSample filenames:")
            for v in videos[:5]:
                print(f"  - {v.name}")
    else:
        print(f"Found {len(subfolders)} class folders:")
        
        total_videos = 0
        for folder in sorted(subfolders):
            videos = list(folder.glob("*.mp4")) + list(folder.glob("*.MP4"))
            total_videos += len(videos)
            print(f"  {folder.name}: {len(videos)} videos")
        
        print(f"\nTotal videos: {total_videos}")
        print("Structure is CORRECT! ✓")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Reorganize WL-BISINDO dataset')
    
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Input directory (flat structure)')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Output directory (class folders)')
    parser.add_argument('--move', action='store_true',
                        help='Move files instead of copy (default: copy)')
    parser.add_argument('--verify', '-v', action='store_true',
                        help='Only verify structure, no reorganization')
    
    args = parser.parse_args()
    
    if args.verify:
        verify_structure(args.input)
    else:
        reorganize_dataset(
            input_dir=args.input,
            output_dir=args.output,
            copy=not args.move
        )