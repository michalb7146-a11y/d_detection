import os
import glob
import soundfile as sf
from tqdm import tqdm

DATA_DIRECTORIES = [
    r"/Users/deviceone/Documents/data/2026.04.28_omesi/raw_extracted_segments",
    r"/Users/deviceone/Documents/data/2026.05.01_omesi/raw_extracted_segments",
    # r"/Users/deviceone/Documents/data/dregon/slice_2s_overlap_dregon",
    r"/Users/deviceone/Documents/data/nasa_2/raw_extracted_segments_nasa_2",
    # r"/Users/deviceone/Documents/data/tut/slice_2s_overlap_tut",
    # r"/Users/deviceone/Documents/data/DNC/slice_2s_overlap_dnc",
    # r"/Users/deviceone/Documents/data/ESC-50/slice_2s_overlap_ESC-50"
]

def calculate_raw_dataset_hours():
    total_drone_seconds = 0.0
    total_bg_seconds = 0.0
    
    total_drone_files = 0
    total_bg_files = 0
    
    for base_path in DATA_DIRECTORIES:
        drone_path = os.path.join(base_path, "raw_drone")
        bg_path = os.path.join(base_path, "raw_background")
        
        # 1. סריקת רחפנים מקוריים (raw_drone)
        if os.path.exists(drone_path):
            drone_files = glob.glob(os.path.join(drone_path, "*.wav"))
            total_drone_files += len(drone_files)
            if drone_files:
                print(f"⏳ Calculating duration for raw drone files in {os.path.basename(base_path)}...")
                for f in tqdm(drone_files, desc="Raw Drone"):
                    try:
                        info = sf.info(f)
                        total_drone_seconds += info.duration
                    except Exception as e:
                        print(f"Error reading {f}: {e}")
            
        # 2. סריקת רקע מקורי (raw_background)
        if os.path.exists(bg_path):
            bg_files = glob.glob(os.path.join(bg_path, "*.wav"))
            total_bg_files += len(bg_files)
            if bg_files:
                print(f"⏳ Calculating duration for raw background files in {os.path.basename(base_path)}...")
                for f in tqdm(bg_files, desc="Raw Background"):
                    try:
                        info = sf.info(f)
                        total_bg_seconds += info.duration
                    except Exception as e:
                        print(f"Error reading {f}: {e}")

    # המרה לשעות
    drone_hours = total_drone_seconds / 3600
    bg_hours = total_bg_seconds / 3600
    
    print("\n==================================================")
    print("   📊 RAW VARIABLE DATASET DURATION REPORT        ")
    print("==================================================")
    print(f"🛸 RAW DRONE (raw_drone):")
    print(f"   • Total Audio Files: {total_drone_files:,}")
    print(f"   • Total Exact Duration: {drone_hours:.2f} hours (~{int(total_drone_seconds // 60)} minutes)")
    print(f"--------------------------------------------------")
    print(f"🍃 RAW BACKGROUND (raw_background):")
    print(f"   • Total Audio Files: {total_bg_files:,}")
    print(f"   • Total Exact Duration: {bg_hours:.2f} hours (~{int(total_bg_seconds // 60)} minutes)")
    print(f"==================================================")
    print(f"Total Raw Dataset Size: {drone_hours + bg_hours:.2f} hours.")

if __name__ == "__main__":
    calculate_raw_dataset_hours()