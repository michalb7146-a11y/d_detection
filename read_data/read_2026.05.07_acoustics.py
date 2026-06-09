import os
import glob
import pandas as pd
import librosa
import soundfile as sf
from tqdm import tqdm

# --- הגדרות נתיבים חדשות ---
# AUDIO_DIR מכילה תתי-תיקיות ובתוכן קבצי WAV
AUDIO_DIR = r"/Users/deviceone/Downloads/data/2026.05.07_acoustics/audio"
# CSV_DIR מכילה תתי-תיקיות עם שמות זהים/דומים ובתוכן קבצי CSV
CSV_DIR = r"/Users/deviceone/Downloads/data/2026.05.07_acoustics/tagged_2026.05.07_acoustics"
OUTPUT_DIR = r"/Users/deviceone/Downloads/data/2026.05.07_acoustics/new_balanced_2s_dataset_2026.05.07_acoustics"

SEGMENT_DURATION = 2.0 
SR = 16000 

DRONE_OUT = os.path.join(OUTPUT_DIR, "target_drone")
BG_OUT = os.path.join(OUTPUT_DIR, "background")
os.makedirs(DRONE_OUT, exist_ok=True)
os.makedirs(BG_OUT, exist_ok=True)

def time_str_to_seconds(time_str):
    """הופך פורמט של MM:SS.mmm או HH:MM:SS.mmm לשניות"""
    if pd.isna(time_str):
        return None
    try:
        time_str = str(time_str).strip()
        parts = time_str.split(':')
        if len(parts) == 2:  # MM:SS.mmm
            return float(parts[0]) * 60 + float(parts[1])
        elif len(parts) == 3:  # HH:MM:SS.mmm
            return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
        return float(time_str)
    except Exception:
        return None

def parse_tsv_file(file_path):
    """קריאת קובץ ה-TSV/CSV המופרד בטאבים והחזרת רשימת זמנים"""
    drone_intervals = []
    try:
        df = pd.read_csv(file_path, sep=None, engine='python')
        df.columns = [col.strip() for col in df.columns]
        
        start_col = next((c for c in df.columns if 'start' in c.lower()), None)
        duration_col = next((c for c in df.columns if 'duration' in c.lower()), None)
        
        if start_col and duration_col:
            for _, row in df.iterrows():
                start_sec = time_str_to_seconds(row[start_col])
                dur_sec = time_str_to_seconds(row[duration_col])
                
                if start_sec is not None:
                    if dur_sec is None or dur_sec <= 0:
                        dur_sec = 4.0  # ברירת מחדל
                    drone_intervals.append((start_sec, start_sec + dur_sec))
    except Exception as e:
        print(f"Error reading metadata file {file_path}: {e}")
        
    return drone_intervals

def slice_and_save(y, sr, start_time, end_time, output_folder, base_name, folder_name, index):
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)
    
    if start_sample >= len(y) or start_sample == end_sample:
        return
    
    chunk = y[start_sample:end_sample]
    if len(chunk) < int(SEGMENT_DURATION * sr):
        return 
        
    # הוספת שם תיקיית המקור לשם הקובץ השמור כדי למנוע דריסת קבצים בעלי שם זהה מתיקיות שונות
    out_filename = f"{folder_name}_{base_name}_part_{index}.wav"
    out_path = os.path.join(output_folder, out_filename)
    sf.write(out_path, chunk, sr)

def process_dataset():
    # סריקת כל תתי-התיקיות בתוך תיקיית ה-CSV
    csv_subfolders = [f for f in os.listdir(CSV_DIR) if os.path.isdir(os.path.join(CSV_DIR, f))]
    print(f"Found {len(csv_subfolders)} subfolders in CSV_DIR to process.")

    for subfolder in csv_subfolders:
        current_csv_dir = os.path.join(CSV_DIR, subfolder)
        current_audio_dir = os.path.join(AUDIO_DIR, subfolder)
        
        # בדיקה האם קיימת תיקיית אודיו תואמת באותו השם
        if not os.path.exists(current_audio_dir):
            print(f"\nWarning: Audio subfolder '{subfolder}' not found. Skipping this folder.")
            continue
            
        # מציאת קבצי ה-CSV/TSV בתוך תת-התיקייה הנוכחית
        metadata_files = glob.glob(os.path.join(current_csv_dir, "*.csv")) + glob.glob(os.path.join(current_csv_dir, "*.tsv"))
        if not metadata_files:
            continue
            
        print(f"\nProcessing subfolder: {subfolder} ({len(metadata_files)} metadata files)")
        
        for meta_path in tqdm(metadata_files, desc=f"Folder {subfolder}"):
            base_name = os.path.splitext(os.path.basename(meta_path))[0]
            
            # חיפוש קובץ ה-WAV בתוך תת-התיקייה הנוכחית באודיו
            wav_path = os.path.join(current_audio_dir, f"{base_name}.wav")
            
            if not os.path.exists(wav_path):
                print(f"\nWarning: Audio file '{base_name}.wav' not found in '{subfolder}'. Skipping.")
                continue
                
            drone_intervals = []
            if os.path.getsize(meta_path) > 0:
                drone_intervals = parse_tsv_file(meta_path)
                
            try:
                y, sr = librosa.load(wav_path, sr=SR, mono=True)
                total_duration = librosa.get_duration(y=y, sr=sr)
            except Exception as e:
                print(f"\nError loading {wav_path}: {e}")
                continue

            chunk_idx = 0
            
            # תרחיש 1: אין רחפנים מתועדים
            if not drone_intervals:
                current_time = 0.0
                while current_time + SEGMENT_DURATION <= total_duration:
                    slice_and_save(y, sr, current_time, current_time + SEGMENT_DURATION, BG_OUT, base_name, subfolder, chunk_idx)
                    current_time += SEGMENT_DURATION
                    chunk_idx += 1
            else:
                # תרחיש 2: יש רחפנים
                for start, end in drone_intervals:
                    start_curr = min(start, total_duration)
                    end_curr = min(end, total_duration)
                    
                    current_time = start_curr
                    while current_time + SEGMENT_DURATION <= end_curr:
                        slice_and_save(y, sr, current_time, current_time + SEGMENT_DURATION, DRONE_OUT, base_name, subfolder, chunk_idx)
                        current_time += SEGMENT_DURATION
                        chunk_idx += 1
                
                drone_intervals.sort()
                bg_start = 0.0
                
                for start, end in drone_intervals:
                    start_curr = min(start, total_duration)
                    current_time = bg_start
                    while current_time + SEGMENT_DURATION <= start_curr:
                        slice_and_save(y, sr, current_time, current_time + SEGMENT_DURATION, BG_OUT, base_name, subfolder, chunk_idx)
                        current_time += SEGMENT_DURATION
                        chunk_idx += 1
                    bg_start = max(bg_start, end)
                
                bg_start_curr = min(bg_start, total_duration)
                current_time = bg_start_curr
                while current_time + SEGMENT_DURATION <= total_duration:
                    slice_and_save(y, sr, current_time, current_time + SEGMENT_DURATION, BG_OUT, base_name, subfolder, chunk_idx)
                    current_time += SEGMENT_DURATION
                    chunk_idx += 1

    print("\n--- Data Preparation Finished! ---")
    print(f"Drone segments saved to: {DRONE_OUT}")
    print(f"Background segments saved to: {BG_OUT}")

if __name__ == "__main__":
    process_dataset()