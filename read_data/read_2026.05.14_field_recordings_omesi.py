import os
import glob
import pandas as pd
import librosa
import soundfile as sf
from tqdm import tqdm

AUDIO_DIR = r"/Users/deviceone/Downloads/data/2026.05.14_field_recordings_omesi/audio"
CSV_DIR = r"/Users/deviceone/Downloads/data/2026.05.14_field_recordings_omesi/tagged_2026.05.14_field_recordings_omesi"
OUTPUT_DIR = r"/Users/deviceone/Downloads/data/2026.05.14_field_recordings_omesi/new_balanced_2s_dataset_2026.05.14"

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

def slice_and_save(y, sr, start_time, end_time, output_folder, base_name, index):
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)
    
    if start_sample >= len(y) or start_sample == end_sample:
        return
    
    chunk = y[start_sample:end_sample]
    if len(chunk) < int(SEGMENT_DURATION * sr):
        return 
        
    out_filename = f"{base_name}_part_{index}.wav"
    out_path = os.path.join(output_folder, out_filename)
    sf.write(out_path, chunk, sr)

def process_dataset():
    # מוצא את כל קבצי ה-CSV או ה-TSV בתיקיית התיוגים השטוחה
    metadata_files = glob.glob(os.path.join(CSV_DIR, "*.csv")) + glob.glob(os.path.join(CSV_DIR, "*.tsv"))
    print(f"Found {len(metadata_files)} metadata files to process.")

    for meta_path in tqdm(metadata_files, desc="Processing files"):
        # חילוץ שם הקובץ הבסיסי (ללא הסיומת של ה-CSV)
        base_name = os.path.splitext(os.path.basename(meta_path))[0]
        
        # --- שינוי 1: חיפוש קובץ flac במקום קובץ wav ---
        audio_path = os.path.join(AUDIO_DIR, f"{base_name}.flac")
        
        # אם אין קובץ אודיו תואם, נדלג
        if not os.path.exists(audio_path):
            print(f"\nWarning: Audio file '{base_name}.flac' not found for metadata. Skipping.")
            continue
            
        # חילוץ זמני הרחפנים
        drone_intervals = []
        if os.path.getsize(meta_path) > 0:
            drone_intervals = parse_tsv_file(meta_path)
            
        try:
            # --- שינוי 2: טעינת ה-flac (librosa יודעת לקרוא flac באופן טבעי) ---
            y, sr = librosa.load(audio_path, sr=SR, mono=True)
            total_duration = librosa.get_duration(y=y, sr=sr)
        except Exception as e:
            print(f"\nError loading {audio_path}: {e}")
            continue

        chunk_idx = 0
        
        # תרחיש 1: אין רחפנים מתועדים בקובץ זה
        if not drone_intervals:
            current_time = 0.0
            while current_time + SEGMENT_DURATION <= total_duration:
                slice_and_save(y, sr, current_time, current_time + SEGMENT_DURATION, BG_OUT, base_name, chunk_idx)
                current_time += SEGMENT_DURATION
                chunk_idx += 1
        else:
            # תרחיש 2: יש רחפנים - חותכים את מקטעי הרחפנים
            for start, end in drone_intervals:
                start_curr = min(start, total_duration)
                end_curr = min(end, total_duration)
                
                current_time = start_curr
                while current_time + SEGMENT_DURATION <= end_curr:
                    slice_and_save(y, sr, current_time, current_time + SEGMENT_DURATION, DRONE_OUT, base_name, chunk_idx)
                    current_time += SEGMENT_DURATION
                    chunk_idx += 1
            
            # חיתוך מקטעי הרקע (הסרת זמני הרחפנים)
            drone_intervals.sort()
            bg_start = 0.0
            
            for start, end in drone_intervals:
                start_curr = min(start, total_duration)
                current_time = bg_start
                while current_time + SEGMENT_DURATION <= start_curr:
                    slice_and_save(y, sr, current_time, current_time + SEGMENT_DURATION, BG_OUT, base_name, chunk_idx)
                    current_time += SEGMENT_DURATION
                    chunk_idx += 1
                bg_start = max(bg_start, end)
            
            bg_start_curr = min(bg_start, total_duration)
            current_time = bg_start_curr
            while current_time + SEGMENT_DURATION <= total_duration:
                slice_and_save(y, sr, current_time, current_time + SEGMENT_DURATION, BG_OUT, base_name, chunk_idx)
                current_time += SEGMENT_DURATION
                chunk_idx += 1

    print("\n--- Data Preparation Finished! ---")
    print(f"Drone segments saved to: {DRONE_OUT}")
    print(f"Background segments saved to: {BG_OUT}")

if __name__ == "__main__":
    process_dataset()