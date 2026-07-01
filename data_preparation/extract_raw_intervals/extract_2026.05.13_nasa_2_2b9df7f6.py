import os
import glob
import pandas as pd
import librosa
import soundfile as sf
from tqdm import tqdm

# --- הגדרות נתיבים מעודכנות ---
# AUDIO_DIR = r"/Users/deviceone/Documents/data/nasa_2/2026.05.13_nasa_2_2b9df7f6"
# CSV_DIR = r"/Users/deviceone/Documents/data/nasa_2/tagged_2026.05.13_nasa_2_2b9df7f6"
# OUTPUT_DIR = r"/Users/deviceone/Documents/data/nasa_2/raw_extracted_segments_nasa_2" 

# AUDIO_DIR = r"/Users/deviceone/Documents/data/tut/tut_audio"
# CSV_DIR = r""
# OUTPUT_DIR = r"/Users/deviceone/Documents/data/tut/raw_extracted_segments_tut" 

# AUDIO_DIR = r"/Users/deviceone/Documents/data/2026.06.01_fang/audio"
# CSV_DIR = r"/Users/deviceone/Documents/data/2026.06.01_fang/tagged_2026.06.01_fang"
# OUTPUT_DIR = r"/Users/deviceone/Documents/data/tut/raw_extracted_segments" 

# AUDIO_DIR = r"/Users/deviceone/Documents/data/2026.06.07_manatees/audio"
# CSV_DIR = r"/Users/deviceone/Documents/data/2026.06.07_manatees/tagged_2026.06.07_manatees"
# OUTPUT_DIR = r"/Users/deviceone/Documents/data/2026.06.07_manatees/raw_extracted_segments" 

# AUDIO_DIR = r"/Users/deviceone/Documents/data/2026.06.17_swan/audio"
# CSV_DIR = r"/Users/deviceone/Documents/data/2026.06.17_swan/tagged_2026.06.17_swan"
# OUTPUT_DIR = r"/Users/deviceone/Documents/data/2026.06.17_swan/raw_extracted_segments"

AUDIO_DIR = r"/Users/deviceone/Documents/data/2026.06.09_kakadoo/audio/Sensor_1"
CSV_DIR = r"/Users/deviceone/Documents/data/2026.06.09_kakadoo/tagged_2026.06.09_kakadoo/Sensor_1"
OUTPUT_DIR = r"/Users/deviceone/Documents/data/2026.06.09_kakadoo/raw_extracted_segments"

SR = 16000 

DRONE_OUT = os.path.join(OUTPUT_DIR, "raw_drone")
BG_OUT = os.path.join(OUTPUT_DIR, "raw_background")
os.makedirs(DRONE_OUT, exist_ok=True)
os.makedirs(BG_OUT, exist_ok=True)

# רשימת מילות המפתח המאושרות לרחפנים
VALID_DRONE_KEYWORDS = ['drone', 'רחפן', 'fly', 'flies', 'מעוף']

def time_str_to_seconds(time_str):
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

def process_tsv_file(file_path):
    """
    מחלץ אינטרוולים של רחפנים קשיחים ואינטרוולים שיש להתעלם מהם (Unclear)
    """
    drone_intervals = []
    ignore_intervals = []
    
    try:
        df = pd.read_csv(file_path, sep=None, engine='python')
        df.columns = [col.strip() for col in df.columns]
        
        label_col = next((c for c in df.columns if 'name' in c.lower() or 'type' in c.lower() or 'label' in c.lower()), df.columns[0])
        start_col = next((c for c in df.columns if 'start' in c.lower()), None)
        duration_col = next((c for c in df.columns if 'duration' in c.lower()), None)
        
        if start_col and duration_col:
            for _, row in df.iterrows():
                current_label = str(row[label_col]).strip().lower() if not pd.isna(row[label_col]) else ""
                
                start_sec = time_str_to_seconds(row[start_col])
                dur_sec = time_str_to_seconds(row[duration_col])
                if start_sec is None:
                    continue
                if dur_sec is None or dur_sec <= 0:
                    dur_sec = 4.0
                end_sec = start_sec + dur_sec
                
                # 1. סינון קשוח ל-Unclear - נכנס לרשימת התעלמות
                if 'unclear' in current_label:
                    ignore_intervals.append((start_sec, end_sec))
                    continue
                
                # 2. בדיקה אם מדובר ברחפן מאושר
                is_drone = any(keyword in current_label for keyword in VALID_DRONE_KEYWORDS)
                if is_drone:
                    drone_intervals.append((start_sec, end_sec))
                else:
                    # כל רעש מוגדר אחר (כמו car) נחשב כרעש שחוסם רקע (לא ייגזר כרקע נקי רגיל)
                    ignore_intervals.append((start_sec, end_sec))
                    
    except Exception as e:
        print(f"Error reading metadata file {file_path}: {e}")
        
    return drone_intervals, ignore_intervals

def process_dataset():
    metadata_files = glob.glob(os.path.join(CSV_DIR, "*.csv")) + glob.glob(os.path.join(CSV_DIR, "*.tsv"))
    print(f"Found {len(metadata_files)} metadata files to process.")

    for meta_path in tqdm(metadata_files, desc="Extracting cleaned datasets"):
        base_name = os.path.splitext(os.path.basename(meta_path))[0]
        wav_path = os.path.join(AUDIO_DIR, f"{base_name}.wav")
        
        if not os.path.exists(wav_path):
            continue
            
        drone_intervals, ignore_intervals = [], []
        if os.path.getsize(meta_path) > 0:
            drone_intervals, ignore_intervals = process_tsv_file(meta_path)
            
        try:
            y, sr = librosa.load(wav_path, sr=SR, mono=True)
            total_duration = librosa.get_duration(y=y, sr=sr)
        except Exception as e:
            print(f"\nError loading {wav_path}: {e}")
            continue

        chunk_idx = 0
        
        # 1. שמירת מקטעי הרחפנים המאושרים
        for start, end in drone_intervals:
            save_full_segment(y, sr, min(start, total_duration), min(end, total_duration), DRONE_OUT, base_name, chunk_idx)
            chunk_idx += 1
            
        # 2. בניית מפת הגנות מפני חיתוך רקע (כל מה שרחפן או Unclear או Car אסור שיהפוך לרקע)
        all_forbidden_zones = drone_intervals + ignore_intervals
        all_forbidden_zones.sort()
        
        # בניית מקטעי רקע נקיים לחלוטין מתוך האזורים המותרים
        bg_start = 0.0
        for start, end in all_forbidden_zones:
            start_curr = min(start, total_duration)
            if start_curr > bg_start:
                # גזירה רק אם המרווח גדול מספיק
                save_full_segment(y, sr, bg_start, start_curr, BG_OUT, base_name, chunk_idx)
                chunk_idx += 1
            bg_start = max(bg_start, end)
            
        if bg_start < total_duration:
            save_full_segment(y, sr, bg_start, total_duration, BG_OUT, base_name, chunk_idx)

    print(f"\n--- Process Finished! Cleaned segments saved to: {OUTPUT_DIR} ---")

def save_full_segment(y, sr, start_time, end_time, output_folder, base_name, index):
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)
    if start_sample >= len(y) or start_sample == end_sample:
        return
    chunk = y[start_sample:end_sample]
    if len(chunk) == 0:
        return 
    out_filename = f"{base_name}_seg_{index}.wav"
    out_path = os.path.join(output_folder, out_filename)
    sf.write(out_path, chunk, sr)

if __name__ == "__main__":
    process_dataset()