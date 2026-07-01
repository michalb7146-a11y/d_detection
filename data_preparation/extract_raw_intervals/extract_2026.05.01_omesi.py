import os
import glob
import pandas as pd
import librosa
import soundfile as sf
from tqdm import tqdm

AUDIO_DIR = r"/Users/deviceone/Documents/data/2026.04.28_omesi/2026.04.28_omesi"
CSV_DIR = r"/Users/deviceone/Documents/data/2026.04.28_omesi/tagged_2026.04.28_omesi"
OUTPUT_DIR = r"/Users/deviceone/Documents/data/2026.04.28_omesi/raw_extracted_segments" 

# AUDIO_DIR = r"/Users/deviceone/Documents/data/2026.05.01_omesi/2026.05.01_omesi"
# CSV_DIR = r"/Users/deviceone/Documents/data/2026.05.01_omesi/tagged_2026.05.01_omesi"
# OUTPUT_DIR = r"/Users/deviceone/Documents/data/2026.05.01_omesi/raw_extracted_segments" 

SR = 16000 

DRONE_OUT = os.path.join(OUTPUT_DIR, "raw_drone")
BG_OUT = os.path.join(OUTPUT_DIR, "raw_background")
os.makedirs(DRONE_OUT, exist_ok=True)
os.makedirs(BG_OUT, exist_ok=True)

# מילות מפתח מוגדרות לסינון התיוגים
VALID_DRONE_KEYWORDS = ['drone', 'רחפן', 'fly', 'flies', 'מעוף']

def time_str_to_seconds(time_str):
    if pd.isna(time_str):
        return None
    try:
        time_str = str(time_str).strip()
        parts = time_str.split(':')
        if len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])
        elif len(parts) == 3:
            return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
        return float(time_str)
    except Exception:
        return None

def parse_tsv_file(file_path):
    """
    מחלץ אינטרוולים של רחפנים מאושרים ואינטרוולים של אזורים שיש להתעלם מהם (Unclear, Car, וכו')
    """
    drone_intervals = []
    ignore_intervals = []
    
    try:
        # קריאה אוטומטית לפי מפריד (תומך ב-CSV וב-TSV)
        df = pd.read_csv(file_path, sep=None, engine='python')
        df.columns = [col.strip() for col in df.columns]
        
        # איתור עמודת הלייבל והזמנים
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
                
                # 1. סינון מחמיר ל-Unclear - נכנס מיד לרשימת התעלמות
                if 'unclear' in current_label:
                    ignore_intervals.append((start_sec, end_sec))
                    continue
                
                # 2. בדיקה אם זו אחת ממילות המפתח המאושרות לרחפן
                is_drone = any(keyword in current_label for keyword in VALID_DRONE_KEYWORDS)
                if is_drone:
                    drone_intervals.append((start_sec, end_sec))
                else:
                    # כל רעש ספציפי אחר (כמו car) נכנס לאזורים אסורים ולא ישמש כרקע נקי
                    ignore_intervals.append((start_sec, end_sec))
                    
    except Exception as e:
        print(f"Error reading metadata file {file_path}: {e}")
        
    return drone_intervals, ignore_intervals

def save_full_segment(y, sr, start_time, end_time, output_folder, base_name, index):
    """שומר את המקטע המלא כפי שהוא, ללא הגבלת אורך קבועה"""
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)
    if start_sample >= len(y) or start_sample == end_sample:
        return
    chunk = y[start_sample:end_sample]
    if len(chunk) == 0:
        return
    out_filename = f"{base_name}_seg_{index}.wav"
    sf.write(os.path.join(output_folder, out_filename), chunk, sr)

def process_dataset():
    csv_files = glob.glob(os.path.join(CSV_DIR, "*.csv")) + glob.glob(os.path.join(CSV_DIR, "*.tsv"))
    print(f"Found {len(csv_files)} metadata files to process.")

    for csv_path in csv_files:
        folder_name = os.path.splitext(os.path.basename(csv_path))[0]
        target_audio_folder = os.path.join(AUDIO_DIR, folder_name)
        
        if not os.path.exists(target_audio_folder):
            continue
            
        wav_files = glob.glob(os.path.join(target_audio_folder, "*.wav"))
        if not wav_files:
            continue
            
        drone_intervals, ignore_intervals = [], []
        if os.path.getsize(csv_path) > 0:
            drone_intervals, ignore_intervals = parse_tsv_file(csv_path)
            
        print(f"Extracting robust segments from folder '{folder_name}'...")

        for wav_path in tqdm(wav_files):
            base_name = os.path.splitext(os.path.basename(wav_path))[0]
            try:
                y, sr = librosa.load(wav_path, sr=SR, mono=True)
                total_duration = librosa.get_duration(y=y, sr=sr)
            except Exception as e:
                print(f"Error loading {wav_path}: {e}")
                continue

            chunk_idx = 0
            
            # תרחיש 1: אין רחפנים ואין אזורים חסומים - כל הקובץ הוא רקע מלא ונקי
            if not drone_intervals and not ignore_intervals:
                save_full_segment(y, sr, 0.0, total_duration, BG_OUT, f"{folder_name}_{base_name}", chunk_idx)
            else:
                # תרחיש 2: חילוץ גושי הרחפנים המאושרים בלבד
                for start, end in drone_intervals:
                    start_curr = min(start, total_duration)
                    end_curr = min(end, total_duration)
                    save_full_segment(y, sr, start_curr, end_curr, DRONE_OUT, f"{folder_name}_{base_name}", chunk_idx)
                    chunk_idx += 1
                
                # יצירת מפת אזורים אסורים משולבת (רחפנים + unclear + רעשים אחרים)
                all_forbidden_zones = drone_intervals + ignore_intervals
                all_forbidden_zones.sort()
                
                # חילוץ גושי רקע נקיים לחלוטין מתוך האזורים שלא נחסמו
                bg_start = 0.0
                for start, end in all_forbidden_zones:
                    start_curr = min(start, total_duration)
                    if start_curr > bg_start:
                        save_full_segment(y, sr, bg_start, start_curr, BG_OUT, f"{folder_name}_{base_name}", chunk_idx)
                        chunk_idx += 1
                    bg_start = max(bg_start, end)
                
                if bg_start < total_duration:
                    save_full_segment(y, sr, bg_start, total_duration, BG_OUT, f"{folder_name}_{base_name}", chunk_idx)

    print(f"\n--- Stage 1 Finished! Cleaned segments saved to {OUTPUT_DIR} ---")

if __name__ == "__main__":
    process_dataset()