import os
import glob
import pandas as pd
import librosa
import soundfile as sf
from tqdm import tqdm

# --- הגדרות נתיבים ---
AUDIO_DIR = r"/Users/deviceone/Downloads/2026.05.01_omesi"
CSV_DIR = r"/Users/deviceone/Downloads/tagged_2026.05.01_omesi"         # התיקייה שבה נמצאים כל קבצי ה-csv
OUTPUT_DIR = r"/Users/deviceone/Downloads/new_balanced_2s_dataset" # לאן להציל את התוצאות

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
    """קריאת קובץ ה-TSV המופרד בטאבים והחזרת רשימת זמנים"""
    drone_intervals = []
    try:
        df = pd.read_csv(file_path, sep='\t')
        df.columns = [col.strip() for col in df.columns]
        
        start_col = next((c for c in df.columns if 'start' in c.lower()), None)
        duration_col = next((c for c in df.columns if 'duration' in c.lower()), None)
        
        if start_col and duration_col:
            for _, row in df.iterrows():
                start_sec = time_str_to_seconds(row[start_col])
                dur_sec = time_str_to_seconds(row[duration_col])
                
                if start_sec is not None:
                    if dur_sec is None or dur_sec <= 0:
                        dur_sec = 4.0  # ברירת מחדל אם ה-duration ריק או 0
                    drone_intervals.append((start_sec, start_sec + dur_sec))
    except Exception as e:
        print(f"Error reading TSV {file_path}: {e}")
        
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
    # מוצא את כל קבצי ה-CSV או ה-TSV בתיקיית ה-CSV האחת
    csv_files = glob.glob(os.path.join(CSV_DIR, "*.csv")) + glob.glob(os.path.join(CSV_DIR, "*.tsv"))
    print(f"Found {len(csv_files)} metadata files (TSV/CSV) to process.")

    for csv_path in csv_files:
        # שם התיקייה התואמת (למשל "Folder_A") מתוך שם הקובץ ("Folder_A.csv")
        folder_name = os.path.splitext(os.path.basename(csv_path))[0]
        target_audio_folder = os.path.join(AUDIO_DIR, folder_name)
        
        # אם אין תיקיית אודיו תואמת לשם ה-CSV, נדלג
        if not os.path.exists(target_audio_folder):
            print(f"Warning: Audio folder '{target_audio_folder}' not found for metadata '{os.path.basename(csv_path)}'. Skipping.")
            continue
            
        # מציאת כל קבצי ה-WAV בתוך אותה תת-תיקייה ספציפית
        wav_files = glob.glob(os.path.join(target_audio_folder, "*.wav"))
        if not wav_files:
            print(f"No WAV files found in {target_audio_folder}")
            continue
            
        # חילוץ זמני הרחפנים עבור התיקייה הזו (אם הקובץ ריק או לא קיים רחפן, הרשימה תהיה ריקה)
        drone_intervals = []
        if os.path.getsize(csv_path) > 0:
            drone_intervals = parse_tsv_file(csv_path)
            
        print(f"Processing folder '{folder_name}' with {len(wav_files)} files...")

        for wav_path in tqdm(wav_files):
            base_name = os.path.splitext(os.path.basename(wav_path))[0]
            
            try:
                y, sr = librosa.load(wav_path, sr=SR, mono=True)
                total_duration = librosa.get_duration(y=y, sr=sr)
            except Exception as e:
                print(f"Error loading {wav_path}: {e}")
                continue

            chunk_idx = 0
            
            # תרחיש 1: אין רחפנים מתועדים בתיקייה הזו בכלל (קובץ ריק או ללא אינטרוולים)
            if not drone_intervals:
                current_time = 0.0
                while current_time + SEGMENT_DURATION <= total_duration:
                    slice_and_save(y, sr, current_time, current_time + SEGMENT_DURATION, BG_OUT, f"{folder_name}_{base_name}", chunk_idx)
                    current_time += SEGMENT_DURATION
                    chunk_idx += 1
            else:
                # תרחיש 2: יש רחפנים - חותכים את מקטעי הרחפנים
                for start, end in drone_intervals:
                    # הגנה למקרה שזמני הרחפן ב-CSV ארוכים יותר מאורך קובץ הסאונד הנוכחי
                    start_curr = min(start, total_duration)
                    end_curr = min(end, total_duration)
                    
                    current_time = start_curr
                    while current_time + SEGMENT_DURATION <= end_curr:
                        slice_and_save(y, sr, current_time, current_time + SEGMENT_DURATION, DRONE_OUT, f"{folder_name}_{base_name}", chunk_idx)
                        current_time += SEGMENT_DURATION
                        chunk_idx += 1
                
                # חיתוך מקטעי הרקע (הסרת זמני הרחפנים מקובץ השמע)
                drone_intervals.sort()
                bg_start = 0.0
                
                for start, end in drone_intervals:
                    start_curr = min(start, total_duration)
                    current_time = bg_start
                    while current_time + SEGMENT_DURATION <= start_curr:
                        slice_and_save(y, sr, current_time, current_time + SEGMENT_DURATION, BG_OUT, f"{folder_name}_{base_name}", chunk_idx)
                        current_time += SEGMENT_DURATION
                        chunk_idx += 1
                    bg_start = max(bg_start, end)
                
                bg_start_curr = min(bg_start, total_duration)
                current_time = bg_start_curr
                while current_time + SEGMENT_DURATION <= total_duration:
                    slice_and_save(y, sr, current_time, current_time + SEGMENT_DURATION, BG_OUT, f"{folder_name}_{base_name}", chunk_idx)
                    current_time += SEGMENT_DURATION
                    chunk_idx += 1

    print("\n--- Data Preparation Finished! ---")
    print(f"Drone segments saved to: {DRONE_OUT}")
    print(f"Background segments saved to: {BG_OUT}")

if __name__ == "__main__":
    process_dataset()