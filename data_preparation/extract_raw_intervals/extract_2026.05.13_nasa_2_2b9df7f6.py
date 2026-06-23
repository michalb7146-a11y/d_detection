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

UDIO_DIR = r"/Users/deviceone/Documents/data/tut/tut_audio"
CSV_DIR = r""
OUTPUT_DIR = r"/Users/deviceone/Documents/data/tut/raw_extracted_segments_tut" 

SR = 16000 

DRONE_OUT = os.path.join(OUTPUT_DIR, "raw_drone")
BG_OUT = os.path.join(OUTPUT_DIR, "raw_background")
os.makedirs(DRONE_OUT, exist_ok=True)
os.makedirs(BG_OUT, exist_ok=True)

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

def parse_tsv_file(file_path):
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
                        dur_sec = 4.0  
                    drone_intervals.append((start_sec, start_sec + dur_sec))
    except Exception as e:
        print(f"Error reading metadata file {file_path}: {e}")
    return drone_intervals

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

def process_dataset():
    metadata_files = glob.glob(os.path.join(CSV_DIR, "*.csv")) + glob.glob(os.path.join(CSV_DIR, "*.tsv"))
    print(f"Found {len(metadata_files)} metadata files to process.")

    for meta_path in tqdm(metadata_files, desc="Extracting raw intervals"):
        base_name = os.path.splitext(os.path.basename(meta_path))[0]
        wav_path = os.path.join(AUDIO_DIR, f"{base_name}.wav")
        
        if not os.path.exists(wav_path):
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
        
        # תרחיש 1: אין רחפנים - כל הקובץ הוא רקע מלא
        if not drone_intervals:
            save_full_segment(y, sr, 0.0, total_duration, BG_OUT, base_name, chunk_idx)
        else:
            # תרחיש 2: יש רחפנים - חילוץ גושי רחפנים מלאים
            for start, end in drone_intervals:
                start_curr = min(start, total_duration)
                end_curr = min(end, total_duration)
                save_full_segment(y, sr, start_curr, end_curr, DRONE_OUT, base_name, chunk_idx)
                chunk_idx += 1
            
            # חילוץ גושי הרקע המלאים
            drone_intervals.sort()
            bg_start = 0.0
            for start, end in drone_intervals:
                start_curr = min(start, total_duration)
                save_full_segment(y, sr, bg_start, start_curr, BG_OUT, base_name, chunk_idx)
                chunk_idx += 1
                bg_start = max(bg_start, end)
            
            bg_start_curr = min(bg_start, total_duration)
            if bg_start_curr < total_duration:
                save_full_segment(y, sr, bg_start_curr, total_duration, BG_OUT, base_name, chunk_idx)

    print(f"\n--- Stage 1 Finished! Full raw intervals saved to: {OUTPUT_DIR} ---")

if __name__ == "__main__":
    process_dataset()