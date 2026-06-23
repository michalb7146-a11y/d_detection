import os
import glob
import librosa
import soundfile as sf
from tqdm import tqdm

# --- הגדרות נתיבים ---
# INPUT_DIR = r"/Users/deviceone/Documents/data/2026.04.28_omesi/raw_extracted_segments" # תיקיית המקור מסקריפט 1
# FINAL_OUTPUT_DIR = r"/Users/deviceone/Documents/data/2026.04.28_omesi/raw_extracted_segments/slice_2s_overlap_2026.04.28_omesi" # תיקיית היעד הסופית

INPUT_DIR = r"/Users/deviceone/Documents/data/2026.05.01_omesi/raw_extracted_segments" # תיקיית המקור מסקריפט 1
FINAL_OUTPUT_DIR = r"/Users/deviceone/Documents/data/2026.05.01_omesi/slice_2s_overlap_2026.05.01_omesi" # תיקיית היעד הסופית

SEGMENT_DURATION = 2.0
SR = 16000

# נתיבי קלט
RAW_DRONE_DIR = os.path.join(INPUT_DIR, "raw_drone")
RAW_BG_DIR = os.path.join(INPUT_DIR, "raw_background")

# נתיבי פלט סופיים
FINAL_DRONE_OUT = os.path.join(FINAL_OUTPUT_DIR, "target_drone")
FINAL_BG_OUT = os.path.join(FINAL_OUTPUT_DIR, "background")

os.makedirs(FINAL_DRONE_OUT, exist_ok=True)
os.makedirs(FINAL_BG_OUT, exist_ok=True)

def slice_folder(input_folder, output_folder, overlap_percentage):
    wav_files = glob.glob(os.path.join(input_folder, "*.wav"))
    print(f"\n✂️ Slicing files from {os.path.basename(input_folder)} with {overlap_percentage*100}% overlap...")
    
    # חישוב הצעד (Hop Size) בשניות: עבור 75% חפיפה מחלון של 2 שניות, נקפוץ ב-0.5 שניות
    hop_duration = SEGMENT_DURATION * (1.0 - overlap_percentage)
    total_segments_created = 0

    for wav_path in tqdm(wav_files):
        base_name = os.path.splitext(os.path.basename(wav_path))[0]
        try:
            y, sr = librosa.load(wav_path, sr=SR, mono=True)
            total_duration = librosa.get_duration(y=y, sr=sr)
        except Exception as e:
            print(f"Error loading {wav_path}: {e}")
            continue

        current_time = 0.0
        segment_idx = 0
        
        # ריצה בלולאה עם חלון מתגלגל וחופף
        while current_time + SEGMENT_DURATION <= total_duration:
            start_sample = int(current_time * sr)
            end_sample = int((current_time + SEGMENT_DURATION) * sr)
            
            chunk = y[start_sample:end_sample]
            
            # שמירת החלון בגודל 2 שניות בדיוק
            out_filename = f"{base_name}_win_{segment_idx}.wav"
            sf.write(os.path.join(output_folder, out_filename), chunk, sr)
            
            segment_idx += 1
            total_segments_created += 1
            
            # התקדמות הצעד: ברחפנים נקפוץ ב-0.5 שניות, ברקע נקפוץ ב-2.0 שניות
            current_time += hop_duration

    print(f"✅ Generated {total_segments_created} chunks of 2-seconds in: {output_folder}")

if __name__ == "__main__":
    # 1. חיתוך רחפנים עם 75% חפיפה (מייצר המון חלונות יציבים למודל)
    slice_folder(RAW_DRONE_DIR, FINAL_DRONE_OUT, overlap_percentage=0.75)
    
    # 2. חיתוך רקע ללא חפיפה (0% חפיפה - קפיצות קשיחות של 2 שניות)
    slice_folder(RAW_BG_DIR, FINAL_BG_OUT, overlap_percentage=0.0)
    
    print(f"\n🚀 Slicing Finished! Final dataset is waiting for your models at: {FINAL_OUTPUT_DIR}")