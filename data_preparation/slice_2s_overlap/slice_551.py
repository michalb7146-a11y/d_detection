import os
import glob
import librosa
import soundfile as sf
from tqdm import tqdm

# --- הגדרות נתיבים ---
INPUT_DIR = r"/Users/deviceone/Documents/data/551/raw_extracted_segments_551" 
FINAL_OUTPUT_DIR = r"/Users/deviceone/Documents/data/551/slice_2s_overlap_551_device_1"

SEGMENT_DURATION = 2.0
SR = 16000

RAW_DRONE_DIR = os.path.join(INPUT_DIR, "raw_drone")
RAW_BG_DIR = os.path.join(INPUT_DIR, "raw_background")

FINAL_DRONE_OUT = os.path.join(FINAL_OUTPUT_DIR, "target_drone")
FINAL_BG_OUT = os.path.join(FINAL_OUTPUT_DIR, "background")

os.makedirs(FINAL_DRONE_OUT, exist_ok=True)
os.makedirs(FINAL_BG_OUT, exist_ok=True)

def slice_folder(input_folder, output_folder, overlap_percentage):
    wav_files = glob.glob(os.path.join(input_folder, "*.wav"))
    print(f"\n✂️ Slicing files from {os.path.basename(input_folder)} with {overlap_percentage*100}% overlap...")
    
    # חישוב הצעד בשניות (עבור 75% חפיפה - קפיצות של 0.5 שניות)
    hop_duration = SEGMENT_DURATION * (1.0 - overlap_percentage)
    total_segments_created = 0

    for wav_path in tqdm(wav_files, desc="Slicing"):
        base_name = os.path.splitext(os.path.basename(wav_path))[0]
        try:
            y, sr = librosa.load(wav_path, sr=SR, mono=True)
            total_duration = librosa.get_duration(y=y, sr=sr)
        except Exception as e:
            print(f"Error loading {wav_path}: {e}")
            continue

        current_time = 0.0
        segment_idx = 0
        
        while current_time + SEGMENT_DURATION <= total_duration:
            start_sample = int(current_time * sr)
            end_sample = int((current_time + SEGMENT_DURATION) * sr)
            
            chunk = y[start_sample:end_sample]
            
            out_filename = f"{base_name}_win_{segment_idx}.wav"
            sf.write(os.path.join(output_folder, out_filename), chunk, sr)
            
            segment_idx += 1
            total_segments_created += 1
            current_time += hop_duration

    print(f"✅ Generated {total_segments_created} chunks of 2-seconds in: {output_folder}")

if __name__ == "__main__":
    # 1. חיתוך רחפנים עם 75% חפיפה
    slice_folder(RAW_DRONE_DIR, FINAL_DRONE_OUT, overlap_percentage=0.75)
    
    # 2. חיתוך רקע ללא חפיפה (0% חפיפה)
    slice_folder(RAW_BG_DIR, FINAL_BG_OUT, overlap_percentage=0.0)
    
    print(f"\n🚀 Pipeline Finished! Your final dataset is ready at: {FINAL_OUTPUT_DIR}")