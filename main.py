import os
import pandas as pd
import librosa
import soundfile as sf
from tqdm import tqdm

def preprocess_and_organize_dataset(wav_dir, csv_dir, output_dir, segment_duration=2.0):
    """
    מארגן ומכין את הדאטא החדש על ידי חיתוך סגמנטים של שמע מבוססי CSV.
    
    :param wav_dir: נתיב לתיקיית קבצי ה-wav (כולל תתי-תיקיות)
    :param csv_dir: נתיב לתיקיית קבצי ה-CSV
    :param output_dir: נתיב לתיקיית היעד שבה ייבנה המבנה החדש
    :param segment_duration: אורך כל סגמנט בשניות (ברירת מחדל: 2 שניות)
    """
    
    # יצירת התיקיות החדשות לפי המבנה של הסקריפט המקורי שלך
    drone_out_dir = os.path.join(output_dir, "target_drone")
    bg_out_dir = os.path.join(output_dir, "background")
    os.makedirs(drone_out_dir, exist_ok=True)
    os.makedirs(bg_out_dir, exist_ok=True)
    
    # סריקת כל קבצי ה-wav (גם אם הם בתתי-תיקיות)
    wav_files = []
    for root, _, files in os.walk(wav_dir):
        for file in files:
            if file.lower().endswith('.wav'):
                wav_files.append(os.path.join(root, file))
                
    print(f"Found {len(wav_files)} WAV files to process.")
    
    drone_segments_count = 0
    bg_segments_count = 0
    
    for wav_path in tqdm(wav_files):
        base_name = os.path.splitext(os.path.basename(wav_path))[0]
        csv_path = os.path.join(csv_dir, f"{base_name}.csv")
        
        # טעינת קובץ השמע המלא
        try:
            y, sr = librosa.load(wav_path, sr=16000, mono=True)
            total_duration = librosa.get_duration(y=y, sr=sr)
        except Exception as e:
            print(f"Error loading {wav_path}: {e}")
            continue
            
        # מערך עזר באורך השניות של הקובץ כדי לדעת איזה חלק תפוס ע"י רחפן
        # 0 = רקע, 1 = רחפן
        time_mask = [0] * int(total_duration + 1)
        has_drone = False
        
        # בדיקה אם קיים קובץ CSV והוא אינו ריק
        if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
            try:
                df = pd.read_csv(csv_path)
                if not df.empty and 'start' in df.columns and 'duration' in df.columns:
                    has_drone = True
                    # סימון הזמנים שבהם יש רחפן
                    for _, row in df.iterrows():
                        start = float(row['start'])
                        end = start + float(row['duration'])
                        
                        # חיתוך סגמנטים של רחפן מתוך חלון הזמן
                        current_time = start
                        while current_time + segment_duration <= end and current_time <= total_duration:
                            start_sample = int(current_time * sr)
                            end_sample = int((current_time + segment_duration) * sr)
                            y_segment = y[start_sample:end_sample]
                            
                            # שמירת הסגמנט לתיקיית רחפנים
                            out_filename = f"{base_name}_drone_{current_time:.2f}.wav"
                            sf.write(os.path.join(drone_out_dir, out_filename), y_segment, sr)
                            drone_segments_count += 1
                            
                            # סימון השניות האלו כ"תפוסות" כדי שלא נחתוך מהן רקע בטעות
                            for sec in range(int(current_time), int(current_time + segment_duration) + 1):
                                if sec < len(time_mask):
                                    time_mask[sec] = 1
                                    
                            current_time += segment_duration # קפיצה קדימה ללא חפיפה
            except Exception as e:
                print(f"Error parsing CSV {csv_path}: {e}")
                has_drone = False # נתייחס אליו כרקע אם ה-CSV פגום
                
        # ייצור סגמנטים של רקע (Background) מהחלקים הריקים בקובץ
        current_time = 0.0
        while current_time + segment_duration <= total_duration:
            start_sec = int(current_time)
            end_sec = int(current_time + segment_duration)
            
            # בודקים אם כל החלון הנוכחי נקי מרחפנים
            is_window_clean = True
            for sec in range(start_sec, end_sec + 1):
                if sec < len(time_mask) and time_mask[sec] == 1:
                    is_window_clean = False
                    break
            
            if is_window_clean:
                start_sample = int(current_time * sr)
                end_sample = int((current_time + segment_duration) * sr)
                y_segment = y[start_sample:end_sample]
                
                # שמירת הסגמנט לתיקיית רקע
                out_filename = f"{base_name}_bg_{current_time:.2f}.wav"
                sf.write(os.path.join(bg_out_dir, out_filename), y_segment, sr)
                bg_segments_count += 1
                
                current_time += segment_duration
            else:
                # אם החלון תפוס ע"י רחפן, נתקדם מעט קדימה לחפש אזור נקי
                current_time += 1.0

    print("\n--- Processing Summary ---")
    print(f"Created {drone_segments_count} drone segments.")
    print(f"Created {bg_segments_count} background segments.")
    print(f"Organized dataset saved to: {output_dir}")

# --- הרצה ---
if __name__ == "__main__":
    # הגדר את הנתיבים שלך כאן:
    RAW_WAV_DIR = r"/Users/michalh1/Downloads/2026.05.01_omesi" 
    CSV_DIR = r"/Users/michalh1/Downloads/tagged_2026.05.01_omesi"
    OUTPUT_DATASET_DIR = r"/Users/michalh1/Downloads/new_balanced_2s_dataset_processed"

    preprocess_and_organize_dataset(
        wav_dir=RAW_WAV_DIR,
        csv_dir=CSV_DIR,
        output_dir=OUTPUT_DATASET_DIR,
        segment_duration=2.0 # תואם ל-2s של הסקריפט המקורי
    )