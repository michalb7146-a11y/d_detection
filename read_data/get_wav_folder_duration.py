import os
import wave
import contextlib

def get_total_wav_duration(folder_path):
    total_seconds = 0.0
    wav_count = 0

    # בדיקה האם התיקייה קיימת
    if not os.path.exists(folder_path):
        print(f"Error: The specified path does not exist: {folder_path}")
        return

    # מעבר על כל הקבצים בתיקייה
    for filename in os.listdir(folder_path):
        # סינון קבצים עם סיומת .wav בלבד
        if filename.lower().endswith('.wav'):
            file_path = os.path.join(folder_path, filename)
            try:
                # פתיחת קובץ ה-WAV וחישוב האורך שלו
                with contextlib.closing(wave.open(file_path, 'r')) as f:
                    frames = f.getnframes()
                    rate = f.getframerate()
                    duration = frames / float(rate)
                    total_seconds += duration
                    wav_count += 1
            except Exception as e:
                print(f"Error reading file {filename}: {e}")

    # המרת השניות לשעות, דקות ושניות לטובת תצוגה נוחה
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    
    total_hours_decimal = total_seconds / 3600

    print(f"Total WAV files found: {wav_count}")
    print(f"Total duration (HH:MM:SS): {hours:02d}:{minutes:02d}:{seconds:02d}")
    print(f"Total duration in hours (decimal): {total_hours_decimal:.2f} hours")

# לשימוש בקוד, שנה את הנתיב למטה לנתיב של התיקייה שלך:
folder_path = r"/Users/deviceone/Downloads/ddl"
get_total_wav_duration(folder_path)