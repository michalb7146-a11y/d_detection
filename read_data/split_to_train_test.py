import os
import glob
import shutil
import numpy as np
from sklearn.model_selection import train_test_split

# ======================================================================
# 🛠️ CONFIGURATION - הגדרת נתיבים
# ======================================================================
# הנתיב לתיקייה הראשית הנוכחית שמכילה את raw_background ו-raw_drone
SOURCE_DIR = r"/Users/deviceone/Documents/data/2026.06.07_manatees/raw_extracted_segments"

# הנתיב לתיקייה החדשה שבה תרצה ליצור את הפיצול
OUTPUT_DIR = r"/Users/deviceone/Documents/data/2026.06.07_manatees/SPLITTED"

SUB_FOLDERS = ['raw_background', 'raw_drone']

def split_dataset_on_disk():
    print(f"🚀 Starting file-based split (80/20) from: {SOURCE_DIR}")
    
    for folder in SUB_FOLDERS:
        source_folder_path = os.path.join(SOURCE_DIR, folder)
        
        # בדיקה שהתיקייה קיימת (למקרה שאחת מהן ריקה או לא קיימת)
        if not os.path.exists(source_folder_path):
            print(f"⚠️ Warning: Folder {folder} not found in source directory. Skipping.")
            continue
            
        # שליפת כל קובצי ה-wav (תומך גם ב-WAV באותיות גדולות)
        all_files = glob.glob(os.path.join(source_folder_path, "*.wav")) + glob.glob(os.path.join(source_folder_path, "*.WAV"))
        
        if len(all_files) == 0:
            print(f"⚠️ Warning: No .wav files found inside {folder}.")
            continue
            
        print(f"📂 Found {len(all_files)} files in '{folder}'. Splitting...")
        
        # פיצול רשימת הקבצים הספציפית הזו ל-80% ו-20%
        # משתמשים ב-random_state קבוע כדי שהתוצאה תהיה שחזורה במידת הצורך
        train_files, test_files = train_test_split(all_files, test_size=0.2, random_state=42)
        
        # הגדרת נתיבי יעד
        train_target_dir = os.path.join(OUTPUT_DIR, "train_set", folder)
        test_target_dir = os.path.join(OUTPUT_DIR, "test_set", folder)
        
        # יצירת התיקיות במידה ואינן קיימות
        os.makedirs(train_target_dir, exist_ok=True)
        os.makedirs(test_target_dir, exist_ok=True)
        
        # העתקת קבצי ה-Train (80%)
        print(f"   -> Copying {len(train_files)} files to Train...")
        for file_path in train_files:
            file_name = os.path.basename(file_path)
            shutil.copy2(file_path, os.path.join(train_target_dir, file_name))
            
        # העתקת קבצי ה-Test (20%)
        print(f"   -> Copying {len(test_files)} files to Test...")
        for file_path in test_files:
            file_name = os.path.basename(file_path)
            shutil.copy2(file_path, os.path.join(test_target_dir, file_name))
            
    print(f"\n✅ Splitting complete! Your new datasets are ready at:\n➡️ {OUTPUT_DIR}")

if __name__ == "__main__":
    split_dataset_on_disk()