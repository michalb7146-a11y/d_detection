import os
import pickle
import csv  # ייבוא ספריית CSV לשמירת הנתונים
import numpy as np
import librosa

# 1. טעינת המודל המאומן
MODEL_PATH = r"models/2s_model.pickle"  # נתיב לקובץ הפיקל שלך
with open(MODEL_PATH, 'wb' if os.name == 'nt' else 'rb') as file:
    model = pickle.load(file)

def extract_features_vectorized(y, sr=16000):
    """
    פונקציית חילוץ הפיצ'רים המקורית שלך - חייבת להיות זהה ב-100% לאימון!
    """
    y = y / (np.max(y) + 1e-5)
    stft = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    chroma = librosa.feature.chroma_stft(S=stft, sr=sr)
    
    centroid = librosa.feature.spectral_centroid(S=stft, sr=sr)
    flatness = librosa.feature.spectral_flatness(S=stft)
    rolloff = librosa.feature.spectral_rolloff(S=stft, sr=sr, roll_percent=0.85)
    
    features = []
    for feat in [stft, chroma]:
        features.extend(np.mean(feat, axis=1))
        features.extend(np.std(feat, axis=1))
        
    features.extend([
        np.mean(centroid), np.std(centroid), 
        np.mean(flatness), np.std(flatness),
        np.mean(rolloff), np.std(rolloff)
    ])
    
    mfccs = librosa.feature.mfcc(S=librosa.amplitude_to_db(stft + 1e-5), sr=sr, n_mfcc=5)
    features.extend(np.mean(mfccs, axis=1))
    features.extend(np.std(mfccs, axis=1))
    
    return np.array(features)

def inspect_audio_file(file_path):
    """
    פונקציה שמקבלת קובץ שמע, חותכת לחלקים של 2 שניות ומנבאת עבור כל חלק.
    מחזירה רשימה של תוצאות עבור הקובץ הנוכחי.
    """
    print(f"\n--- Testing File: {os.path.basename(file_path)} ---")
    
    file_results = [] # רשימה מקומית לשמירת תוצאות הקובץ הנוכחי
    
    # טעינת השמע המלא בעזרת librosa
    y_full, sr = librosa.load(file_path, sr=16000, mono=True)
    
    segment_length_samples = 2 * sr 
    total_samples = len(y_full)
    
    start = 0
    chunk_idx = 1
    
    while start < total_samples:
        end = start + segment_length_samples
        chunk = y_full[start:end]
        
        if len(chunk) < segment_length_samples:
            break
            
        features = extract_features_vectorized(chunk, sr)
        features = features.reshape(1, -1)
        
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][prediction]
        
        label = "Drone" if prediction == 1 else "Background"
        start_sec = chunk_idx * 2 - 2
        end_sec = chunk_idx * 2
        
        print(f"Seconds {start_sec:02d}-{end_sec:02d}: Predicted as {label} (Confidence: {probability:.2%})")
        
        # הוספת הנתונים למבנה נתונים מסודר
        file_results.append({
            "File Name": os.path.basename(file_path),
            "Start Second": start_sec,
            "End Second": end_sec,
            "Prediction": label,
            "Confidence": f"{probability:.2%}"
        })
        
        start = end
        chunk_idx += 1
        
    return file_results

if __name__ == "__main__":
    folder_path = r"/Users/deviceone/Downloads/data/2026.05.01_omesi/new_balanced_2s_dataset/target_drone"
    output_csv_path = "detection_results.csv" # שם קובץ הפלט שיעבור לתיקיית העבודה הנוכחית
    
    all_results = [] # רשימה שתכיל את כל התוצאות מכל הקבצים
    
    # בדיקה האם הנתיב הוא אכן תיקייה
    if os.path.isdir(folder_path):
        for filename in os.listdir(folder_path):
            if filename.lower().endswith('.wav'):
                file_path = os.path.join(folder_path, filename)
                results = inspect_audio_file(file_path)
                all_results.extend(results) # איחוד התוצאות לרשימה הראשית
    else:
        # אם זה קובץ בודד
        all_results = inspect_audio_file(folder_path)
        
    # שמירת כל הנתונים שנאספו לקובץ CSV
    if all_results:
        keys = all_results[0].keys() # כותרות העמודות מתוך המפתחות של הדיקשנרי
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as output_file:
            dict_writer = csv.DictWriter(output_file, fieldnames=keys)
            dict_writer.writeheader() # כתיבת שורת הכותרות
            dict_writer.writerows(all_results) # כתיבת כל שורות הנתונים
            
        print(f"\n[+] All results successfully saved to: {os.path.abspath(output_csv_path)}")
    else:
        print("\n[-] No results to save.")