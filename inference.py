import os
import glob
import numpy as np
import pandas as pd
import librosa
from sklearn.metrics import confusion_matrix
import xgboost as xgb  # או הספריה שבה שמרת את המודל שלך (למשל pickle / joblib)
# לחילופין, אם שמרת ב-pickle: import pickle

# --- הגדרות נתיבים (עדכני לנתיבים שלך) ---
# התיקייה שבה יושבים תתי-התיקיות של הדאטהסט הסופי (background ו-target_drone) שאת רוצה לבחון
TEST_DATA_DIR = r"/Users/deviceone/Downloads/new_balanced_2s_dataset_551_device_1"
MODEL_PATH = r"/Users/deviceone/Downloads/models/cascade_xgb_model.json" # נתיב למודל המאומן שלך

SR = 16000

def extract_features_from_file(wav_path):
    """
    פונקציית עזר לחילוץ הפיצ'רים מקובץ בודד של 2 שניות.
    חשוב: פונקציה זו חייבת להיות זהה לחלוטין לדרך שבה חילצת פיצ'רים בזמן האימון!
    (התאימי את הבלוק הזה בדיוק לפי הפיצ'רים שאת משתמשת בהם בקוד האימון שלך, 
    למשל: MFCC, Mel-Spectrogram, השטחה ל-1D וכו').
    """
    try:
        y, sr = librosa.load(wav_path, sr=SR, mono=True)
        
        # דוגמה לחילוץ מל-ספקטרוגרמה (שני את זה לפי מה שיש לך בקוד האימון):
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # השטחה לוקטור מאפיינים אחד
        features = mel_spec_db.flatten()
        return features
    except Exception as e:
        print(f"Error extracting features from {wav_path}: {e}")
        return None

def load_test_dataset():
    """טוען את קבצי הבדיקה ומכין מערכים של פיצ'רים ותגיות אמת"""
    X_test = []
    y_true = []
    
    drone_dir = os.path.join(TEST_DATA_DIR, "target_drone")
    bg_dir = os.path.join(TEST_DATA_DIR, "background")
    
    # 1. טעינת רחפנים (תגית 1)
    drone_files = glob.glob(os.path.join(drone_dir, "*.wav"))
    print(f"📦 Loading {len(drone_files)} drone files for testing...")
    for wav_path in drone_files:
        feats = extract_features_from_file(wav_path)
        if feats is not None:
            X_test.append(feats)
            y_true.append(1)
            
    # 2. טעינת רקע (תגית 0)
    bg_files = glob.glob(os.path.join(bg_dir, "*.wav"))
    print(f"📦 Loading {len(bg_files)} background files for testing...")
    for wav_path in bg_files:
        feats = extract_features_from_file(wav_path)
        if feats is not None:
            X_test.append(feats)
            y_true.append(0)
            
    return np.array(X_test), np.array(y_true)

def evaluate_thresholds():
    # 1. טעינת הדאטה של הבדיקה
    X_test, y_true = load_test_dataset()
    if len(X_test) == 0:
        print("❌ No test files loaded. Check your TEST_DATA_DIR paths.")
        return
        
    # 2. טעינת המודל המאומן
    print("\n🤖 Loading trained XGBoost model...")
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    # הערה: אם שמרת את המודל באמצעות pickle, החליפי את השורות למעלה ב:
    # with open(MODEL_PATH, 'rb') as f:
    #     model = pickle.load(f)

    # 3. הרצת אינפרנס גולמי - קבלת הסתברויות (Probabilities) לכל חלון
    print("🔮 Running inference on test set...")
    # predict_proba מחזיר מערך של [הסתברות ל-0, הסתברות ל-1]. אנחנו צריכים את ההסתברות ל-1 (רחפן)
    probabilities = model.predict_proba(X_test)[:, 1]

    # 4. לולאת ספים לבדיקת ביצועים
    thresholds_to_test = [0.3, 0.4, 0.5, 0.6]
    
    print("\n==================================================")
    print("       🎯 THRESHOLD OPTIMIZATION REPORT          ")
    print("==================================================")
    
    for thresh in thresholds_to_test:
        # סיווג סופי לפי הסף הנוכחי
        y_pred = (probabilities >= thresh).astype(int)
        
        # חישוב מטריצת הבלבול
        cm = confusion_matrix(y_true, y_pred)
        
        # חילוץ המדדים מתוך המטריצה
        # cm[0,0] = True Negative (רקע שזוהה כרקע)
        # cm[0,1] = False Positive (אזעקת שווא)
        # cm[1,0] = False Negative (פספוס רחפן)
        # cm[1,1] = True Positive (רחפן שזוהה כרחפן)
        tn, fp, fn, tp = cm.ravel()
        
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        print(f"\n📊 Results for Threshold = {thresh}:")
        print(f"------------------------------------")
        print(f"  [True Background]  Correct: {tn} | False Alarms (FA): {fp}")
        print(f"  [True Drone     ]  Correct: {tp} | Missed Drones (MD): {fn}")
        print(f"  --> Detection Rate (Recall): {recall*100:.2f}%")
        print(f"  --> Precision: {precision*100:.2f}%")
        
    print("\n==================================================")

if __name__ == "__main__":
    evaluate_thresholds()