import os
import glob
import numpy as np
import librosa
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle

def save_trained_model_as_pickle(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model saved successfully as {filename}")

def print_detailed_errors(y_test, preds, stage_name=""):
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    print(f"[{stage_name}] OK Background: {tn} | FA (False Alarms): {fp} | MD (Missed): {fn} | OK Drone: {tp}")

def plot_confusion_matrix_graphic(y_test, preds, title):
    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots(figsize=(5, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Background', 'Drone'])
    disp.plot(cmap=plt.cm.Blues, values_format='d', ax=ax, colorbar=False)
    plt.title(title, fontsize=11, fontweight='bold', pad=15)
    plt.xlabel("Predicted Label", fontsize=10, fontweight='bold')
    plt.ylabel("True Label", fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.show()

# def extract_features_vectorized(file_path):
#     y, sr = librosa.load(file_path, sr=16000, mono=True)
#     y = y / (np.max(y) + 1e-5)
    
#     features = []
#     # חישוב התמרת פורייה (STFT) וספקטרוגרמת אמפליטודה
#     stft = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
#     chroma = librosa.feature.chroma_stft(S=stft, sr=sr)
    
#     mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=64)
#     mel_db = librosa.amplitude_to_db(mel_spec, ref=np.max)
    
#     # חילוץ מאפיינים ספקטרליים מבוססי תדר
#     centroid = librosa.feature.spectral_centroid(S=stft, sr=sr)
#     flatness = librosa.feature.spectral_flatness(S=stft)
#     rolloff_85 = librosa.feature.spectral_rolloff(S=stft, sr=sr, roll_percent=0.85)
#     rolloff_50 = librosa.feature.spectral_rolloff(S=stft, sr=sr, roll_percent=0.50)
#     zcr = librosa.feature.zero_crossing_rate(y)
    
#     # פיצ'ר רחפנים ייעודי: הפרדה הרמונית-פרקסיבית (HPSS) לשם סינון רוח וחבטות
#     y_harmonic, y_percussive = librosa.effects.hpss(y)
#     rms_harmonic = librosa.feature.rms(y=y_harmonic, hop_length=512)
#     rms_percussive = librosa.feature.rms(y=y_percussive, hop_length=512)
    
#     # שרשור ממוצע וסטיית תקן של הפיצ'רים המטריציוניים
#     for feat in [stft, chroma, mel_db]:
#         features.extend(np.mean(feat, axis=1))
#         features.extend(np.std(feat, axis=1))
        
#     # הוספת הסטטיסטיקות של הפיצ'רים הווקטוריים (כולל הפיצ'רים החדשים לרחפנים)
#     features.extend([
#         np.mean(centroid), np.std(centroid), 
#         np.mean(flatness), np.std(flatness), 
#         np.mean(rolloff_85), np.std(rolloff_85),
#         np.mean(rolloff_50), np.std(rolloff_50),
#         np.mean(zcr), np.std(zcr),
#         np.mean(rms_harmonic), np.std(rms_harmonic),      # ממוצע וסטיית תקן של עוצמה הרמונית
#         np.mean(rms_percussive), np.std(rms_percussive)    # ממוצע וסטיית תקן של עוצמה פרקסיבית
#     ])
    
#     # חישוב MFCCs בסיסיים
#     mfccs = librosa.feature.mfcc(S=librosa.amplitude_to_db(stft + 1e-5), sr=sr, n_mfcc=13)
#     features.extend(np.mean(mfccs, axis=1))
#     features.extend(np.std(mfccs, axis=1))
    
#     # הפיצ'ר החדש: Delta MFCCs (מייצג את הדינמיקה/שינוי בזמן של הפיצ'רים)
#     mfcc_delta = librosa.feature.delta(mfccs, width=3)
#     features.extend(np.mean(mfcc_delta, axis=1))
#     features.extend(np.std(mfcc_delta, axis=1))
    
#     return np.array(features)


def extract_features_vectorized(file_path):
    # 1. טעינת האודיו
    y, sr = librosa.load(file_path, sr=16000, mono=True)
    
    # [שדרוג] מסנן High-Pass Filter דיגיטלי להפחתת זמזומים קבועים מתחת ל-100Hz
    # משתמשים בפילטר butterworth בסיסי דרך scipy (או פשוט חותכים בספקטרום, אך כאן נעשה זאת באמצעות librosa/scipy)
    # כדי לשמור על פשטות ללא ספריות נוספות, ננקה את קו הבסיס ישירות מה-STFT בהמשך.
    
    y = y / (np.max(y) + 1e-5)
    
    features = []
    
    # חישוב התמרת פורייה (STFT) וספקטרוגרמת אמפליטודה
    stft = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    
    # [שדרוג] ניקוי תדרים נמוכים מאוד (מתחת ל-100 הרץ) ישירות מהספקטרוגרמה
    # ב-sr=16000 ו-n_fft=2048, כל bin בספקטרום מייצג בערך 7.8Hz. חיתוך 13 ה-bins הראשונים ינקה עד ~100Hz.
    stft[:13, :] = 0
    
    chroma = librosa.feature.chroma_stft(S=stft, sr=sr)
    
    # חישוב מיל-ספקטרוגרמה מתוך ה-STFT המנוקה
    mel_spec = librosa.feature.melspectrogram(S=stft**2, sr=sr, n_mels=64)
    mel_db = librosa.amplitude_to_db(mel_spec, ref=np.max)
    
    # חילוץ מאפיינים ספקטרליים מבוססי תדר
    centroid = librosa.feature.spectral_centroid(S=stft, sr=sr)
    flatness = librosa.feature.spectral_flatness(S=stft)
    rolloff_85 = librosa.feature.spectral_rolloff(S=stft, sr=sr, roll_percent=0.85)
    rolloff_50 = librosa.feature.spectral_rolloff(S=stft, sr=sr, roll_percent=0.50)
    zcr = librosa.feature.zero_crossing_rate(y)
    
    # [🔥 פיצ'ר חדש נגד FA] Spectral Flux: מודד שינויים דינמיים בצורת הספקטרום
    # רעש סטטי יקבל ערכי Flux נמוכים קבועים, רחפן חי ורוטט יקבל ערכים משתנים
    onset_env = librosa.onset.onset_strength(S=mel_db, sr=sr) 
    
    # פיצ'ר רחפנים ייעודי: הפרדה הרמונית-פרקסיבית (HPSS) לשם סינון רוח וחבטות
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    rms_harmonic = librosa.feature.rms(y=y_harmonic, hop_length=512)
    rms_percussive = librosa.feature.rms(y=y_percussive, hop_length=512)
    
    # שרשור ממוצע וסטיית תקן של הפיצ'רים המטריציוניים
    for feat in [stft, chroma, mel_db]:
        features.extend(np.mean(feat, axis=1))
        features.extend(np.std(feat, axis=1))
        
    # הוספת הסטטיסטיקות של הפיצ'רים הווקטוריים (כולל הפיצ'רים החדשים לרחפנים)
    features.extend([
        np.mean(centroid), np.std(centroid), 
        np.mean(flatness), np.std(flatness), 
        np.mean(rolloff_85), np.std(rolloff_85),
        np.mean(rolloff_50), np.std(rolloff_50),
        np.mean(zcr), np.std(zcr),
        np.mean(rms_harmonic), np.std(rms_harmonic),      # ממוצע וסטיית תקן של עוצמה הרמונית
        np.mean(rms_percussive), np.std(rms_percussive),  # ממוצע וסטיית תקן של עוצמה פרקסיבית
        np.mean(onset_env), np.std(onset_env)             # [🔥 חדש] ממוצע וסטיית תקן של ה-Flux (תנודתיות)
    ])
    
    # חישוב MFCCs בסיסיים
    mfccs = librosa.feature.mfcc(S=librosa.amplitude_to_db(stft + 1e-5), sr=sr, n_mfcc=13)
    features.extend(np.mean(mfccs, axis=1))
    features.extend(np.std(mfccs, axis=1))
    
    # הפיצ'ר הקיים: Delta MFCCs (מייצג את הדינמיקה/שינוי בזמן של הפיצ'רים)
    mfcc_delta = librosa.feature.delta(mfccs, width=3)
    features.extend(np.mean(mfcc_delta, axis=1))
    features.extend(np.std(mfcc_delta, axis=1))
    
    return np.array(features)

def prepare_data(base_paths, label_folder_map):
    X, y = [], []
    file_paths = []  # רשימה לשמירת נתיבי הקבצים כדי לאבד את המקור
    folder_to_label = {folder: label for label, folders in label_folder_map.items() for folder in folders}
    if isinstance(base_paths, str):
        base_paths = [base_paths]
    for base_path in base_paths:
        print(f"\n--- Scanning Base Directory: {base_path} ---")
        for folder, label in folder_to_label.items():
            folder_path = os.path.join(base_path, folder)
            if not os.path.exists(folder_path):
                continue
            files = glob.glob(os.path.join(folder_path, "*.wav"))
            if len(files) == 0:
                continue
            print(f"Loading {len(files)} files for label: {label} (from {folder})")
            for f in tqdm(files):
                try:
                    X.append(extract_features_vectorized(f))
                    y.append(label)
                    file_paths.append(f)  # שמירת נתיב הקובץ הנוכחי
                except Exception as e:
                    print(f"Error processing {f}: {e}")
    return np.array(X), np.array(y), np.array(file_paths)

# ======================================================================
# CONFIGURATION
# ======================================================================
DATA_DIRECTORIES = [
    r"/Users/deviceone/Documents/data/2026.04.28_omesi/slice_2s_overlap_2026.04.28_omesi",
    r"/Users/deviceone/Documents/data/2026.05.01_omesi/slice_2s_overlap_2026.05.01_omesi",
    r"/Users/deviceone/Documents/data/dregon/slice_2s_overlap_dregon",
    r"/Users/deviceone/Documents/data/nasa_2/slice_2s_overlap_nasa_2",
    r"/Users/deviceone/Documents/data/tut/slice_2s_overlap_tut",
    r"/Users/deviceone/Documents/data/DNC/slice_2s_overlap_dnc",
    r"/Users/deviceone/Documents/data/ESC-50/slice_2s_overlap_ESC-50"
]

MODEL_OUTPUT_DIR = r"/Users/deviceone/Documents/d_detection/models"
binary_map = {0: ['background'], 1: ['target_drone']}

if __name__ == "__main__":
    # קבלת הנתונים יחד עם רשימת הנתיבים שלהם
    X_all, y_all, paths_all = prepare_data(DATA_DIRECTORIES, binary_map)
    
    # פיצול הנתונים תוך שמירה על סנכרון עם רשימת נתיבי הקבצים (באמצעות אותו random_state)
    X_train, X_test, y_train, y_test, paths_train, paths_test = train_test_split(
        X_all, y_all, paths_all, test_size=0.2, random_state=42, stratify=y_all
    )
    
    print(f"\n--- Training Set Size: {len(X_train)} | Test Set Size: {len(X_test)} ---")
    
    # ----------------------------------------------------------------------
    # שלב 1: מודל אגרסיבי במיוחד (אבטחה מקסימלית)
    # ----------------------------------------------------------------------
    print("\n>>> Training Stage 1 Model (Hyper-Aggressive Shield)...")
    model_stage1 = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        device="cpu",
        colsample_bytree=0.7,
        learning_rate=0.05,
        max_depth=4,
        n_estimators=500,
        subsample=0.7,
        scale_pos_weight=40.0  
    )
    model_stage1.fit(X_train, y_train)
    save_trained_model_as_pickle(model_stage1, os.path.join(MODEL_OUTPUT_DIR, "cascade_stage1.pickle"))
    
    STAGE1_THRESHOLD = 0.001  
    
    # ----------------------------------------------------------------------
    # שלב 2: סינון דגימות ואימון מודל מנקה רעשים
    # ----------------------------------------------------------------------
    print("\n>>> Filtering Training Data for Stage 2...")
    probs_train_s1 = model_stage1.predict_proba(X_train)[:, 1]
    passed_s1_idx = np.where(probs_train_s1 >= STAGE1_THRESHOLD)[0]
    
    X_train_stage2 = X_train[passed_s1_idx]
    y_train_stage2 = y_train[passed_s1_idx]
    
    print(f"Stage 2 Dataset Created: {len(X_train_stage2)} samples passed Stage 1 filter.")
    
    print("\n>>> Training Stage 2 Model (The False Alarm Cleaner)...")
    model_stage2 = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        device="cpu",
        colsample_bytree=0.7,
        learning_rate=0.05,
        max_depth=5,
        n_estimators=500,
        subsample=0.8,
        scale_pos_weight=1.0  
    )
    model_stage2.fit(X_train_stage2, y_train_stage2)
    save_trained_model_as_pickle(model_stage2, os.path.join(MODEL_OUTPUT_DIR, "cascade_stage2.pickle"))
    
    # ----------------------------------------------------------------------
    # שלב 3: הרצת סימולציית סריקה על נתוני הבדיקה
    # ----------------------------------------------------------------------
    print("\n" + "="*70)
    print("🔎 SCANNING STAGE 2 THRESHOLDS FOR OPTIMAL BALANCE")
    print("="*70)
    
    probs_test_s1 = model_stage1.predict_proba(X_test)[:, 1]
    passed_test_s1_idx = np.where(probs_test_s1 >= STAGE1_THRESHOLD)[0]
    
    probs_test_s2 = np.zeros(len(X_test))
    if len(passed_test_s1_idx) > 0:
        probs_test_s2[passed_test_s1_idx] = model_stage2.predict_proba(X_test[passed_test_s1_idx])[:, 1]
    
    stage2_test_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    for t2 in stage2_test_thresholds:
        final_preds = np.zeros(len(X_test), dtype=int)
        final_preds[passed_test_s1_idx] = (probs_test_s2[passed_test_s1_idx] >= t2).astype(int)
        
        print(f"Stage 2 Threshold: {t2:<3} -> ", end="")
        print_detailed_errors(y_test, final_preds, stage_name="Cascade System")
        
    print("="*70 + "\n")
    
    # הסף הנבחר להדמיה הסופית ולניתוח הקבצים
    chosen_stage2_threshold = 0.35
    
    print(f"📊 Drawing Final Matrix for Stage 2 Threshold: {chosen_stage2_threshold}")
    final_system_preds = np.zeros(len(X_test), dtype=int)
    final_system_preds[passed_test_s1_idx] = (probs_test_s2[passed_test_s1_idx] >= chosen_stage2_threshold).astype(int)
    
    # ----------------------------------------------------------------------
    # הפקת דוח שגיאות מפורט על שמות קבצי ה-WAV
    # ----------------------------------------------------------------------
    print("\n" + "="*70)
    print("❌ DETAILED ERROR ANALYSIS - MISCLASSIFIED FILES")
    print("="*70)
    
    false_alarms = []
    missed_detections = []
    
    for i in range(len(y_test)):
        true_label = y_test[i]
        pred_label = final_system_preds[i]
        file_path = paths_test[i]
        
        # False Alarm: רקע (0) שסווג בטעות כרחפן (1)
        if true_label == 0 and pred_label == 1:
            false_alarms.append(file_path)
            
        # Missed Detection: רחפן (1) שסווג בטעות כרקע (0)
        elif true_label == 1 and pred_label == 0:
            missed_detections.append(file_path)
            
    print(f"\n--- 🚨 FALSE ALARMS ({len(false_alarms)} קבצים) ---")
    print("קולות רקע שסווגו בטעות כרחפן:")
    for fa_file in false_alarms:
        print(f"  [FA] -> {fa_file}")
        
    print(f"\n--- 📉 MISSED DETECTIONS ({len(missed_detections)} קבצים) ---")
    print("רחפנים שפוספסו וסווגו כרקע:")
    for md_file in missed_detections:
        print(f"  [MD] -> {md_file}")
        
    print("\n" + "="*70)
    
    # הצגת הגרף
    plot_confusion_matrix_graphic(y_test, final_system_preds, f"Cascade System (Stage 2 Threshold = {chosen_stage2_threshold})")