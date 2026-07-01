import os
import glob
import numpy as np
import librosa
import xgboost as xgb
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
from collections import defaultdict

# ======================================================================
# 🛠️ CONFIGURATION - הגדרות נתיבים ומילוני מיפוי
# ======================================================================
# שנה נתיב זה לנתיב שבו נמצאת התיקייה המכילה את raw_drone ו-raw_background החדשים
NEW_TEST_DATA_DIR = r"/Users/deviceone/Documents/data/2026.06.09_kakadoo_just_background" 
MODEL_PICKLE_PATH = r"/Users/deviceone/Documents/d_detection/models/2s_model_omesi.pickle"
MODEL_OUTPUT_DIR = r"/Users/deviceone/Documents/d_detection/models" # נתיב לשמירת גרפי ה-Timeline
CHOSEN_THRESHOLD = 0.8  

binary_map = {
    0: ['raw_background'], 
    1: ['raw_drone']  
}

# ======================================================================
# 🧬 פונקציית חילוץ הפיצ'רים המקורית (זהה לחלוטין לאימון)
# ======================================================================
def extract_windows_from_stft(file_path):
    target_sr = 16000
    
    # 1. טעינת הקובץ בפורמט המקורי שלו (ללא מיזוג אוטומטי למונו)
    y, sr = librosa.load(file_path, sr=target_sr, mono=False)
    
    # 2. אם הקובץ הוא סטריאו (מערך דו-ממדי), ניקח רק את הערוץ הראשון (השמאלי)
    if len(y.shape) > 1 and y.shape[0] > 1:
        y = y[0] # לקיחת ערוץ שמאל בלבד כדי לשמור על פאזה נקייה מהתאבכויות
        
    # 3. נורמליזציה רגילה כפי שהיה לך
    y = y / (np.max(np.abs(y)) + 1e-5)

    stft_complex_full = librosa.stft(y, n_fft=2048, hop_length=512)
    stft_full = np.abs(stft_complex_full)
    
    frames_per_2s = int((2 * target_sr) / 512) 
    step_size = int(frames_per_2s * 0.25) 
    
    total_frames = stft_full.shape[1]
    window_features = []
    
    if total_frames < frames_per_2s:
        return window_features

    for start_frame in range(0, total_frames - frames_per_2s + 1, step_size):
        end_frame = start_frame + frames_per_2s
        
        stft = stft_full[:, start_frame:end_frame]
        stft_complex = stft_complex_full[:, start_frame:end_frame]
        
        phase = np.angle(stft_complex)
        unwrapped_phase = np.unwrap(phase, axis=1)
        phase_derivative = np.diff(unwrapped_phase, axis=1)
        phase_std_per_freq = np.std(phase_derivative, axis=1)
        
        phase_stability_low = np.mean(phase_std_per_freq[0:128])       
        phase_stability_mid = np.mean(phase_std_per_freq[384:512])     
        phase_stability_high = np.mean(phase_std_per_freq[896:1024])   
        
        low_band = stft[0:128, :]
        mid_high_band = stft[384:512, :]
        extreme_high_band = stft[896:1024, :]
        
        low_energy = np.mean(low_band) + 1e-5
        mid_high_energy = np.mean(mid_high_band)
        extreme_high_energy = np.mean(extreme_high_band)
        
        ratio_mid_low = mid_high_energy / low_energy          
        ratio_extreme_low = extreme_high_energy / low_energy  
        
        stft_mean = np.mean(stft, axis=1)
        stft_std = np.std(stft, axis=1)
        
        mfccs = librosa.feature.mfcc(S=librosa.amplitude_to_db(stft + 1e-5), sr=target_sr, n_mfcc=5)
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)
        
        flat_features = np.concatenate([
            stft_mean, stft_std,
            mfcc_mean, mfcc_std,
            [ratio_mid_low, ratio_extreme_low],
            [phase_stability_low, phase_stability_mid, phase_stability_high]
        ])
        
        window_features.append(flat_features)
        
    return window_features

def load_new_test_data(base_path, label_folder_map):
    X, y, file_paths, scenarios, timestamps = [], [], [], [], []
    window_stride_seconds = (int(((2 * 16000) / 512) * 0.25) * 512) / 16000 

    print(f"\n--- Scanning Directory for New Test Set: {base_path} ---")
    
    # בדיקה מקדימה שהתיקייה הראשית קיימת
    if not os.path.exists(base_path):
        print(f"❌ Error: Base path does not exist: {base_path}")
        return np.array(X), np.array(y), np.array(file_paths), np.array(scenarios), np.array(timestamps)

    for label, folders in label_folder_map.items():
        for folder in folders:
            folder_path = os.path.join(base_path, folder)
            print(f"🔎 Checking folder: {folder_path} (Expected Label: {label})")
            
            if not os.path.exists(folder_path):
                print(f"⚠️ Warning: Sub-folder not found: {folder_path}")
                continue
                
            # תמיכה גם ב-.wav וגם ב-.WAV
            files = glob.glob(os.path.join(folder_path, "*.wav")) + glob.glob(os.path.join(folder_path, "*.WAV"))
            
            if len(files) == 0:
                print(f"⚠️ Warning: No .wav or .WAV files found inside: {folder_path}")
                continue
                
            print(f"🚀 Found {len(files)} files in '{folder}'. Extracting features...")
            for f in tqdm(files):
                try:
                    feats_list = extract_windows_from_stft(f)
                    if len(feats_list) == 0:
                        print(f"⚠️ File too short or empty (less than 2 seconds): {os.path.basename(f)}")
                        continue
                        
                    for win_idx, feat in enumerate(feats_list):
                        X.append(feat)
                        y.append(label)
                        file_paths.append(f)
                        scenarios.append("New_Evaluation_Set")
                        exact_time_sec = win_idx * window_stride_seconds
                        timestamps.append(exact_time_sec)
                except Exception as e:
                    print(f"❌ Error processing file {f}: {e}")
                    
    return np.array(X), np.array(y), np.array(file_paths), np.array(scenarios), np.array(timestamps)

# ======================================================================
# 📊 פונקציות ניתוח והדמיה גרפית (Confusion Matrix, ROC, Timelines)
# ======================================================================
def print_detailed_errors(y_test, preds, show_matrix=True):
    if not show_matrix:
        return
    cm = confusion_matrix(y_test, preds)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
        print("\n--- DETAILED ERROR ANALYSIS (Confusion Matrix) ---")
        print(f"OK True Negatives  (Correct Background): {tn}")
        print(f"XX False Positives (False Alarms):       {fp}")
        print(f"XX False Negatives (Missed Detections):  {fn}")
        print(f"OK True Positives  (Correct Drone):     {tp}\n")
    else:
        print("\n--- Raw Confusion Matrix ---")
        print(cm)

def plot_confusion_matrix_graphic(y_test, preds):
    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Background', 'Drone'])
    disp.plot(cmap=plt.cm.Blues, values_format='d', ax=ax, colorbar=False)
    plt.title("Confusion Matrix - Drone Detection Performance (New Test)", fontsize=12, fontweight='bold', pad=15)
    plt.xlabel("Predicted Label", fontsize=10, fontweight='bold')
    plt.ylabel("True Label", fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_log_roc_curve(model, X_test, y_test, target_class_index=1):
    y_probs = model.predict_proba(X_test)[:, target_class_index]
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 7))
    plt.semilogx(fpr, tpr, color='darkorange', lw=2, label=f'Log ROC (area = {roc_auc:.4f})')
    plt.semilogx(fpr, fpr, color='navy', lw=1, linestyle='--', label='Random Classifier')
    
    plt.xlim([0.001, 1.0]) 
    plt.ylim([0.0, 1.05])
    plt.yticks(np.arange(0, 1.05, 0.05))
    plt.xlabel('False Positive Rate (Log Scale)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('Logarithmic ROC Curve (Focus on High Rejection) - New Test')
    plt.legend(loc="lower right")
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.show()

def plot_scenarios_timeline_by_recordings(y_test, y_probs, custom_preds, scenarios_test, paths_test, timestamps_test):
    nested_data = defaultdict(lambda: defaultdict(list))
    
    for i in range(len(y_test)):
        scenario = scenarios_test[i] 
        file_path = paths_test[i]
        recording_name = os.path.basename(file_path)
        
        true_label = y_test[i]
        pred_label = custom_preds[i]
        
        if true_label == 1 and pred_label == 1:
            color = '#2ca02c'   
            label_text = 'Correct Drone'
        elif true_label == 0 and pred_label == 0:
            color = '#98df8a'   
            label_text = 'Correct Background'
        elif true_label == 0 and pred_label == 1:
            color = '#d62728'   
            label_text = 'False Alarm (FA)'
        elif true_label == 1 and pred_label == 0:
            color = '#ff7f0e'   
            label_text = 'Missed Detection (MD)'
            
        nested_data[scenario][recording_name].append({
            'true': true_label,
            'color': color,
            'label_text': label_text,
            'time_sec': timestamps_test[i]
        })

    print("\n" + "="*80)
    print("🎯 LOGGING FALSE ALARM TIMESTAMPS FOR AUDACITY INSPECTION (MATCHED 100%):")
    print("="*80)

    for scenario_name, recordings in sorted(nested_data.items()):
        num_recordings = len(recordings)
        if num_recordings == 0: continue
        
        fig, axes = plt.subplots(num_recordings, 1, figsize=(15, 4.2 * num_recordings), sharex=False)
        if num_recordings == 1:
            axes = [axes]
            
        fig.suptitle(f"🎬 EXPERIMENT TIMELINE: {scenario_name}", fontsize=14, fontweight='bold', y=0.98)
        
        for idx, (rec_name, windows) in enumerate(sorted(recordings.items())):
            ax = axes[idx]
            windows.sort(key=lambda x: x['time_sec'])
            
            times_sec = np.array([w['time_sec'] for w in windows])
            y_values = [w['true'] for w in windows]
            colors = [w['color'] for w in windows]
            
            ax.plot(times_sec, y_values, color='gray', linestyle='--', alpha=0.2, lw=1.0)
            
            visible_labels = set()
            for i in range(len(windows)):
                col = colors[i]
                lbl = windows[i]['label_text']
                current_time = times_sec[i]
                
                if lbl in visible_labels:
                    lbl = "_" + lbl
                else:
                    visible_labels.add(lbl)
                
                ax.scatter(current_time, y_values[i], color=col, s=80, edgecolors='black', linewidths=0.8, zorder=3, label=lbl)
                
                if col == '#d62728':
                    minutes = int(current_time // 60)
                    seconds = current_time % 60
                    time_str = f"{minutes}:{seconds:05.2f}"
                    
                    ax.annotate(time_str, 
                                xy=(current_time, y_values[i]),
                                xytext=(0, 10), 
                                textcoords='offset points', 
                                ha='center', 
                                va='bottom',
                                fontsize=8, 
                                color='darkred', 
                                fontweight='bold',
                                bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.7, ec='red', lw=0.5))
                    
                    print(f"  ❌ [FA] File: {rec_name:<35} | Audacity Time: {time_str} ({current_time:.2f}s)")
            
            ax.set_title(f"🎵 Recording File: {rec_name}", fontsize=10, fontweight='bold', color='navy', loc='left')
            ax.set_yticks([0, 1])
            ax.set_yticklabels(['Background (0)', 'Drone (1)'], fontsize=9, fontweight='bold')
            ax.set_xlabel("Real Audio Time (Seconds)", fontsize=9)
            
            if len(times_sec) > 0:
                ax.set_xlim([-10, max(times_sec) + 30])
                
            ax.set_ylim([-0.3, 1.45])
            ax.grid(True, linestyle=':', alpha=0.5)
            
            if idx == 0:
                ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1.2), fontsize=8, fancybox=True, shadow=True)
                
        plt.tight_layout()
        plt.subplots_adjust(top=0.92) 
        
        save_path = os.path.join(MODEL_OUTPUT_DIR, f"new_test_timeline_{scenario_name}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)  
        print(f"  🎯 Saved Image for [{scenario_name}] -> {save_path}")

# ======================================================================
# 🚀 הריצה הראשית (Main Execution)
# ======================================================================
if __name__ == "__main__":
    print(f"🔄 Loading trained model from: {MODEL_PICKLE_PATH}")
    if not os.path.exists(MODEL_PICKLE_PATH):
        raise FileNotFoundError(f"Could not find pickle file at {MODEL_PICKLE_PATH}")
        
    with open(MODEL_PICKLE_PATH, 'rb') as file:
        loaded_model = pickle.load(file)
    print("✅ Model loaded successfully.")

    # 1. טעינת הנתונים החדשים
    X_new, y_new, paths_new, scenarios_new, timestamps_new = load_new_test_data(NEW_TEST_DATA_DIR, binary_map)
    
    if len(X_new) == 0:
        print("❌ No data was extracted. Please check your folder paths and file formats.")
    else:
        print(f"\n✅ Data generation complete. Total windows for testing: {len(X_new)}")
        
        # 2. הרצת החיזוי והוצאת הסתברויות
        y_probs = loaded_model.predict_proba(X_new)[:, 1]
        
        # 3. החלת הסף המותאם אישית
        custom_preds = (y_probs >= CHOSEN_THRESHOLD).astype(int)
        
        # 4. הפקת דוחות וגרפים (Confusion Matrix ו-ROC)
        print(f"\n--- Performance Evaluation (Threshold = {CHOSEN_THRESHOLD}) ---")
        print(classification_report(y_new, custom_preds, target_names=['Background', 'Drone'], zero_division=0))
        
        print_detailed_errors(y_new, custom_preds)
        
        print("📈 Displaying Confusion Matrix...")
        plot_confusion_matrix_graphic(y_new, custom_preds)
        
        print("📈 Plotting Log ROC Curve...")
        plot_log_roc_curve(loaded_model, X_new, y_new, target_class_index=1)
        
        # 5. ציור גרפי ה-Timeline המיושרים ל-Audacity ושמירתם בדיסק
        print("📈 Plotting timelines matched perfectly to Audacity timelines (Minutes:Seconds)...")
        plot_scenarios_timeline_by_recordings(y_new, y_probs, custom_preds, scenarios_new, paths_new, timestamps_new)