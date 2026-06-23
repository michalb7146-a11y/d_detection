import os
import glob
import numpy as np
import librosa
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
from collections import defaultdict

def save_trained_model_as_pickle(model, filename="2s_model_omesi.pkl"):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model saved successfully as {filename}")

def print_detailed_errors(y_test, preds, show_matrix=True):
    if not show_matrix:
        return
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    print("\n--- DETAILED ERROR ANALYSIS (Confusion Matrix) ---")
    print(f"OK True Negatives  (Correct Background): {tn}")
    print(f"XX False Positives (False Alarms):       {fp}")
    print(f"XX False Negatives (Missed Detections):  {fn}")
    print(f"OK True Positives  (Correct Drone):     {tp}\n")

def plot_confusion_matrix_graphic(y_test, preds):
    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Background', 'Drone'])
    disp.plot(cmap=plt.cm.Blues, values_format='d', ax=ax, colorbar=False)
    plt.title("Confusion Matrix - Drone Detection Performance", fontsize=12, fontweight='bold', pad=15)
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
    plt.title('Logarithmic ROC Curve (Focus on High Rejection)')
    plt.legend(loc="lower right")
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.show()

    max_allowed_fpr = 0.015 
    idx = np.argmin(np.abs(fpr - max_allowed_fpr))
    
    print("\n" + "="*50)
    print("🎯 RECOMMENDED THRESHOLD BASED ON ROC CURVE:")
    print(f"To achieve {fpr[idx]*100:.2f}% False Alarms:")
    print(f"-> Detection Rate (TPR) will be: {tpr[idx]*100:.2f}%")
    print(f"-> SET YOUR THRESHOLD TO: {thresholds[idx]:.4f}")
    print("="*50 + "\n")
    
    return fpr, tpr, thresholds, thresholds[idx]

def extract_features_vectorized(file_path):
    """
    📌 ADVANCED FEATURE EXTRACTION (רעיון 1 + רעיון 2):
    מחלץ STFT מורכב, מנתח יציבות פאזה אקוסטית ויחסי אנרגיה בין תדרי עבודה של רחפן.
    """
    y, sr = librosa.load(file_path, sr=16000, mono=True)
    y = y / (np.max(y) + 1e-5)
    
    # 1. חישוב STFT מורכב (Complex) לקבלת עוצמה וגם פאזה
    stft_complex = librosa.stft(y, n_fft=2048, hop_length=512)
    stft = np.abs(stft_complex)
    
    # -----------------------------------------------------------------
    # 🔥 רעיון 1: ניתוח פאזה ויציבות אקוסטית (Phase Deviation)
    # -----------------------------------------------------------------
    phase = np.angle(stft_complex)
    unwrapped_phase = np.unwrap(phase, axis=1)
    phase_derivative = np.diff(unwrapped_phase, axis=1)
    phase_std_per_freq = np.std(phase_derivative, axis=1)
    
    # סיכום יציבות הפאזה לשלושה מדדים נוחים לפי רצועות תדר קריטיות
    phase_stability_low = np.mean(phase_std_per_freq[0:128])       # תדר נמוך (0-1kHz)
    phase_stability_mid = np.mean(phase_std_per_freq[384:512])     # אזור הלהבים (3-4kHz)
    phase_stability_high = np.mean(phase_std_per_freq[896:1024])   # תדר גבוה מאוד (7-8kHz)
    
    # -----------------------------------------------------------------
    # 🔥 רעיון 2: יחסי אנרגיה ברצועות תדר (Band-Specific Energy Ratios)
    # -----------------------------------------------------------------
    low_band = stft[0:128, :]
    mid_high_band = stft[384:512, :]
    extreme_high_band = stft[896:1024, :]
    
    low_energy = np.mean(low_band) + 1e-5
    mid_high_energy = np.mean(mid_high_band)
    extreme_high_energy = np.mean(extreme_high_band)
    
    ratio_mid_low = mid_high_energy / low_energy          
    ratio_extreme_low = extreme_high_energy / low_energy  
    # -----------------------------------------------------------------
    
    # חישוב סטטיסטיקות עבור STFT המלא
    stft_mean = np.mean(stft, axis=1)
    stft_std = np.std(stft, axis=1)
    
    # חישוב 5 מקדמי MFCC מתוך ה-STFT הקיים
    mfccs = librosa.feature.mfcc(S=librosa.amplitude_to_db(stft + 1e-5), sr=sr, n_mfcc=5)
    mfcc_mean = np.mean(mfccs, axis=1)
    mfcc_std = np.std(mfccs, axis=1)
    
    # איחוד כל המערכים לוקטור אחד ארוך
    flat_features = np.concatenate([
        stft_mean, stft_std,
        mfcc_mean, mfcc_std,
        [ratio_mid_low, ratio_extreme_low],
        [phase_stability_low, phase_stability_mid, phase_stability_high]
    ])
    
    return flat_features

def prepare_data(base_paths, label_folder_map):
    X, y, file_paths = [], [], []
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
                    feat = extract_features_vectorized(f)
                    X.append(feat)
                    y.append(label)
                    file_paths.append(f)
                except Exception as e:
                    print(f"Error processing {f}: {e}")
    return np.array(X), np.array(y), np.array(file_paths)

def analyze_model_errors(y_test, y_probs, custom_preds, paths_test):
    print("\n" + "="*80)
    print("🔍 DEEP ERROR ANALYSIS & SCENARIO BREAKDOWN")
    print("="*80)

    GREEN = '\033[92m'
    PURPLE = '\033[95m'
    RESET = '\033[0m'

    scenario_stats = defaultdict(lambda: {'total': 0, 'fa_count': 0, 'md_count': 0, 'error_files': []})

    for i in range(len(y_test)):
        true_label = y_test[i]
        pred_label = custom_preds[i]
        prob = y_probs[i]
        file_path = paths_test[i]
        
        parts = file_path.split(os.sep)
        scenario = parts[-3] if len(parts) >= 3 else "Unknown"
        scenario_stats[scenario]['total'] += 1
        
        if true_label == 0 and pred_label == 1:
            scenario_stats[scenario]['fa_count'] += 1
            scenario_stats[scenario]['error_files'].append({'path': file_path, 'type': '[FA]', 'conf': prob})
        elif true_label == 1 and pred_label == 0:
            scenario_stats[scenario]['md_count'] += 1
            scenario_stats[scenario]['error_files'].append({'path': file_path, 'type': '[MD]', 'conf': 1 - prob})

    scen_w, tot_w, fa_w, md_w = 45, 10, 18, 18
    header = f"{'📋 SCENARIO SUMMARY':<{scen_w}} | {'TOTAL':<{tot_w}} | {'FALSE ALARMS (FA)':<{fa_w}} | {'MISSED DET. (MD)':<{md_w}}"
    print(f"{GREEN}{header}{RESET}")
    print(f"{GREEN}" + "-" * len(header) + f"{RESET}")
    
    for scenario, stats in scenario_stats.items():
        total = stats['total']
        fa = stats['fa_count']
        md = stats['md_count']
        fa_pct = (fa / total) * 100 if total > 0 else 0
        md_pct = (md / total) * 100 if total > 0 else 0
        fa_str = f"{fa} ({fa_pct:.1f}%)"
        md_str = f"{md} ({md_pct:.1f}%)"
        row_str = f"{scenario:<{scen_w}} | {total:<{tot_w}} | {fa_str:<{fa_w}} | {md_str:<{md_w}}"
        print(f"{GREEN}{row_str}{RESET}")
        
    print(f"{GREEN}" + "=" * len(header) + f"{RESET}")
    print("\n" + "="*80)
    print("📂 DETAILED ERROR FILE LOGS")
    print("="*80)

    has_any_errors = False
    for scenario, stats in scenario_stats.items():
        if len(stats['error_files']) == 0:
            continue
        has_any_errors = True
        print(f"\n🎬 SCENARIO: {PURPLE}{scenario}{RESET}")
        print("-" * (len(scenario) + 11))
        
        stats['error_files'].sort(key=lambda x: x['conf'], reverse=True)
        for err in stats['error_files']:
            print(f"  {err['type']} [Conf: {err['conf']*100:.1f}%] {err['path']}")
        print("." * 60)

    if not has_any_errors:
        print("\n🎉 No errors found in any scenario! Perfect classification.")
    print("=" * 80 + "\n")

def plot_feature_importance_debug(model, feature_names=None):
    importances = model.feature_importances_
    
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(len(importances))]
        
    feat_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    print("\n📊 TOP 10 MOST IMPORTANT FEATURES (XGBoost Debugging):")
    print("-" * 50)
    for idx, row in feat_df.head(10).iterrows():
        print(f"  {row['Feature']:<30} : {row['Importance']*100:.2f}%")
    print("-" * 50)
    
    plt.figure(figsize=(11, 7))
    top_n = feat_df.head(20)
    plt.barh(top_n['Feature'][::-1], top_n['Importance'][::-1], color='teal', edgecolor='black', alpha=0.8)
    plt.xlabel('Importance Score', fontweight='bold', labelpad=10)
    plt.title('Top 20 Feature Importances - Advanced Audio Feature Analysis', fontsize=12, fontweight='bold', pad=15)
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# ======================================================================
# 🛠️ CONFIGURATION
# ======================================================================

DATA_DIRECTORIES = [
    r"/Users/deviceone/Documents/data/2026.04.28_omesi/slice_2s_overlap_2026.04.28_omesi",
    r"/Users/deviceone/Documents/data/2026.05.01_omesi/slice_2s_overlap_2026.05.01_omesi",
    r"/Users/deviceone/Documents/data/dregon/slice_2s_overlap_dregon",
    r"/Users/deviceone/Documents/data/nasa_2/slice_2s_overlap_nasa_2",
    r"/Users/deviceone/Documents/data/tut/slice_2s_overlap_tut",
    r"/Users/deviceone/Documents/data/ESC-50/slice_2s_overlap_ESC-50"
]

MODEL_OUTPUT_DIR = r"/Users/deviceone/Documents/d_detection/models"

binary_map = {
    0: ['background'], 
    1: ['target_drone']  
}

if __name__ == "__main__":
    X_bin, y_bin, file_paths = prepare_data(DATA_DIRECTORIES, binary_map)
    unique_labels = np.unique(y_bin)
    
    if len(unique_labels) < 2:
        print("⚠️ Error: Dataset must contain both background and drone samples to train.")
    else:
        X_train, X_test, y_train, y_test, paths_train, paths_test = train_test_split(
            X_bin, y_bin, file_paths, test_size=0.2, random_state=42, stratify=y_bin
        )
        
        best_params = {
            'colsample_bytree': 0.7, 
            'learning_rate': 0.05, 
            'max_depth': 4, 
            'n_estimators': 500, 
            'subsample': 0.7
        }

        model_bin = xgb.XGBClassifier(
            objective='binary:logistic', eval_metric='logloss', device="cpu", **best_params
        )
        
        print("\n--- Training Model... ---")
        model_bin.fit(X_train, y_train)
        
        preds = model_bin.predict(X_test)
        print("\n--- Baseline Classifier Results (Threshold = 0.5) ---")
        print(classification_report(y_test, preds))
        
        fpr, tpr, thresholds, chosen_threshold = plot_log_roc_curve(model_bin, X_test, y_test, target_class_index=1)

        print(f"📊 Applying Custom Threshold ({chosen_threshold:.4f}) to Test Data...")
        y_probs = model_bin.predict_proba(X_test)[:, 1]
        custom_preds = (y_probs >= chosen_threshold).astype(int)
        
        print_detailed_errors(y_test, custom_preds, show_matrix=True)
        plot_confusion_matrix_graphic(y_test, custom_preds)

        analyze_model_errors(y_test, y_probs, custom_preds, paths_test)

        # 📌 בניית שמות הפיצ'רים המעודכנת לצורך הגרף הסטטיסטי
        stft_len = 1025     # n_fft // 2 + 1
        
        feature_names = []
        feature_names += [f"STFT_Mean_{i}" for i in range(stft_len)]
        feature_names += [f"STFT_Std_{i}" for i in range(stft_len)]
        feature_names += [f"MFCC_Mean_{i}" for i in range(5)]
        feature_names += [f"MFCC_Std_{i}" for i in range(5)]
        
        # שמות פיצ'רי יחסי האנרגיה (רעיון 2)
        feature_names += ["Ratio_Mid_Low_Energy", "Ratio_Extreme_Low_Energy"]
        
        # שמות פיצ'רי יציבות הפאזה (רעיון 1)
        feature_names += ["Phase_Stability_Low", "Phase_Stability_Mid", "Phase_Stability_High"]

        # הרצת גרף חשיבות הפיצ'רים המעודכן
        plot_feature_importance_debug(model_bin, feature_names=feature_names)

        if not os.path.exists(MODEL_OUTPUT_DIR):
            os.makedirs(MODEL_OUTPUT_DIR)
        model_out_path = os.path.join(MODEL_OUTPUT_DIR, "2s_model_omesi.pickle")
        save_trained_model_as_pickle(model_bin, model_out_path)