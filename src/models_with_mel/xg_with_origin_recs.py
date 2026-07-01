import os
import glob
import numpy as np
import librosa
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
from collections import defaultdict
import librosa.display

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

def extract_windows_from_stft(file_path):
    """
    📌 SLIDING WINDOW STFT FEATURE EXTRACTION WITH RMS NORMALIZATION
    """
    target_sr = 16000
    y, sr = librosa.load(file_path, sr=target_sr, mono=False)
    
    if len(y.shape) > 1 and y.shape[0] > 1:
        y = y[0]
        
    # 🛠️ עדכון נורמליזציה: שימוש ב-RMS במקום Max כדי למנוע FA של הגברת רעשי רקע שקטים
    rms = np.sqrt(np.mean(y**2)) + 1e-5
    y = (y / rms) * 0.1 # מביא את האות לעוצמה ממוצעת בריאה ויציבה
    
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

def prepare_data_raw(base_paths_config, label_folder_map):
    X, y, file_paths, sample_scenarios, timestamps = [], [], [], [], []
    window_stride_seconds = (int(((2 * 16000) / 512) * 0.25) * 512) / 16000 

    for item in base_paths_config:
        base_path = item['path']
        dataset_name = item['name']
        print(f"\n--- Scanning Directory for [{dataset_name}]: {base_path} ---")
        
        for label, folders in label_folder_map.items():
            for folder in folders:
                folder_path = os.path.join(base_path, folder)
                if not os.path.exists(folder_path):
                    continue
                    
                files = glob.glob(os.path.join(folder_path, "*.wav")) + glob.glob(os.path.join(folder_path, "*.WAV"))
                if len(files) == 0:
                    continue
                    
                print(f"Processing {len(files)} raw files from '{folder}' (Label: {label})")
                for f in tqdm(files):
                    try:
                        feats_list = extract_windows_from_stft(f)
                        for win_idx, feat in enumerate(feats_list):
                            X.append(feat)
                            y.append(label)
                            file_paths.append(f)
                            sample_scenarios.append(dataset_name)
                            exact_time_sec = win_idx * window_stride_seconds
                            timestamps.append(exact_time_sec)
                            
                    except Exception as e:
                        print(f"Error processing {f}: {e}")
                        
    return np.array(X), np.array(y), np.array(file_paths), np.array(sample_scenarios), np.array(timestamps)

def analyze_model_errors(y_test, y_probs, custom_preds, scenarios_test, paths_test):
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
        scenario = scenarios_test[i]
        file_path = paths_test[i]
        
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

    output_dir = MODEL_OUTPUT_DIR
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for scenario_name, recordings in sorted(nested_data.items()):
        num_recordings = len(recordings)
        fig, axes = plt.subplots(num_recordings, 1, figsize=(15, 4.2 * num_recordings), sharex=False)
        if num_recordings == 1:
            axes = [axes]
            
        for idx, (rec_name, windows) in enumerate(sorted(recordings.items())):
            ax = axes[idx]
            windows.sort(key=lambda x: x['time_sec'])
            times_sec = np.array([w['time_sec'] for w in windows])
            y_values = [w['true'] for w in windows]
            colors = [w['color'] for w in windows]
            
            ax.plot(times_sec, y_values, color='gray', linestyle='--', alpha=0.2, lw=1.0)
            for i in range(len(windows)):
                ax.scatter(times_sec[i], y_values[i], color=colors[i], s=80, edgecolors='black', linewidths=0.8, zorder=3)
            ax.set_title(f"🎵 {rec_name}", fontsize=10, loc='left')
            ax.set_yticks([0, 1])
            ax.set_ylim([-0.3, 1.45])
            
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"timeline_{scenario_name}.png"), dpi=150)
        plt.close(fig)

# ======================================================================
# 🛠️ CONFIGURATION - מילון הנתונים (כולל הפיצול של manatees)
# ======================================================================
DATA_DIRECTORIES = [
    {'name': '2026.04.28_omesi',  'path': r"/Users/deviceone/Documents/data/2026.04.28_omesi/raw_extracted_segments"},
    {'name': '2026.05.01_omesi',  'path': r"/Users/deviceone/Documents/data/2026.05.01_omesi/raw_extracted_segments"},
    {'name': 'dregon',             'path': r"/Users/deviceone/Documents/data/dregon/raw_extracted_segments"},
    {'name': 'nasa_2',             'path': r"/Users/deviceone/Documents/data/nasa_2/raw_extracted_segments"},
    {'name': 'tut',                'path': r"/Users/deviceone/Documents/data/tut/raw_extracted_segments"},
    {'name': 'ESC-50',             'path': r"/Users/deviceone/Documents/data/ESC-50/raw_extracted_segments"},
    {'name': 'kakadoo_train',      'path': r"/Users/deviceone/Documents/data/2026.06.09_kakadoo_SPLITTED/train_set"},
    {'name': 'manatees_train',     'path': r"/Users/deviceone/Documents/data/2026.06.07_manatees/SPLITTED/train_set"},
    {'name': 'swan', 'path': r"/Users/deviceone/Documents/data/2026.06.17_swan/SPLITTED/train_set"}
]

MODEL_OUTPUT_DIR = r"/Users/deviceone/Documents/d_detection/models"
binary_map = {0: ['raw_background'], 1: ['raw_drone']}

if __name__ == "__main__":
    X_bin, y_bin, file_paths, scenarios_bin, timestamps_bin = prepare_data_raw(DATA_DIRECTORIES, binary_map)
    unique_labels = np.unique(y_bin)
    
    if len(unique_labels) < 2:
        print("⚠️ Error: Dataset must contain both background and drone samples.")
    else:
        # 1. פיצול מבוסס קבצים (GroupSplit)
        gkf = GroupKFold(n_splits=5)
        train_idx, test_idx = next(gkf.split(X_bin, y_bin, groups=file_paths))
        
        X_train, X_test = X_bin[train_idx], X_bin[test_idx]
        y_train, y_test = y_bin[train_idx], y_bin[test_idx]
        paths_train, paths_test = file_paths[train_idx], file_paths[test_idx]
        scenarios_train, scenarios_test = scenarios_bin[train_idx], scenarios_bin[test_idx]
        timestamps_train, timestamps_test = timestamps_bin[train_idx], timestamps_bin[test_idx]
        
        # ======================================================================
        # ⚖️ חישוב דינמי של משקולת איזון המחלקות (Class Balancing)
        # ======================================================================
        count_background = np.sum(y_train == 0)
        count_drone = np.sum(y_train == 1)
        # הנוסחה הקלאסית למניעת False Alarms עקב חוסר איזון בדאטה
        calculated_weight = count_background / (count_drone + 1e-5)
        print(f"\n⚖️ DATA BALANCE ANALYSIS:")
        print(f"   -> Background Windows (0): {count_background}")
        print(f"   -> Drone Windows (1):      {count_drone}")
        print(f"   -> Auto-Calculated scale_pos_weight: {calculated_weight:.4f}\n")
        # ======================================================================
        
        best_params = {
            'colsample_bytree': 0.7, 
            'learning_rate': 0.05, 
            'max_depth': 4, 
            'n_estimators': 500, 
            'subsample': 0.7
        }

        # הזנת משקולת האיזון הדינמית לתוך ה-XGBoost
        model_bin = xgb.XGBClassifier(
            objective='binary:logistic', 
            eval_metric='logloss', 
            device="cpu", 
            scale_pos_weight=calculated_weight, # 🛠️ תיקון משקולות
            **best_params
        )
        
        print("--- Training Model on Raw Windowed Data... ---")
        model_bin.fit(X_train, y_train)
        
        # 2. החלת Threshold מחמיר (0.85 ומעלה מומלץ להורדת FA)
        chosen_threshold = 0.85  
        y_probs = model_bin.predict_proba(X_test)[:, 1]
        custom_preds = (y_probs >= chosen_threshold).astype(int)
        
        print_detailed_errors(y_test, custom_preds, show_matrix=True)
        analyze_model_errors(y_test, y_probs, custom_preds, scenarios_test, paths_test)
        plot_scenarios_timeline_by_recordings(y_test, y_probs, custom_preds, scenarios_test, paths_test, timestamps_test)

        if not os.path.exists(MODEL_OUTPUT_DIR):
            os.makedirs(MODEL_OUTPUT_DIR)
        model_out_path = os.path.join(MODEL_OUTPUT_DIR, "2s_model_omesi.pickle")
        save_trained_model_as_pickle(model_bin, model_out_path)