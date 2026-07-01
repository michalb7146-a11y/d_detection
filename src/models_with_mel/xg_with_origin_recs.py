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
    📌 SLIDING WINDOW STFT FEATURE EXTRACTION:
    טוען קובץ ארוך, מחשב STFT מלא, ומפרק לחלונות של 2 שניות עם חפיפה של 75% בזיכרון.
    """
    target_sr = 16000
    y, sr = librosa.load(file_path, sr=target_sr, mono=True)
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

def prepare_data_raw(base_paths_config, label_folder_map):
    """
    טוענת קבצים בהתאם למבנה המילונים החדש (name ו-path).
    שומרת את זמן תחילת החלון האמיתי בשניות (timestamps) מתחילת הקובץ המקורי.
    """
    X, y, file_paths, sample_scenarios, timestamps = [], [], [], [], []
    
    # חישוב קפיצת הזמן האמיתית בשניות בין חלון לחלון (Stride)
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
                    
                files = glob.glob(os.path.join(folder_path, "*.wav"))
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
                            
                            # שמירת הזמן האמיתי בשניות של החלון הנוכחי מתחילת הקובץ
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

def plot_scenarios_timeline_by_recordings(y_test, y_probs, custom_preds, scenarios_test, paths_test, timestamps_test):
    """
    מייצרת ומצילה תמונה (Figure) נפרדת עבור כל שם ניסוי מוגדר במילון (name).
    מציגה את נקודות הזמן האמיתיות (מ-0 ועד הטווח המלא של הקובץ, למשל 6:30 דקות)
    ומסמנת FA בחלוניות צהובות מעוצבות בפורמט של דקות:שניות.
    """
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
            'time_sec': timestamps_test[i] # שימוש בזמן האמיתי של החלון בקובץ
        })

    output_dir = MODEL_OUTPUT_DIR
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("\n" + "="*80)
    print("🎯 LOGGING FALSE ALARM TIMESTAMPS FOR AUDACITY INSPECTION (MATCHED 100%):")
    print("="*80)

    for scenario_name, recordings in sorted(nested_data.items()):
        num_recordings = len(recordings)
        
        fig, axes = plt.subplots(num_recordings, 1, figsize=(15, 4.2 * num_recordings), sharex=False)
        if num_recordings == 1:
            axes = [axes]
            
        fig.suptitle(f"🎬 EXPERIMENT TIMELINE: {scenario_name}", fontsize=14, fontweight='bold', y=0.98)
        
        for idx, (rec_name, windows) in enumerate(sorted(recordings.items())):
            ax = axes[idx]
            
            # מיון כרונולוגי לפי הזמן האמיתי של החלונות בקובץ
            windows.sort(key=lambda x: x['time_sec'])
            
            times_sec = np.array([w['time_sec'] for w in windows])
            y_values = [w['true'] for w in windows]
            colors = [w['color'] for w in windows]
            
            # קו רשת דק ברקע של הזרימה
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
                
                # אם זו נקודת False Alarm (אדומה), נוסיף כיתוב זמן בפורמט דקות:שניות לגרף ולטרמינל
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
            
            # פרישת ציר ה-X לפי אורך האודיו האמיתי המלא בקובץ (עם מרווח ביטחון קטן)
            if len(times_sec) > 0:
                ax.set_xlim([-10, max(times_sec) + 30])
                
            ax.set_ylim([-0.3, 1.45])
            ax.grid(True, linestyle=':', alpha=0.5)
            
            if idx == 0:
                ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1.2), fontsize=8, fancybox=True, shadow=True)
                
        plt.tight_layout()
        plt.subplots_adjust(top=0.92) 
        
        save_path = os.path.join(output_dir, f"timeline_{scenario_name}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)  
        
        print(f"  🎯 Saved Image for [{scenario_name}] -> {save_path}")
        
    print("\n" + "="*80 + "\n")

def plot_feature_importance_debug(model, feature_names=None):
    importances = model.feature_importances_
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(len(importances))]
        
    feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)
    
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
# 🛠️ CONFIGURATION - מילון נתונים (רשימת מילונים)
# ======================================================================
DATA_DIRECTORIES = [
    {'name': '2026.04.28_omesi', 'path': r"/Users/deviceone/Documents/data/2026.04.28_omesi/raw_extracted_segments"},
    {'name': '2026.05.01_omesi', 'path': r"/Users/deviceone/Documents/data/2026.05.01_omesi/raw_extracted_segments"},
    {'name': 'dregon',            'path': r"/Users/deviceone/Documents/data/dregon/raw_extracted_segments"},
    {'name': 'nasa_2',            'path': r"/Users/deviceone/Documents/data/nasa_2/raw_extracted_segments"},
    {'name': 'tut',               'path': r"/Users/deviceone/Documents/data/tut/raw_extracted_segments"},
    {'name': 'ESC-50',            'path': r"/Users/deviceone/Documents/data/ESC-50/raw_extracted_segments"}
]

MODEL_OUTPUT_DIR = r"/Users/deviceone/Documents/d_detection/models"

binary_map = {
    0: ['raw_background'], 
    1: ['raw_drone']  
}

if __name__ == "__main__":
    # 1. טעינת הנתונים וקבלת ה-timestamps_bin האמיתיים
    X_bin, y_bin, file_paths, scenarios_bin, timestamps_bin = prepare_data_raw(DATA_DIRECTORIES, binary_map)
    unique_labels = np.unique(y_bin)
    
    if len(unique_labels) < 2:
        print("⚠️ Error: Dataset must contain both background and drone samples to train.")
    else:
        print(f"\n✅ Data generation complete. Total 2-second windows extracted: {len(X_bin)}")
        
        # 2. פיצול הנתונים תוך שמירה על סנכרון הזמנים (timestamps) ל-Test Set
        X_train, X_test, y_train, y_test, paths_train, paths_test, scenarios_train, scenarios_test, timestamps_train, timestamps_test = train_test_split(
            X_bin, y_bin, file_paths, scenarios_bin, timestamps_bin, test_size=0.2, random_state=42, stratify=y_bin
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
        
        print("\n--- Training Model on Raw Windowed Data... ---")
        model_bin.fit(X_train, y_train)
        
        preds = model_bin.predict(X_test)
        print("\n--- Baseline Classifier Results (Threshold = 0.5) ---")
        print(classification_report(y_test, preds))
        
        fpr, tpr, thresholds, recommended_threshold = plot_log_roc_curve(model_bin, X_test, y_test, target_class_index=1)
        
        chosen_threshold = 0.8  

        print(f"📊 Applying Custom Threshold ({chosen_threshold:.4f}) to Test Data...")
        y_probs = model_bin.predict_proba(X_test)[:, 1]
        custom_preds = (y_probs >= chosen_threshold).astype(int)
        
        print_detailed_errors(y_test, custom_preds, show_matrix=True)
        plot_confusion_matrix_graphic(y_test, custom_preds)

        analyze_model_errors(y_test, y_probs, custom_preds, scenarios_test, paths_test)

        # 3. ציור גרפים מיושרים לחלוטין לזמנים של Audacity
        print("📈 Plotting timelines matched perfectly to Audacity timelines (Minutes:Seconds)...")
        plot_scenarios_timeline_by_recordings(y_test, y_probs, custom_preds, scenarios_test, paths_test, timestamps_test)

        # בניית שמות הפיצ'רים לגרף החשיבות
        stft_len = 1025     
        feature_names = []
        feature_names += [f"STFT_Mean_{i}" for i in range(stft_len)]
        feature_names += [f"STFT_Std_{i}" for i in range(stft_len)]
        feature_names += [f"MFCC_Mean_{i}" for i in range(5)]
        feature_names += [f"MFCC_Std_{i}" for i in range(5)]
        feature_names += ["Ratio_Mid_Low_Energy", "Ratio_Extreme_Low_Energy"]
        feature_names += ["Phase_Stability_Low", "Phase_Stability_Mid", "Phase_Stability_High"]

        plot_feature_importance_debug(model_bin, feature_names=feature_names)

        if not os.path.exists(MODEL_OUTPUT_DIR):
            os.makedirs(MODEL_OUTPUT_DIR)
        model_out_path = os.path.join(MODEL_OUTPUT_DIR, "2s_model_omesi.pickle")
        save_trained_model_as_pickle(model_bin, model_out_path)