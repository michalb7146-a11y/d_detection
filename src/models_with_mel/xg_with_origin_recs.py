import os
import glob
import numpy as np
import librosa
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import pickle
from collections import defaultdict

def save_trained_model_as_pickle(model, filename="2s_model_omesi.pickle"):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model saved successfully as {filename}")

def extract_windows_from_stft(file_path):
    """
    📌 FREQUENCY FILTERING BACK TO 2000Hz - BASED ON SPECTROGRAM ANALYSIS
    """
    target_sr = 16000
    y, sr = librosa.load(file_path, sr=target_sr, mono=False)
    
    if len(y.shape) > 1 and y.shape[0] > 1:
        y = y[0]
        
    rms = np.sqrt(np.mean(y**2)) + 1e-5
    y = (y / rms) * 0.1 
    
    n_fft = 2048
    hop_length = 512
    n_mels = 64
    
    mel_spec_power = librosa.feature.melspectrogram(
        y=y, sr=target_sr, n_fft=n_fft, hop_length=hop_length, 
        n_mels=n_mels, fmax=2000
    )
    mel_spec_db = librosa.power_to_db(mel_spec_power + 1e-5)
    
    mfccs = librosa.feature.mfcc(S=mel_spec_db, sr=target_sr, n_mfcc=13)
    
    frames_per_2s = int((2 * target_sr) / hop_length) 
    step_size = int(frames_per_2s * 0.25) 
    
    total_frames = mel_spec_db.shape[1]
    window_features = []
    
    if total_frames < frames_per_2s:
        return window_features

    for start_frame in range(0, total_frames - frames_per_2s + 1, step_size):
        end_frame = start_frame + frames_per_2s
        
        mel_window = mel_spec_db[:, start_frame:end_frame]
        mfcc_window = mfccs[:, start_frame:end_frame]
        
        mel_mean = np.mean(mel_window, axis=1)
        mel_std = np.std(mel_window, axis=1)
        
        mfcc_mean = np.mean(mfcc_window, axis=1)
        mfcc_std = np.std(mfcc_window, axis=1)
        
        flat_features = np.concatenate([
            mel_mean, mel_std,
            mfcc_mean, mfcc_std
        ])
        
        window_features.append(flat_features)
        
    return window_features

def prepare_data_raw(base_paths_config, label_folder_map):
    X, y, file_paths, sample_scenarios, timestamps = [], [], [], [], []
    window_stride_seconds = (int(((2 * 16000) / 512) * 0.25) * 512) / 16000 

    for item in base_paths_config:
        base_path = item['path']
        dataset_name = item['name']
        print(f"Scanning Directory for [{dataset_name}]")
        
        for label, folders in label_folder_map.items():
            for folder in folders:
                folder_path = os.path.join(base_path, folder)
                if not os.path.exists(folder_path):
                    continue
                    
                files = glob.glob(os.path.join(folder_path, "*.wav")) + glob.glob(os.path.join(folder_path, "*.WAV"))
                for f in files:
                    try:
                        feats_list = extract_windows_from_stft(f)
                        for win_idx, feat in enumerate(feats_list):
                            X.append(feat)
                            y.append(label)
                            file_paths.append(f)
                            sample_scenarios.append(dataset_name)
                            timestamps.append(win_idx * window_stride_seconds)
                    except Exception as e:
                        pass
                        
    return np.array(X), np.array(y), np.array(file_paths), np.array(sample_scenarios), np.array(timestamps)

def analyze_model_errors(y_test, custom_preds, scenarios_test):
    print("\n" + "="*85)
    print("🔍 SCENARIO BREAKDOWN TABLE")
    print("="*85)

    scenario_stats = defaultdict(lambda: {'total_bg': 0, 'total_drone': 0, 'fa_count': 0, 'md_count': 0})

    for i in range(len(y_test)):
        true_label = y_test[i]
        pred_label = custom_preds[i]
        scenario = scenarios_test[i]
        
        if true_label == 0:
            scenario_stats[scenario]['total_bg'] += 1
            if pred_label == 1:
                scenario_stats[scenario]['fa_count'] += 1
        elif true_label == 1:
            scenario_stats[scenario]['total_drone'] += 1
            if pred_label == 0:
                scenario_stats[scenario]['md_count'] += 1

    scen_w, bg_w, fa_w, dr_w, md_w = 25, 12, 20, 12, 20
    header = f"{'📋 DATASET (SCENARIO)':<{scen_w}} | {'TOTAL BG':<{bg_w}} | {'FALSE ALARMS (FA)':<{fa_w}} | {'TOTAL DRONE':<{dr_w}} | {'MISSED DET. (MD)':<{md_w}}"
    print(header)
    print("-" * len(header))
    
    for scenario, stats in sorted(scenario_stats.items()):
        tot_bg = stats['total_bg']
        tot_dr = stats['total_drone']
        fa = stats['fa_count']
        md = stats['md_count']
        
        fa_str = f"{fa:<4} ({ (fa/tot_bg)*100 if tot_bg>0 else 0 :.2f}%)"
        md_str = f"{md:<4} ({ (md/tot_dr)*100 if tot_dr>0 else 0 :.2f}%)"
        
        print(f"{scenario:<{scen_w}} | {tot_bg:<{bg_w}} | {fa_str:<{fa_w}} | {tot_dr:<{dr_w}} | {md_str:<{md_w}}")
    print("=" * len(header) + "\n")

# DATA CONFIG
DATA_DIRECTORIES = [
    {'name': '2026.04.28_omesi',   'path': r"/Users/deviceone/Documents/data/2026.04.28_omesi/raw_extracted_segments"},
    {'name': '2026.05.01_omesi',   'path': r"/Users/deviceone/Documents/data/2026.05.01_omesi/raw_extracted_segments"},
    {'name': 'dregon',             'path': r"/Users/deviceone/Documents/data/dregon/raw_extracted_segments"},
    {'name': 'nasa_2',             'path': r"/Users/deviceone/Documents/data/nasa_2/raw_extracted_segments"},
    {'name': 'tut',                'path': r"/Users/deviceone/Documents/data/tut/raw_extracted_segments"},
    {'name': 'ESC-50',             'path': r"/Users/deviceone/Documents/data/ESC-50/raw_extracted_segments"},
    {'name': 'kakadoo_train',      'path': r"/Users/deviceone/Documents/data/2026.06.09_kakadoo_SPLITTED/train_set"},
    {'name': 'manatees_train',     'path': r"/Users/deviceone/Documents/data/2026.06.07_manatees/SPLITTED/train_set"},
    {'name': 'swan',               'path': r"/Users/deviceone/Documents/data/2026.06.17_swan/SPLITTED/train_set"}
]
MODEL_OUTPUT_DIR = r"/Users/deviceone/Documents/d_detection/models"
binary_map = {0: ['raw_background'], 1: ['raw_drone']}

if __name__ == "__main__":
    X_bin, y_bin, file_paths, scenarios_bin, timestamps_bin = prepare_data_raw(DATA_DIRECTORIES, binary_map)
    
    print("\n📦 Splitting files balanced per dataset (Group-Stratified Split)...")
    train_idx, test_idx = [], []
    
    # חלוקה מבוקרת: מבטיחים חלוקת 80-20 של קבצים שלמים בתוך כל סצינה בנפרד
    unique_scenarios = np.unique(scenarios_bin)
    for scenario in unique_scenarios:
        scen_mask = (scenarios_bin == scenario)
        scen_files = np.unique(file_paths[scen_mask])
        
        if len(scen_files) >= 2:
            train_files, test_files = train_test_split(scen_files, test_size=0.20, random_state=42)
        else:
            train_files, test_files = scen_files, scen_files  # קבצי גיבוי במקרה קיצון של קובץ בודד
            
        indices = np.where(scen_mask)[0]
        for idx in indices:
            if file_paths[idx] in train_files:
                train_idx.append(idx)
            if file_paths[idx] in test_files:
                test_idx.append(idx)
                
    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)
    
    X_train, X_test = X_bin[train_idx], X_bin[test_idx]
    y_train, y_test = y_bin[train_idx], y_bin[test_idx]
    scenarios_test = scenarios_bin[test_idx]
    
    calculated_weight = np.sum(y_train == 0) / (np.sum(y_train == 1) + 1e-5)
    
    model_bin = xgb.XGBClassifier(
        objective='binary:logistic', eval_metric='logloss', device="cpu",
        scale_pos_weight=calculated_weight,
        colsample_bytree=0.7, learning_rate=0.05, max_depth=4, n_estimators=500, subsample=0.7
    )
    
    print(f"--- Training Robust Model (fmax=2000Hz) on {len(X_train)} windows... ---")
    model_bin.fit(X_train, y_train)
    
    # נקודת איזון יציבה להערכה
    chosen_threshold = 0.55
    y_probs = model_bin.predict_proba(X_test)[:, 1]
    
    # הפעלת סימולציית רילטיים (Causal Smoothing)
    smoothed_probs = np.zeros_like(y_probs)
    for i in range(len(y_probs)):
        smoothed_probs[i] = np.mean(y_probs[max(0, i - 3):i + 1])
        
    custom_preds = (smoothed_probs >= chosen_threshold).astype(int)
    
    analyze_model_errors(y_test, custom_preds, scenarios_test)
    save_trained_model_as_pickle(model_bin, os.path.join(MODEL_OUTPUT_DIR, "2s_model_omesi.pickle"))