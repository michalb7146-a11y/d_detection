import os
import glob
import pickle
import numpy as np
import pandas as pd
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from collections import defaultdict
from tqdm import tqdm

# ======================================================================
# 🛠️ GLOBAL CONFIGURATION
# ======================================================================
# NEW_TEST_DATA_DIR = r"/Users/deviceone/Documents/data/2026.05.06_nautilus_1_snake/raw_extracted_segments"
# NEW_TEST_DATA_DIR = r"/Users/deviceone/Documents/data/2026.06.09_kakadoo/raw_extracted_segments" 
NEW_TEST_DATA_DIR = r"/Users/deviceone/Documents/data/2026.06.07_manatees/raw_extracted_segments"
MODEL_PICKLE_PATH = r"/Users/deviceone/Documents/d_detection/models/2s_model.pickle"
MODEL_OUTPUT_DIR = r"/Users/deviceone/Documents/d_detection/models" 
CHOSEN_THRESHOLD = 0.8

BINARY_FOLDER_MAP = {
    0: ['raw_background'], 
    1: ['raw_drone']  
}

# ======================================================================
# 🧬 FEATURE EXTRACTION PIPELINE
# ======================================================================
def extract_windows_from_stft(file_path):
    """
    📌 FEATURE EXTRACTION VIA SLIDING WINDOW STFT
    -------------------------------------------------------------------------
    Extracts acoustic features from an audio file using Mel-Spectrogram and 
    MFCC analysis, designed to perfectly match the training pipeline.
    
    Key Processing Steps:
    1. Audio Loading: Loads audio at a fixed 16kHz sampling rate, enforcing mono channel.
    2. RMS Normalization: Computes Root Mean Square (RMS) energy and scales signal 
       amplitude to prevent False Alarms triggered by weak background environmental noise.
    3. Mel-Spectrogram: Computes power spectrogram with tuned frequency boundaries 
       (fmin=150Hz, fmax=2000Hz) to isolate target signatures.
    4. Feature Extraction & Windowing: Extracts 13 MFCC coefficients over a 2-second 
       sliding window with a 25% overlap (stride).
    5. Output: Returns a flattened numpy array containing the Mean and Standard Deviation 
       of both Mel and MFCC parameters (exactly 154 features per window).
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
        n_mels=n_mels, fmin=150, fmax=2000
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

def load_new_test_data(base_path, label_folder_map):
    """
    Loads test datasets across target labels, extracts processing windows dynamically,
    and returns arrays of features, true categories, paths, and exact sample timestamps.
    """
    X, y, file_paths, scenarios, timestamps = [], [], [], [], []
    window_stride_seconds = (int(((2 * 16000) / 512) * 0.25) * 512) / 16000 

    print(f"\n--- Scanning Directory for New Test Set: {base_path} ---")
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
                
            files = glob.glob(os.path.join(folder_path, "*.wav")) + glob.glob(os.path.join(folder_path, "*.WAV"))
            if len(files) == 0:
                print(f"⚠️ Warning: No .wav or .WAV files found inside: {folder_path}")
                continue
                
            print(f"🚀 Found {len(files)} files in '{folder}'. Extracting features...")
            for f in tqdm(files):
                try:
                    feats_list = extract_windows_from_stft(f)
                    for win_idx, feat in enumerate(feats_list):
                        X.append(feat)
                        y.append(label)
                        file_paths.append(f)
                        scenarios.append("New_Evaluation_Set")
                        timestamps.append(win_idx * window_stride_seconds)
                except Exception as e:
                    print(f"❌ Error processing file {f}: {e}")
                    
    return np.array(X), np.array(y), np.array(file_paths), np.array(scenarios), np.array(timestamps)

# ======================================================================
# 📊 EVALUATION METRICS & GRAPHICAL VISUALIZATIONS
# ======================================================================
def print_detailed_errors(y_test, preds):
    """
    Calculates and prints absolute counts and percentages for individual
    prediction outcomes (TN, FP, FN, TP) at a frame level.
    """
    cm = confusion_matrix(y_test, preds, labels=[0, 1])
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
        total_bg = tn + fp if (tn + fp) > 0 else 1
        total_drone = fn + tp if (fn + tp) > 0 else 1
        
        print("\n--- DETAILED ERROR ANALYSIS (Confusion Matrix) ---")
        print(f"OK True Negatives  (Correct Background): {tn:<5} ({ (tn/total_bg)*100 :.2f}%)")
        print(f"XX False Positives (False Alarms):       {fp:<5} ({ (fp/total_bg)*100 :.2f}%)")
        print(f"XX False Negatives (Missed Detections):  {fn:<5} ({ (fn/total_drone)*100 :.2f}%)")
        print(f"OK True Positives  (Correct Drone):     {tp:<5} ({ (tp/total_drone)*100 :.2f}%)")
        print("-" * 50 + "\n")

def plot_dual_confusion_matrices(y_test, preds, total_true_events, detected_events, false_alarms):
    """
    📊 DUAL FRAME-LEVEL & EVENT-LEVEL VISUALIZER (IDENTICAL AXIS LABELS)
    -------------------------------------------------------------------------
    Generates a side-by-side graphical visualization layout using Seaborn and Matplotlib 
    to compare frame-level accuracy against real macro event-level performance.
    Both matrices now share the exact same 'Background' and 'Drone' axis structures.
    """
    # 1. Frame-Level Matrix Calculations
    cm_frames = confusion_matrix(y_test, preds, labels=[0, 1])
    row_sums = cm_frames.sum(axis=1)[:, np.newaxis]
    row_sums = np.where(row_sums == 0, 1, row_sums)
    cm_frames_perc = (cm_frames.astype('float') / row_sums) * 100
    
    # 2. Event-Level Matrix Assembly (Mapped exactly to Background [0] and Drone [1] axes)
    missed_events = total_true_events - detected_events
    cm_events = np.array([
        [0, false_alarms],       # True Background row: [TN (N/A), FP (False Alarms)]
        [missed_events, detected_events] # True Drone row:      [FN (Missed),  TP (Detected)]
    ])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # ------------------------------------------------------------------
    # Plot Left: Frame-Level Matrix
    # ------------------------------------------------------------------
    labels_frames = [
        f"{cm_frames[0,0]}\n({cm_frames_perc[0,0]:.1f}%)", f"{cm_frames[0,1]}\n({cm_frames_perc[0,1]:.1f}%)",
        f"{cm_frames[1,0]}\n({cm_frames_perc[1,0]:.1f}%)", f"{cm_frames[1,1]}\n({cm_frames_perc[1,1]:.1f}%)"
    ]
    labels_frames = np.asarray(labels_frames).reshape(2,2)
    
    sns.heatmap(cm_frames_perc, annot=labels_frames, fmt="", cmap="Blues", cbar=False,
                xticklabels=["Background", "Drone"], yticklabels=["Background", "Drone"], ax=axes[0], 
                annot_kws={"size": 11, "weight": "bold"})
    axes[0].set_title("Frame-Level Confusion Matrix\n(Normalized % per True class)", fontsize=12, fontweight='bold', pad=10)
    axes[0].set_xlabel("Predicted Labels", fontsize=10, fontweight='bold')
    axes[0].set_ylabel("True Labels", fontsize=10, fontweight='bold')

    # ------------------------------------------------------------------
    # Plot Right: Event-Level Matrix (Now identical with Background/Drone axes)
    # ------------------------------------------------------------------
    labels_events = [
        ["N/A", f"{false_alarms}\n(FA)"],
        [f"{missed_events}\n(Missed)", f"{detected_events}\n(Detected)"]
    ]
    labels_events = np.asarray(labels_events).reshape(2,2)
    
    sns.heatmap(cm_events, annot=labels_events, fmt="", cmap="Greens", cbar=False,
                xticklabels=["Background", "Drone"], yticklabels=["Background", "Drone"], ax=axes[1], 
                annot_kws={"size": 11, "weight": "bold"})
    axes[1].set_title("Event-Level Confusion Matrix\n(Absolute Event Counts)", fontsize=12, fontweight='bold', pad=10)
    axes[1].set_xlabel("Predicted Labels", fontsize=10, fontweight='bold')
    axes[1].set_ylabel("True Labels", fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.show()

def plot_log_roc_curve(model, X_test, y_test):
    """
    Plots logarithmic ROC curves focusing validation on low false-positive constraints.
    """
    if len(np.unique(y_test)) < 2:
        print("⚠️ Skipping ROC Curve plot: Both background and drone classes are required.")
        return
        
    y_probs = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_probs)
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

def plot_scenarios_timeline_by_recordings(y_test, y_probs, custom_preds, scenarios_test, paths_test, timestamps_test, threshold, output_dir):
    """
    Plots an interactive chronological timeline chart mapped to Audacity minute-second tracks,
    highlighting correct windows, misses, and annotations for False Alarm instances.
    """
    nested_data = defaultdict(lambda: defaultdict(list))
    for i in range(len(y_test)):
        rec_name = os.path.basename(paths_test[i])
        lbl_txt = 'Correct Drone' if y_test[i] == 1 and custom_preds[i] == 1 else \
                  'Correct Background' if y_test[i] == 0 and custom_preds[i] == 0 else \
                  'False Alarm (FA)' if y_test[i] == 0 and custom_preds[i] == 1 else 'Missed Detection (MD)'
        col = '#2ca02c' if lbl_txt == 'Correct Drone' else '#98df8a' if lbl_txt == 'Correct Background' else \
              '#d62728' if lbl_txt == 'False Alarm (FA)' else '#ff7f0e'
              
        nested_data[scenarios_test[i]][rec_name].append({
            'true': y_test[i], 'color': col, 'label_text': lbl_txt, 'time_sec': timestamps_test[i], 'prob': y_probs[i]
        })

    for scenario_name, recordings in sorted(nested_data.items()):
        num_recs = len(recordings)
        if num_recs == 0: continue
        
        fig, axes = plt.subplots(num_recs, 1, figsize=(15, 4.2 * num_recs))
        if num_recs == 1: axes = [axes]
        fig.suptitle(f"🎬 EXPERIMENT TIMELINE: {scenario_name}", fontsize=14, fontweight='bold', y=0.98)
        
        for idx, (rec_name, windows) in enumerate(sorted(recordings.items())):
            ax = axes[idx]
            windows.sort(key=lambda x: x['time_sec'])
            
            times_sec = np.array([w['time_sec'] for w in windows])
            y_values = [w['true'] for w in windows]
            colors = [w['color'] for w in windows]
            probs = [w['prob'] for w in windows]
            
            ax.plot(times_sec, y_values, color='gray', linestyle='--', alpha=0.2, lw=1.0)
            ax.plot(times_sec, probs, color='blue', alpha=0.4, linewidth=1.2, label='Confidence Trend')
            ax.axhline(y=threshold, color='red', linestyle=':', alpha=0.6, linewidth=1.5, label=f'Threshold ({threshold})')
            
            visible_labels = set()
            for i in range(len(windows)):
                lbl = windows[i]['label_text']
                curr_t = times_sec[i]
                s_lbl = lbl if lbl not in visible_labels else "_" + lbl
                visible_labels.add(lbl)
                
                ax.scatter(curr_t, y_values[i], color=colors[i], s=80, edgecolors='black', linewidths=0.8, zorder=3, label=s_lbl)
                if colors[i] == '#d62728':
                    ax.annotate(f"{int(curr_t // 60)}:{curr_t % 60:05.2f}", xy=(curr_t, y_values[i]), xytext=(0, 15), 
                                textcoords='offset points', ha='center', va='bottom', fontsize=8, color='darkred', fontweight='bold',
                                bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.7, ec='red', lw=0.5),
                                arrowprops=dict(arrowstyle="->", color='red', lw=0.5))
            
            ax.set_title(f"🎵 Recording File: {rec_name}", fontsize=10, fontweight='bold', color='navy', loc='left')
            ax.set_yticks([0, 1])
            ax.set_yticklabels(['Background (0)', 'Drone (1)'], fontsize=9, fontweight='bold')
            ax.set_xlabel("Real Audio Time (Seconds)", fontsize=9)
            if len(times_sec) > 0: ax.set_xlim([-10, max(times_sec) + 30])
            ax.set_ylim([-0.3, 1.55])
            ax.grid(True, linestyle=':', alpha=0.5)
            if idx == 0: ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1.2), fontsize=8, fancybox=True, shadow=True)
                
        plt.tight_layout()
        plt.subplots_adjust(top=0.92) 
        save_path = os.path.join(output_dir, f"new_test_timeline_{scenario_name}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)  
        print(f"  🎯 Saved Image for [{scenario_name}] -> {save_path}")

# ======================================================================
# 📊📊 MACRO EVENT-LEVEL EVALUATION PARSER
# ======================================================================
def time_str_to_seconds(time_str):
    """
    Converts a timestamp string (MM:SS.mmm, HH:MM:SS.mmm, or raw float strings) 
    into absolute floating-point seconds.
    """
    if pd.isna(time_str): return None
    try:
        time_str = str(time_str).strip()
        parts = time_str.split(':')
        if len(parts) == 2: return float(parts[0]) * 60 + float(parts[1])
        elif len(parts) == 3: return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
        return float(time_str)
    except: return None

def calculate_event_level_metrics(paths_new, timestamps_new, custom_preds, y_new, base_test_dir):
    """
    📌 DYNAMIC EVENT-LEVEL EVALUATION METRICS
    -------------------------------------------------------------------------
    Evaluates continuous model predictions against physical, macro-level events 
    instead of localized micro-level frames.
    """
    VALID_DRONE_KEYWORDS = ['drone', 'רחפן', 'fly', 'flies', 'מעוף']
    parent_dir = os.path.dirname(base_test_dir)
    tagged_folders = [f for f in os.listdir(parent_dir) if f.lower().startswith("tagged") and os.path.isdir(os.path.join(parent_dir, f))]
    
    if not tagged_folders:
        print("\n⚠️ Warning: Sibling directory starting with 'tagged' not found. Event metrics skipped.")
        return None
    
    tagged_dir = os.path.join(parent_dir, tagged_folders[0])
    print(f"\n📂 Target tagging directory identified: {tagged_dir}")

    # 🔄 שינוי: חיפוש רקורסיבי בעזרת os.walk כדי למצוא קבצים גם בתוך תתי-תיקיות
    label_files = []
    valid_extensions = ('.csv', '.tsv', '.txt')
    for root, _, files in os.walk(tagged_dir):
        for file in files:
            if file.lower().endswith(valid_extensions):
                label_files.append(os.path.join(root, file))

    if len(label_files) == 0: return None

    valid_files = [f for f in label_files if os.path.exists(f) and os.path.getsize(f) > 0]
    print(f"🔎 Found {len(label_files)} total label files ({len(valid_files)} contain data)...")
    
    all_tagged_events = []
    for file_path in valid_files:
        for encoding in ['utf-8', 'utf-8-sig', 'cp1255', 'latin-1']:
            try:
                is_tsv = file_path.lower().endswith('.tsv') or file_path.lower().endswith('.txt')
                default_sep = '\t' if is_tsv else ','
                try: df = pd.read_csv(file_path, encoding=encoding, sep=None, engine='python')
                except: df = pd.read_csv(file_path, encoding=encoding, sep=default_sep, engine='python')
                
                if df.empty: continue
                df.columns = [col.strip() for col in df.columns]
                
                label_col = next((c for c in df.columns if 'name' in c.lower() or 'type' in c.lower() or 'label' in c.lower()), df.columns[0])
                start_col = next((c for c in df.columns if 'start' in c.lower()), None)
                duration_col = next((c for c in df.columns if 'duration' in c.lower()), None)
                if not start_col or not duration_col: continue  
                
                rows_counted = 0
                for _, row in df.iterrows():
                    current_label = str(row[label_col]).strip().lower() if not pd.isna(row[label_col]) else ""
                    start_sec = time_str_to_seconds(row[start_col])
                    dur_sec = time_str_to_seconds(row[duration_col])
                    if start_sec is None: continue
                    
                    end_sec = start_sec + (dur_sec if (dur_sec and dur_sec > 0) else 4.0)
                    is_drone = any(keyword in current_label for keyword in VALID_DRONE_KEYWORDS)
                    rows_counted += 1
                    
                    all_tagged_events.append({
                        'start': start_sec, 'end': end_sec, 'is_drone': is_drone, 'label_text': current_label, 'detected': False
                    })
                if rows_counted > 0: break  
            except: continue

    actual_drone_events = [e for e in all_tagged_events if e['is_drone']]
    total_true_drone_events = len(actual_drone_events)
    print(f"📊 Debug Trace: Parsed {len(all_tagged_events)} total rows ({total_true_drone_events} drone tracks).")
    
    if total_true_drone_events == 0: return None

    detected_drone_events = 0
    false_alarm_events = 0
    in_fa_sequence = False

    for i in range(len(paths_new)):
        current_time = timestamps_new[i]
        current_pred = custom_preds[i]
        is_time_drone_in_csv = False
        
        for event in actual_drone_events:
            if event['start'] <= current_time <= event['end']:
                is_time_drone_in_csv = True
                if current_pred == 1 and not event['detected']:
                    event['detected'] = True
                    detected_drone_events += 1
                break
        
        if current_pred == 1 and not is_time_drone_in_csv:
            if not in_fa_sequence: in_fa_sequence = True
        else:
            if in_fa_sequence:
                false_alarm_events += 1
                in_fa_sequence = False
    if in_fa_sequence: false_alarm_events += 1

    missed_drone_events = total_true_drone_events - detected_drone_events
    print("\n" + "="*60 + "\n🎯 EVENT-LEVEL PERFORMANCE REPORT\n" + "="*60)
    print(f"Total Actual Drone Events (Parsed):  {total_true_drone_events}")
    print(f"Successfully Detected Events (TP):   {detected_drone_events} ({ (detected_drone_events/max(1, total_true_drone_events))*100 :.2f}%)")
    print(f"Missed Detection Events (MD):        {missed_drone_events} ({ (missed_drone_events/max(1, total_true_drone_events))*100 :.2f}%)")
    print(f"False Alarm Events (FA Continuous):  {false_alarm_events}\n" + "="*60 + "\n")
    
    return total_true_drone_events, detected_drone_events, false_alarm_events

# ======================================================================
# 🚀 MAIN RUNTIME EXECUTION
# ======================================================================
if __name__ == "__main__":
    print(f"🔄 Loading trained model from: {MODEL_PICKLE_PATH}")
    if not os.path.exists(MODEL_PICKLE_PATH):
        raise FileNotFoundError(f"Could not find pickle file at {MODEL_PICKLE_PATH}")
        
    with open(MODEL_PICKLE_PATH, 'rb') as file:
        loaded_model = pickle.load(file)
    print("✅ Model loaded successfully.")

    X_new, y_new, paths_new, scenarios_new, timestamps_new = load_new_test_data(NEW_TEST_DATA_DIR, BINARY_FOLDER_MAP)
    
    if len(X_new) == 0:
        print("❌ No data was extracted. Please verify directory paths and schemas.")
    else:
        print(f"\n✅ Extraction complete. Total windows for inference evaluation: {len(X_new)}")
        y_probs = loaded_model.predict_proba(X_new)[:, 1]
        
        # Apply 2-second moving window probability smoothing (8 windows lookback)
        smoothed_probs = np.zeros_like(y_probs)
        for i in range(len(y_probs)):
            smoothed_probs[i] = np.mean(y_probs[max(0, i - 8):i + 1])
            
        custom_preds = (smoothed_probs >= CHOSEN_THRESHOLD).astype(int)
        
        print(f"\n--- Performance Evaluation (Threshold = {CHOSEN_THRESHOLD}) ---")
        print(classification_report(y_new, custom_preds, labels=[0, 1], target_names=['Background', 'Drone'], zero_division=0))
        
        print_detailed_errors(y_new, custom_preds)
        
        # Execute Event-level statistics extraction
        event_stats = calculate_event_level_metrics(paths_new, timestamps_new, custom_preds, y_new, NEW_TEST_DATA_DIR)
        
        # Generate Confusion Matrices visually
        if event_stats is not None:
            total_events, det_events, fa_events = event_stats
            print("📈 Displaying Dual Frame-Level and Event-Level Confusion Matrices...")
            plot_dual_confusion_matrices(y_new, custom_preds, total_events, det_events, fa_events)
        
        print("📈 Plotting Log ROC Curve...")
        plot_log_roc_curve(loaded_model, X_new, y_new)
        
        print("📈 Saving experiment timelines imagery tracks...")
        plot_scenarios_timeline_by_recordings(
            y_test=y_new, y_probs=y_probs, custom_preds=custom_preds, scenarios_test=scenarios_new, 
            paths_test=paths_new, timestamps_test=timestamps_new, threshold=CHOSEN_THRESHOLD, output_dir=MODEL_OUTPUT_DIR
        )