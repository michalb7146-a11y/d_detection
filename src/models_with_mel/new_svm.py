import os
import glob
import numpy as np
import librosa
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, ConfusionMatrixDisplay
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle

def save_trained_model_as_pickle(model, filename="2s_model_omesi.pickle"):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model saved successfully as {filename}")

def print_detailed_errors(y_test, preds, show_matrix=True):
    if not show_matrix: return
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
    plt.tight_layout()
    plt.show()

def plot_log_roc_curve(model, X_test, y_test, target_class_index=1):
    y_probs = model.predict_proba(X_test)[:, target_class_index]
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(10, 7))
    plt.semilogx(fpr, tpr, color='darkorange', lw=2, label=f'Log ROC (area = {roc_auc:.4f})')
    plt.semilogx(fpr, fpr, color='navy', lw=1, linestyle='--')
    plt.xlim([0.001, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (Log Scale)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('Logarithmic ROC Curve (Focus on High Rejection)')
    plt.legend(loc="lower right")
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.show()
    return fpr, tpr, thresholds

def extract_features_vectorized(file_path):
    y, sr = librosa.load(file_path, sr=16000, mono=True)
    y = y / (np.max(np.abs(y)) + 1e-5)
    features = []
    
    stft = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    chroma = librosa.feature.chroma_stft(S=stft, sr=sr)
    
    centroid = librosa.feature.spectral_centroid(S=stft, sr=sr)
    flatness = librosa.feature.spectral_flatness(S=stft)
    rolloff = librosa.feature.spectral_rolloff(S=stft, sr=sr, roll_percent=0.85)
    
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

def prepare_data(base_path, label_folder_map):
    X, y = [], []
    folder_to_label = {folder: label for label, folders in label_folder_map.items() for folder in folders}
    for folder, label in folder_to_label.items():
        folder_path = os.path.join(base_path, folder)
        files = glob.glob(os.path.join(folder_path, "*.wav"))
        print(f"Loading {len(files)} files for label: {label} (from {folder})")
        for f in tqdm(files):
            try:
                feat = extract_features_vectorized(f)
                X.append(feat)
                y.append(label)
            except Exception as e:
                print(f"Error processing {f}: {e}")
    return np.array(X), np.array(y)

def train_and_evaluate(X, y, title="Classifier Results", show_matrix=True):
    # פיצול ראשוני לטריין וטסט
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # פיצול פנימי של הטריין לטובת סט וולידציה עבור Early Stopping
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.1, random_state=42, stratify=y_train_full)
    
    # חישוב משקל לאיזון מחלקות (כמות רקע חלקי כמות רחפנים)
    num_bg = np.sum(y_train == 0)
    num_drone = np.sum(y_train == 1)
    scale_weight = num_bg / num_drone

    # הגדרת פרמטרים מתקדמים ומוגנים מרעשים
    best_params = {
        'colsample_bytree': 0.7, 
        'subsample': 0.8,
        'learning_rate': 0.03,        # הגמשה של קצב הלמידה לטובת דיוק מירבי
        'max_depth': 5,               # עומק עץ מאוזן שמונע Overfitting
        'n_estimators': 1500,         # מספר עצים גבוה בשילוב Early Stopping
        'scale_pos_weight': scale_weight, # איזון דינמי מובנה של הדאטה
        'reg_alpha': 0.1,             # L1 regularization
        'reg_lambda': 1.0             # L2 regularization
    }

    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        device="cpu",
        early_stopping_rounds=30,     # עצירה אוטומטית אם המודל מפסיק להשתפר
        **best_params
    )
    
    # אימון המודל תוך פיקוח על סט הוולידציה
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50                     # הדפסת התקדמות כל 50 עצים
    )
    
    preds = model.predict(X_test)
    print(f"\n--- {title} ---")
    print(classification_report(y_test, preds))
    print_detailed_errors(y_test, preds, show_matrix=show_matrix)
    if show_matrix: plot_confusion_matrix_graphic(y_test, preds)
    return model, X_test, y_test

# --- CONFIGURATION ---
BASE_DATA_DIR = r"/Users/deviceone/Downloads/new_balanced_2s_dataset_551_device_1"
binary_map = {0: ['background'], 1: ['target_drone']}

if __name__ == "__main__":
    X_bin, y_bin = prepare_data(BASE_DATA_DIR, binary_map)
    model_bin, x_test, y_test = train_and_evaluate(X_bin, y_bin, "Binary: Drone vs. Environment", show_matrix=True)

    model_dir = r"/Users/deviceone/Documents/d_detection/models"
    os.makedirs(model_dir, exist_ok=True)
    model_out_path = os.path.join(model_dir, "2s_model_omesi.pickle")
    save_trained_model_as_pickle(model_bin, model_out_path)
    
    fpr, tpr, thresholds = plot_log_roc_curve(model_bin, x_test, y_test, target_class_index=1)