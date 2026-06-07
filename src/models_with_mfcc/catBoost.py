import os
import glob
import numpy as np
import librosa
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay
from tqdm import tqdm
import matplotlib.pyplot as plt
import joblib

def save_trained_model(model, filename):
    joblib.dump(model, filename)
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

def plot_confusion_matrix_graphic(y_test, preds, model_name="CatBoost"):
    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Background', 'Drone'])
    disp.plot(cmap=plt.cm.Blues, values_format='d', ax=ax, colorbar=False)
    plt.title(f"Confusion Matrix - {model_name} Performance", fontsize=12, fontweight='bold', pad=15)
    plt.xlabel("Predicted Label", fontsize=10, fontweight='bold')
    plt.ylabel("True Label", fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_log_roc_curve(model, X_test, y_test, target_class_index=1, model_name="CatBoost"):
    y_probs = model.predict_proba(X_test)[:, target_class_index]
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 7))
    plt.semilogx(fpr, tpr, color='darkorange', lw=2, label=f'Log ROC ({model_name}) (area = {roc_auc:.4f})')
    plt.semilogx(fpr, fpr, color='navy', lw=1, linestyle='--', label='Random Classifier')
    
    plt.xlim([0.001, 1.0]) 
    plt.ylim([0.0, 1.05])
    plt.yticks(np.arange(0, 1.05, 0.05))
    plt.xlabel('False Positive Rate (Log Scale)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title(f'Logarithmic ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.show()
    return fpr, tpr, thresholds

def extract_features_vectorized(file_path):
    y, sr = librosa.load(file_path, sr=16000, mono=True)
    y = y / (np.max(y) + 1e-5)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    delta = librosa.feature.delta(mfccs)
    delta2 = librosa.feature.delta(mfccs, order=2)
    
    features = []
    for feat in [mfccs, delta, delta2]:
        features.extend(np.mean(feat, axis=1))
        features.extend(np.std(feat, axis=1))
        
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    flatness = librosa.feature.spectral_flatness(y=y)
    features.extend([np.mean(centroid), np.std(centroid), np.mean(flatness)])
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

def train_and_evaluate_cat(X, y, title="CatBoost: Drone vs. Environment"):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # הגדרת מודל CatBoost עם פרמטרים מותאמים
    model = CatBoostClassifier(
        iterations=600,
        learning_rate=0.05,
        depth=6,
        random_state=42,
        verbose=100  # מדפיס סטטוס אימון כל 100 שלבים
    )
    
    print(f"\n--- Starting CatBoost Training ---")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    print(f"\n--- {title} ---")
    print(classification_report(y_test, preds))
    
    # הצגת שגיאות מפורטות בטקסט
    print_detailed_errors(y_test, preds, show_matrix=True)
    
    # הצגת מטריצת בלבול גרפית
    plot_confusion_matrix_graphic(y_test, preds, model_name="CatBoost")
    
    return model, X_test, y_test

# --- CONFIGURATION ---
BASE_DATA_DIR = r"/Users/deviceone/Downloads/new_balanced_2s_dataset_nasa2"
model_dir = r"/Users/deviceone/Documents/d_detection/models"

binary_map = {
    0: ['background'], 
    1: ['target_drone']  
}

if __name__ == "__main__":
    # 1. טעינת נתונים וחילול מאפיינים
    X_bin, y_bin = prepare_data(BASE_DATA_DIR, binary_map)
    
    # 2. אימון והערכה
    model_bin, x_test, y_test = train_and_evaluate_cat(X_bin, y_bin)

    # 3. שמירת המודל
    model_out_path = os.path.join(model_dir, "2s_model_catboost.joblib")
    save_trained_model(model_bin, model_out_path)
    
    # 4. גרף ROC לוגריתמי
    plot_log_roc_curve(model_bin, x_test, y_test, target_class_index=1, model_name="CatBoost")