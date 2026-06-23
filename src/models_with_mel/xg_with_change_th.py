import os
import glob
import numpy as np
import librosa
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, ConfusionMatrixDisplay
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle

def save_trained_model_as_pickle(model, filename="2s_model_omesi.pkl"):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model saved successfully as {filename}")

def print_detailed_errors(y_test, preds, show_matrix=True):
    """
    Prints the exact number of false alarms and missed detections in English
    to ensure proper formatting in the terminal.
    """
    if not show_matrix:
        return
        
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    
    print("\n--- DETAILED ERROR ANALYSIS (Confusion Matrix) ---")
    print(f"OK True Negatives  (Correct Background): {tn}")
    print(f"XX False Positives (False Alarms):       {fp}")
    print(f"XX False Negatives (Missed Detections):  {fn}")
    print(f"OK True Positives  (Correct Drone):     {tp}\n")

def plot_confusion_matrix_graphic(y_test, preds):
    """Plots a visual and colorful Confusion Matrix using Matplotlib."""
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

    # --- עדכון: חישוב סף אוטומטי מבוסס על אחוז גילוי רצוי (Recall) למניעת פספוסים ---
    # אנו שואפים ל-99% גילוי רחפנים (מוריד משמעותית את ה-False Negatives)
    target_tpr = 0.99 
    idx = np.argmin(np.abs(tpr - target_tpr))
    
    print("\n" + "="*50)
    print("🎯 RECOMMENDED THRESHOLD BASED ON TARGET RECALL:")
    print(f"To achieve {tpr[idx]*100:.2f}% Drone Detection (Recall):")
    print(f"-> Expected False Alarms Rate (FPR) will be: {fpr[idx]*100:.2f}%")
    print(f"-> SET YOUR THRESHOLD TO: {thresholds[idx]:.4f}")
    print("="*50 + "\n")
    
    return fpr, tpr, thresholds, thresholds[idx]

def plot_performance_curves(model, X_test, y_test, target_class_index=1):
    y_probs = model.predict_proba(X_test)[:, target_class_index]
    fpr, tpr, thresholds_roc = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)
    precision, recall, thresholds_pr = precision_recall_curve(y_test, y_probs)
    
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    ax[0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (area = {roc_auc:.4f})')
    ax[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax[0].set_xlabel('False Positive Rate (1 - Specificity)')
    ax[0].set_ylabel('True Positive Rate (Sensitivity)')
    ax[0].set_title('Receiver Operating Characteristic')
    ax[0].legend(loc="lower right")
    ax[0].grid(alpha=0.3)

    ax[1].plot(recall, precision, color='blue', lw=2)
    ax[1].set_xlabel('Recall (Detection Rate)')
    ax[1].set_ylabel('Precision (1 - False Alarm Rate)')
    ax[1].set_title('Precision-Recall Curve')
    ax[1].grid(alpha=0.3)
    plt.show()
    return fpr, tpr, thresholds_roc

def extract_features_vectorized(file_path):
    """
    Optimized feature extraction for drones: Uses linear spectrogram (STFT),
    Chroma features for blade harmonic matching, and advanced spectral descriptors.
    """
    y, sr = librosa.load(file_path, sr=16000, mono=True)
    y = y / (np.max(y) + 1e-5)
    
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

def prepare_data(base_paths, label_folder_map):
    """
    Prepared data from MULTIPLE base directories.
    Loops through all base paths and extracts features for matching folders.
    """
    X, y = [], []
    folder_to_label = {folder: label for label, folders in label_folder_map.items() for folder in folders}
    
    if isinstance(base_paths, str):
        base_paths = [base_paths]
        
    for base_path in base_paths:
        print(f"\n--- Scanning Base Directory: {base_path} ---")
        
        for folder, label in folder_to_label.items():
            folder_path = os.path.join(base_path, folder)
            
            if not os.path.exists(folder_path):
                print(f"Skipping: '{folder}' folder not found in {base_path}")
                continue
                
            files = glob.glob(os.path.join(folder_path, "*.wav"))
            if len(files) == 0:
                print(f"No .wav files found in {folder_path}")
                continue
                
            print(f"Loading {len(files)} files for label: {label} (from {folder})")
            
            for f in tqdm(files):
                try:
                    feat = extract_features_vectorized(f)
                    X.append(feat)
                    y.append(label)
                except Exception as e:
                    print(f"Error processing {f}: {e}")
    return np.array(X), np.array(y)

def optimize_hyperparameters(X, y):
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    base_model = xgb.XGBClassifier(
        objective='binary:logistic',
        device="cpu",
        eval_metric='logloss'
    )
    
    # במידה ומריצים אופטימיזציה, כדאי להוסיף סביב איזה סקייל של משקולות רוצים לבדוק
    param_grid = {
        'max_depth': [4, 6, 8],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [300, 500],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8],
        'scale_pos_weight': [2.0, 3.0] # בדיקת משקולות מועדפות
    }
    
    print("\n--- STARTING HYPERPARAMETER OPTIMIZATION (Grid Search) ---")
    
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=3,
        scoring='f1',
        verbose=1,
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    print("\n==========================================")
    print("🏆 OPTIMIZATION COMPLETE! Best parameters found:")
    print(grid_search.best_params_)
    print(f"📊 Best F1-Score achieved during CV: {grid_search.best_score_:.4f}")
    print("==========================================\n")
    
    return grid_search.best_estimator_

def train_and_evaluate(X, y, title="Classifier Results", show_matrix=True):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    num_class = len(np.unique(y))
    objective = 'binary:logistic' if num_class <= 2 else 'multi:softprob'
    
    # הוספת ה-scale_pos_weight למניעת פספוסי רחפנים (כאן מוגדר כ-5.0 כברירת מחדל אגרסיבית)
    best_params = {
        'colsample_bytree': 0.7, 
        'learning_rate': 0.05, 
        'max_depth': 4, 
        'n_estimators': 500, 
        'subsample': 0.7,
        'scale_pos_weight': 5.0  # <--- נותן משקל פי 5 לטעויות על סיווג רחפנים
    }

    model = xgb.XGBClassifier(
        objective=objective,
        eval_metric='logloss' if num_class <= 2 else 'mlogloss',
        device="cpu",
        **best_params
    )
    
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    print(f"\n--- {title} ---")
    print(classification_report(y_test, preds))
    
    print_detailed_errors(y_test, preds, show_matrix=show_matrix)
    
    if show_matrix:
        plot_confusion_matrix_graphic(y_test, preds)
    
    return model, X_test, y_test


# ======================================================================
# 🛠️ CONFIGURATION
# ======================================================================

DATA_DIRECTORIES = [
    r"/Users/deviceone/Downloads/data/2026.04.28_omesi/new_balanced_2s_dataset_2026.04.28_omesi",
    r"/Users/deviceone/Downloads/data/2026.05.01_omesi/new_balanced_2s_dataset",
    r"/Users/deviceone/Downloads/data/dregon/new_balanced_2s_dataset_dregon",
    r"/Users/deviceone/Downloads/data/nasa_2/new_balanced_2s_dataset_nasa_2",
    r"/Users/deviceone/Downloads/data/tut/new_balanced_2s_dataset_tut",
]

MODEL_OUTPUT_DIR = r"/Users/deviceone/Documents/d_detection/models"

binary_map = {
    0: ['background'], 
    1: ['target_drone']  
}

if __name__ == "__main__":
    # 1. טעינת הנתונים וחילוץ הפיצ'רים
    X_bin, y_bin = prepare_data(DATA_DIRECTORIES, binary_map)
    
    unique_labels = np.unique(y_bin)
    print(f"\nTotal samples loaded: {len(X_bin)}")
    print(f"Labels found in dataset: {unique_labels}")
    
    if len(unique_labels) < 2:
        print("⚠️ Error: Dataset must contain both background and drone samples to train.")
    else:
        # 2. אימון המודל (לפי המשקולות שהגדרנו בתוך train_and_evaluate)
        model_bin, x_test, y_test = train_and_evaluate(X_bin, y_bin, "Binary: Drone vs. Environment", show_matrix=False) # <--- שינינו ל-False כדי לא להציג את המטריצה של ה-0.5 הדיפולטי
        
        model_out_path = os.path.join(MODEL_OUTPUT_DIR, "2s_model_omesi.pickle")
        save_trained_model_as_pickle(model_bin, model_out_path)
        
        # 3. הפקת ההסתברויות הגולמיות מהמודל עבור נתוני הטסט (ערכים בין 0 ל-1)
        y_probs = model_bin.predict_proba(x_test)[:, 1] # <--- הוספנו כאן כדי לשמור את הפרודקשן הלוגיסטי
        
        # 4. הרצת פונקציית ה-ROC לקבלת המלצה ראשונית
        fpr, tpr, thresholds, auto_threshold = plot_log_roc_curve(model_bin, x_test, y_test, target_class_index=1)

        # ======================================================================
        # 🎯 כאן את מכוונת ידנית את ה-THRESHOLD!
        # ======================================================================
        # משחקים עם המספר הזה (למשל: 0.15, 0.25, 0.35) ובודקים את הטרייד-אוף בריצה
        my_custom_threshold = 0.45  # <--- כאן המספר משתנה לפי הניסויים שלך!
        
        print(f"📊 Applying Custom Safety-First Threshold ({my_custom_threshold}) to Test Data...")
        
        # החיתוך בפועל: כל דגימה שההסתברות שלה מעל הסף נחשבת רחפן (1), כל השאר רקע (0)
        custom_preds = (y_probs >= my_custom_threshold).astype(int) # <--- כאן מיושם הסף החדש שלך
        
        # 5. הדפסת התוצאות הסופיות והצגת המטריצה המעודכנת לפי הסף שלך
        print_detailed_errors(y_test, custom_preds, show_matrix=True)
        plot_confusion_matrix_graphic(y_test, custom_preds)