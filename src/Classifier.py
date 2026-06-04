import os
import glob
import numpy as np
import librosa
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, ConfusionMatrixDisplay
from tqdm import tqdm
import matplotlib.pyplot as plt
import joblib

def save_trained_model(model, filename="drone_binary_model.joblib"):
    joblib.dump(model, filename)
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
    return fpr, tpr, thresholds

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

def optimize_hyperparameters(X, y):
    """
    Runs GridSearchCV to find the best XGBoost parameters 
    based on the F1-score metric.
    """
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    base_model = xgb.XGBClassifier(
        objective='binary:logistic',
        device="cpu",
        eval_metric='logloss'
    )
    
    # Hyperparameter grid matrix
    param_grid = {
        'max_depth': [4, 6, 8],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [300, 500],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8]
    }
    
    
    print("\n--- STARTING HYPERPARAMETER OPTIMIZATION (Grid Search) ---")
    print("Testing combinations, this may take a few minutes...")
    
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
    
    # # Current configuration parameters
    # model = xgb.XGBClassifier(
    #     objective=objective, n_estimators=500, max_depth=8, learning_rate=0.1,
    #     gamma=0.1, subsample=0.8, colsample_bytree=0.8, device="cpu",  
    #     eval_metric='logloss' if num_class <= 2 else 'mlogloss'
    # )

    # המילון שקיבלת מהתוצאות
    best_params = {
        'colsample_bytree': 0.7, 
        'learning_rate': 0.05, 
        'max_depth': 4, 
        'n_estimators': 500, 
        'subsample': 0.7
    }

    # תיקון: שימוש ב-xgb.XGBClassifier והוספת הפרמטרים של המשימה
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
    
    # Text terminal analysis in clean English
    print_detailed_errors(y_test, preds, show_matrix=show_matrix)
    
    # Graphic display of the confusion matrix
    if show_matrix:
        plot_confusion_matrix_graphic(y_test, preds)
    
    return model, X_test, y_test


# --- CONFIGURATION ---
BASE_DATA_DIR = r"/Users/deviceone/Downloads/new_balanced_2s_dataset_omesi"

binary_map = {
    0: ['background'], 
    1: ['target_drone']  
}

if __name__ == "__main__":
    # 1. Load data and extract features
    X_bin, y_bin = prepare_data(BASE_DATA_DIR, binary_map)
    
    # ----------------------------------------------------------------------
    # 💡 OPTION A: Standard Training & Evaluation (Currently Active)
    # ----------------------------------------------------------------------
    model_bin, x_test, y_test = train_and_evaluate(
        X_bin, y_bin, "Binary: Drone vs. Environment", show_matrix=True
    )

    model_dir = r"/Users/deviceone/Documents/d_detection/models"
    model_out_path = os.path.join(model_dir, "2s_model_omesi.joblib")
    save_trained_model(model_bin, model_out_path)
    
    fpr, tpr, thresholds = plot_log_roc_curve(model_bin, x_test, y_test, target_class_index=1)

    # ----------------------------------------------------------------------
    # 💡 OPTION B: Parameter Optimization Tuning
    # (To activate: uncomment the two lines below by removing the '#')
    # ----------------------------------------------------------------------
    # best_model = optimize_hyperparameters(X_bin, y_bin)
    # save_trained_model(best_model, os.path.join(model_dir, "optimized_model.joblib"))