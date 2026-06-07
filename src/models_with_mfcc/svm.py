import os
import glob
import numpy as np
import librosa
import joblib
import matplotlib.pyplot as plt

from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(file_path):

    y, sr = librosa.load(
        file_path,
        sr=16000,
        mono=True
    )

    y = y / (np.max(np.abs(y)) + 1e-8)

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=13
    )

    delta = librosa.feature.delta(mfcc)

    delta2 = librosa.feature.delta(
        mfcc,
        order=2
    )

    features = []

    for feat in [mfcc, delta, delta2]:

        features.extend(np.mean(feat, axis=1))
        features.extend(np.std(feat, axis=1))

    centroid = librosa.feature.spectral_centroid(
        y=y,
        sr=sr
    )

    bandwidth = librosa.feature.spectral_bandwidth(
        y=y,
        sr=sr
    )

    rolloff = librosa.feature.spectral_rolloff(
        y=y,
        sr=sr
    )

    flatness = librosa.feature.spectral_flatness(
        y=y
    )

    zcr = librosa.feature.zero_crossing_rate(y)

    rms = librosa.feature.rms(y=y)

    spectral_features = [
        np.mean(centroid),
        np.std(centroid),

        np.mean(bandwidth),
        np.std(bandwidth),

        np.mean(rolloff),
        np.std(rolloff),

        np.mean(flatness),
        np.std(flatness),

        np.mean(zcr),
        np.std(zcr),

        np.mean(rms),
        np.std(rms)
    ]

    features.extend(spectral_features)

    return np.array(features)


# ============================================================
# DATASET LOADING
# ============================================================

def prepare_data(base_dir, label_map):

    X = []
    y = []

    folder_to_label = {
        folder: label
        for label, folders in label_map.items()
        for folder in folders
    }

    for folder, label in folder_to_label.items():

        folder_path = os.path.join(
            base_dir,
            folder
        )

        wav_files = glob.glob(
            os.path.join(folder_path, "*.wav")
        )

        print(
            f"Loading {len(wav_files)} files "
            f"from {folder}"
        )

        for wav_file in tqdm(wav_files):

            try:

                features = extract_features(
                    wav_file
                )

                X.append(features)
                y.append(label)

            except Exception as e:

                print(
                    f"Error processing "
                    f"{wav_file}: {e}"
                )

    return np.array(X), np.array(y)


# ============================================================
# TRAIN
# ============================================================

def train_svm(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "svm",
            SVC(
                kernel="rbf",
                C=10,
                gamma="scale",
                probability=True
            )
        )
    ])

    print("\nTraining SVM...")

    model.fit(
        X_train,
        y_train
    )

    preds = model.predict(
        X_test
    )

    print("\n========================")
    print("CLASSIFICATION REPORT")
    print("========================\n")

    print(
        classification_report(
            y_test,
            preds
        )
    )

    cm = confusion_matrix(
        y_test,
        preds
    )

    tn, fp, fn, tp = cm.ravel()

    print("\n========================")
    print("CONFUSION MATRIX")
    print("========================\n")

    print(f"TN: {tn}")
    print(f"FP: {fp}")
    print(f"FN: {fn}")
    print(f"TP: {tp}")

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "Background",
            "Drone"
        ]
    )

    disp.plot(
        cmap="Blues",
        values_format="d"
    )

    plt.title(
        "Drone Detection - SVM"
    )

    plt.show()

    return model, X_test, y_test


# ============================================================
# ROC
# ============================================================

def plot_roc_curve(
    model,
    X_test,
    y_test
):

    probs = model.predict_proba(
        X_test
    )[:, 1]

    fpr, tpr, _ = roc_curve(
        y_test,
        probs
    )

    roc_auc = auc(
        fpr,
        tpr
    )

    plt.figure(
        figsize=(8, 6)
    )

    plt.plot(
        fpr,
        tpr,
        lw=2,
        label=f"AUC = {roc_auc:.4f}"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        "--"
    )

    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(
        "ROC Curve"
    )

    plt.legend()

    plt.grid(True)

    plt.show()


# ============================================================
# SAVE MODEL
# ============================================================

def save_model(
    model,
    filename
):

    joblib.dump(
        model,
        filename
    )

    print(
        f"\nModel saved to:\n{filename}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    BASE_DATA_DIR = r"/Users/deviceone/Downloads/new_balanced_2s_dataset_omesi"

    LABEL_MAP = {
        0: ["background"],
        1: ["target_drone"]
    }

    print("Loading dataset...")

    X, y = prepare_data(
        BASE_DATA_DIR,
        LABEL_MAP
    )

    print("\nDataset statistics:")
    print(f"Samples: {len(y)}")
    print(f"Features: {X.shape[1]}")

    unique, counts = np.unique(
        y,
        return_counts=True
    )

    print(
        dict(
            zip(unique, counts)
        )
    )

    model, X_test, y_test = train_svm(
        X,
        y
    )

    plot_roc_curve(
        model,
        X_test,
        y_test
    )

    save_model(
        model,
        "drone_svm_model.joblib"
    )