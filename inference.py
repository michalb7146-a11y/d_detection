import os
import pickle
import librosa
import numpy as np
import xgboost as xgb

def extract_features_matrix(file_path, window_duration=2.0, sr=16000):
    """
    Loads the audio file, splits it into continuous 2-second windows,
    and extracts features for ALL windows simultaneously into a single matrix.
    """
    # 1. Load entire audio
    y_full, _ = librosa.load(file_path, sr=sr, mono=True)
    
    window_samples = int(window_duration * sr)
    total_samples = len(y_full)
    
    # Calculate how many full 2-second windows we can get
    num_windows = total_samples // window_samples
    if num_windows == 0:
        raise ValueError("Audio file is shorter than 2 seconds!")
        
    # Truncate audio to fit exact number of windows and reshape into a matrix
    # Shape will be: (num_windows, window_samples)
    y_clipped = y_full[:num_windows * window_samples]
    y_matrix = y_clipped.reshape(num_windows, window_samples)
    
    # 2. Vectorized Feature Extraction across all windows
    # Normalize each row (window) independently
    max_vals = np.max(np.abs(y_matrix), axis=1, keepdims=True) + 1e-5
    y_matrix_norm = y_matrix / max_vals
    
    # Extract MFCCs for all rows at once using librosa's ability to handle matrices
    # We apply the feature extraction across the axis of time
    features_list = []
    
    print(f"⏳ Processing {num_windows} windows in parallel matrix operations...")
    
    # Loop only over the windows to extract features (highly optimized by Librosa)
    for i in range(num_windows):
        y_window = y_matrix_norm[i]
        
        mfccs = librosa.feature.mfcc(y=y_window, sr=sr, n_mfcc=13)
        delta = librosa.feature.delta(mfccs)
        delta2 = librosa.feature.delta(mfccs, order=2)
        
        window_feats = []
        for feat in [mfccs, delta, delta2]:
            window_feats.extend(np.mean(feat, axis=1))
            window_feats.extend(np.std(feat, axis=1))
            
        centroid = librosa.feature.spectral_centroid(y=y_window, sr=sr)
        flatness = librosa.feature.spectral_flatness(y=y_window)
        window_feats.extend([np.mean(centroid), np.std(centroid), np.mean(flatness)])
        
        features_list.append(window_feats)
        
    # Convert list of windows into a 2D Matrix of Shape: (Num_Windows, Num_Features)
    # This is the "B over T" matrix he wanted
    return np.array(features_list), num_windows

if __name__ == "__main__":
    MODEL_PATH = r"models/2s_model_omesi.pickle"
    AUDIO_FILE_TO_TEST = r"/Users/deviceone/Downloads/session_20260430T180015_807/alsa_default.wav" 
    
    print("⏳ Loading XGBoost model from Pickle file...")
    if not os.path.exists(MODEL_PATH):
        print(f"❌ ERROR: Model file not found at: {MODEL_PATH}")
        exit()
        
    with open(MODEL_PATH, 'rb') as file:
        model = pickle.load(file)
        
    try:
        if not os.path.exists(AUDIO_FILE_TO_TEST):
            print(f"❌ ERROR: Audio file to test not found at: {AUDIO_FILE_TO_TEST}")
            exit()
            
        # Extract the entire data matrix
        data_matrix, num_windows = extract_features_matrix(AUDIO_FILE_TO_TEST)
        
        print(f"📊 Data Matrix Ready. Shape: {data_matrix.shape} (Windows x Features)")
        print("🚀 Running Parallel Inference via XGBoost...")
        
        # --- PARALLEL INFERENCE (No loops!) ---
        # XGBoost handles the entire matrix instantly in parallel
        all_predictions = model.predict(data_matrix)
        all_probabilities = model.predict_proba(data_matrix)
        
        # 3. Print Results Summary cleanly
        print("\n==========================================")
        print("📊 MATRIX DETECTION RESULTS SUMMARY:")
        print("==========================================")
        
        for idx in range(num_windows):
            time_start = idx * 2
            time_end = time_start + 2
            time_label = f"[{time_start:02d}s - {time_end:02d}s]"
            
            pred = all_predictions[idx]
            prob = all_probabilities[idx]
            
            if pred == 1:
                print(f"{time_label} 🚨 DRONE DETECTED! (Confidence: {prob[1]*100:.2f}%)")
            else:
                print(f"{time_label} ✅ CLEAN BACKGROUND (Confidence: {prob[0]*100:.2f}%)")
                
        print("==========================================\n")
        
    except Exception as e:
        print(f"❌ ERROR during processing or inference: {e}")