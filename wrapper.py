import abc
import pickle
import numpy as np
import librosa
import xgboost as xgb

# Note: Assumes BaseModel is imported from his infrastructure framework
# from your_infrastructure import BaseModel 

class XGBoostDroneDetectionModel(BaseModel):
    """
    XGBoost drone detection model adapted to the required architecture.
    Uses STFT Magnitude, Chroma, and Extended Spectral Features.
    Takes a batch of raw waveforms (B, samples) and returns per-sample scores (B,).
    """

    def __init__(self, cfg, inference: bool = False, model_path: str = "2s_model_omesi.pickle"):
        # Call parent class initializer
        super().__init__(cfg, inference)
        
        # Load the updated XGBoost model from the pickle file
        with open(model_path, 'rb') as file:
            self.model = pickle.load(file)
            
        # Constant sampling rate configuration (16kHz based on training)
        self.sr = 16000 

    def _extract_features_for_batch(self, x_matrix):
        """
        Internal helper to extract the updated feature matrix from a batch 
        of raw audio waveforms (B, samples).
        """
        num_windows = x_matrix.shape[0]
        features_list = []
        
        # Row-wise normalization of the raw audio waveform amplitude
        max_vals = np.max(np.abs(x_matrix), axis=1, keepdims=True) + 1e-5
        x_matrix_norm = x_matrix / max_vals
        
        # Loop through each window in the batch to perform feature extraction
        for i in range(num_windows):
            y_window = x_matrix_norm[i]
            
            # 1. Linear Spectrogram (STFT Magnitude)
            stft = np.abs(librosa.stft(y_window, n_fft=2048, hop_length=512))
            
            # 2. Chroma Features (Harmonic relationships of drone blades)
            chroma = librosa.feature.chroma_stft(S=stft, sr=self.sr)
            
            # 3. Extended Spectral Descriptors
            centroid = librosa.feature.spectral_centroid(S=stft, sr=self.sr)
            flatness = librosa.feature.spectral_flatness(S=stft)
            rolloff = librosa.feature.spectral_rolloff(S=self.sr, roll_percent=0.85) if hasattr(librosa.feature, 'spectral_rolloff') else librosa.feature.spectral_rolloff(S=stft, sr=self.sr, roll_percent=0.85)
            
            window_feats = []
            
            # Extract mean and standard deviation along the time axis for STFT and Chroma
            for feat in [stft, chroma]:
                window_feats.extend(np.mean(feat, axis=1))
                window_feats.extend(np.std(feat, axis=1))
                
            # Add stats for the 1D spectral arrays
            window_feats.extend([
                np.mean(centroid), np.std(centroid), 
                np.mean(flatness), np.std(flatness),
                np.mean(rolloff), np.std(rolloff)
            ])
            
            # 4. Extract 5 MFCC coefficients for background envelope mapping
            mfccs = librosa.feature.mfcc(S=librosa.amplitude_to_db(stft + 1e-5), sr=self.sr, n_mfcc=5)
            window_feats.extend(np.mean(mfccs, axis=1))
            window_feats.extend(np.std(mfccs, axis=1))
            
            features_list.append(window_feats)
            
        return np.array(features_list)

    def forward(self, x):
        """
        Maps an input batch of raw audio waveforms to per-sample drone probability scores.
        
        Parameters:
        -----------
        x : np.ndarray
            A 2D array of raw audio waveforms with shape (B, samples).
            Where B is the batch size, and samples represents 2 seconds of audio (32,000 samples).
            
        Returns:
        --------
        scores : np.ndarray
            A 1D array of shape (B,) containing the drone probability scores (0.0 to 1.0).
        """
        # Validate that the input is a 2D matrix
        if len(x.shape) != 2:
            raise ValueError(f"Expected input shape (B, samples), but got {x.shape}")
            
        # Extract features using the updated pipeline (STFT, Chroma, Spectral descriptors, 5 MFCCs)
        feature_matrix = self._extract_features_for_batch(x)
        
        # Run vectorized parallel inference using the updated XGBoost model
        probabilities = self.model.predict_proba(feature_matrix)[:, 1]
        
        return probabilities