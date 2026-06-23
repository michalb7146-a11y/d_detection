import os
import glob
import numpy as np
import librosa
import matplotlib.pyplot as plt
from tqdm import tqdm

def calculate_folder_snr(folder_path):
    """
    Calculates the SNR for all WAV files in a given folder
    and provides a statistical analysis of the experiment conditions.
    """
    # Search for all WAV files in the folder
    audio_files = glob.glob(os.path.join(folder_path, "*.wav"))
    
    if not audio_files:
        print(f"❌ No .wav files found in folder: {folder_path}")
        return
    
    print(f"🔍 Found {len(audio_files)} files for SNR analysis...")
    file_snrs = []
    
    for file_path in tqdm(audio_files, desc="Calculating SNR"):
        try:
            # Load audio file
            y, sr = librosa.load(file_path, sr=16000, mono=True)
            if len(y) == 0:
                continue
                
            # Calculate Root Mean Square (RMS) energy in short time windows
            rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
            
            # Estimate noise floor (lower 10th percentile)
            noise_floor = np.percentile(rms, 10) ** 2
            
            # Estimate drone signal peak (upper 90th percentile)
            signal_peak = np.percentile(rms, 90) ** 2
            
            # Calculate SNR in dB
            snr_estimated = 10 * np.log10(signal_peak / (noise_floor + 1e-10))
            file_snrs.append(snr_estimated)
            
        except Exception as e:
            print(f"⚠️ Error processing file {os.path.basename(file_path)}: {e}")

    if not file_snrs:
        print("❌ Could not calculate SNR for any file.")
        return

    # Calculate statistical metrics for the experiment
    avg_snr = np.mean(file_snrs)
    min_snr = np.min(file_snrs)
    max_snr = np.max(file_snrs)
    std_snr = np.std(file_snrs)

    print("\n" + "="*50)
    print("📊 DETAILED EXPERIMENT SNR REPORT")
    print("="*50)
    print(f"📈 Average SNR in Experiment:  {avg_snr:.2f} dB")
    print(f"📉 Minimum SNR (Worst Case):    {min_snr:.2f} dB")
    print(f"🚀 Maximum SNR (Best Case):     {max_snr:.2f} dB")
    print(f"📐 Standard Deviation (Spread): {std_snr:.2f} dB")
    print("-" * 50)
    
    # Interpretation of the experiment quality
    if avg_snr > 20:
        print("📝 Insight: Clean environment / Close range experiment (Strong & clean signal).")
    elif avg_snr > 10:
        print("📝 Insight: Standard experiment conditions (Good signal, mild background noise).")
    else:
        print("📝 Insight: Highly challenging experiment! (Distant drone, strong wind, or noisy environment).")
    print("="*50 + "\n")

    # Plot SNR Distribution Histogram
    plt.figure(figsize=(9, 5))
    plt.hist(file_snrs, bins=15, color='royalblue', edgecolor='black', alpha=0.7)
    plt.axvline(avg_snr, color='red', linestyle='--', linewidth=2, label=f'Average ({avg_snr:.1f} dB)')
    plt.title('Distribution of SNR Across Experiment Files', fontsize=12, fontweight='bold')
    plt.xlabel('SNR (dB)', fontweight='bold')
    plt.ylabel('Number of Files', fontweight='bold')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# ==========================================
# 🛠️ RUN CONFIGURATION
# ==========================================
if __name__ == "__main__":
    # Change this path to your specific experiment folder containing the target drone wav files
    # TARGET_FOLDER = r"/Users/deviceone/Documents/data/2026.04.28_omesi/slice_2s_overlap_2026.04.28_omesi/target_drone"
    # TARGET_FOLDER = r"/Users/deviceone/Documents/data/551/slice_2s_overlap_551_device_1/target_drone"
    TARGET_FOLDER = r"/Users/deviceone/Documents/data/2026.05.01_omesi/slice_2s_overlap_2026.05.01_omesi/target_drone"
    
    calculate_folder_snr(TARGET_FOLDER)