import os
import glob
import numpy as np
import librosa
import matplotlib.pyplot as plt
from tqdm import tqdm

def calculate_cross_folder_snr_peaks(drone_folder, background_folder):
    """
    Advanced Cross-Folder SNR: Compares drone signal peaks (90th percentile) 
    against the baseline noise floor (10th percentile) of the background folder.
    """
    drone_files = glob.glob(os.path.join(drone_folder, "*.wav"))
    bg_files = glob.glob(os.path.join(background_folder, "*.wav"))
    
    if not drone_files or not bg_files:
        print("❌ Missing files in drone or background folders.")
        return
        
    print(f"🔍 Analyzing {len(drone_files)} drone files against {len(bg_files)} background files...")
    
    # STEP A: Calculate the average NOISE FLOOR from the background folder
    bg_noise_floors = []
    for bg_path in tqdm(bg_files, desc="Extracting Background Noise Floor"):
        try:
            y_bg, _ = librosa.load(bg_path, sr=16000, mono=True)
            if len(y_bg) == 0: continue
            
            # Peak normalization
            y_bg = y_bg / (np.max(np.abs(y_bg)) + 1e-5)
            rms_bg = librosa.feature.rms(y=y_bg, frame_length=2048, hop_length=512)[0]
            
            # Take the quietest parts of the background files (10th percentile)
            bg_noise_floors.append(np.percentile(rms_bg, 10) ** 2)
        except Exception:
            continue
            
    global_noise_floor = np.mean(bg_noise_floors)
    print(f"📉 Global Noise Floor Power: {global_noise_floor:.6f}")
    
    # STEP B: Calculate SNR using Drone Peaks (90th percentile) vs Global Noise Floor
    file_snrs = []
    for drone_path in tqdm(drone_files, desc="Calculating True SNR"):
        try:
            y_drone, _ = librosa.load(drone_path, sr=16000, mono=True)
            if len(y_drone) == 0: continue
                
            y_drone = y_drone / (np.max(np.abs(y_drone)) + 1e-5)
            rms_drone = librosa.feature.rms(y=y_drone, frame_length=2048, hop_length=512)[0]
            
            # Extract the actual drone signal peak energy
            drone_signal_power = np.percentile(rms_drone, 90) ** 2
            
            # Calculate SNR directly (No subtraction needed since we use independent folders)
            snr_db = 10 * np.log10(drone_signal_power / (global_noise_floor + 1e-10))
            file_snrs.append(snr_db)
            
        except Exception as e:
            print(f"⚠️ Error: {e}")

    if not file_snrs:
        print("❌ No SNR data generated.")
        return

    # STEP C: Stats & Summary
    avg_snr = np.mean(file_snrs)
    print("\n" + "="*50)
    print("📊 CORRECTED CROSS-FOLDER SNR REPORT")
    print("="*50)
    print(f"📈 Average SNR:  {avg_snr:.2f} dB")
    print(f"📉 Minimum SNR:  {np.min(file_snrs):.2f} dB")
    print(f"🚀 Maximum SNR:  {np.max(file_snrs):.2f} dB")
    print("="*50 + "\n")

    # Plot
    plt.figure(figsize=(9, 5))
    plt.hist(file_snrs, bins=20, color='darkcyan', edgecolor='black', alpha=0.7)
    plt.axvline(avg_snr, color='red', linestyle='--', linewidth=2, label=f'Average ({avg_snr:.1f} dB)')
    plt.title('Corrected True SNR Distribution', fontsize=12, fontweight='bold')
    plt.xlabel('SNR (dB)', fontweight='bold')
    plt.ylabel('Number of Drone Files', fontweight='bold')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # DRONE_DIR = r"/Users/deviceone/Documents/data/2026.04.28_omesi/slice_2s_overlap_2026.04.28_omesi/target_drone"
    # NOISE_DIR = r"/Users/deviceone/Documents/data/2026.04.28_omesi/slice_2s_overlap_2026.04.28_omesi/background"

    # DRONE_DIR = r"/Users/deviceone/Documents/data/551/slice_2s_overlap_551_device_1/target_drone"
    # NOISE_DIR = r"/Users/deviceone/Documents/data/551/slice_2s_overlap_551_device_1/background"

    DRONE_DIR = r"/Users/deviceone/Documents/data/2026.05.01_omesi/slice_2s_overlap_2026.05.01_omesi/target_drone"
    NOISE_DIR = r"/Users/deviceone/Documents/data/2026.05.01_omesi/slice_2s_overlap_2026.05.01_omesi/background"
    calculate_cross_folder_snr_peaks(DRONE_DIR, NOISE_DIR)