
import os
import librosa
import numpy as np
from acoustics import Signal
from scipy.signal import butter, sosfilt, sosfreqz
import scipy.signal as signal
import pandas as pd

def a_weighting_filter(fs):
    """
    Design an IIR filter for A-weighting based on IEC 61672-1.
    Parameters:
        fs: Sampling frequency (Hz)
    Returns:
        sos: Second-order sections for filtering
    """
    f1 = 20.6
    f2 = 107.7
    f3 = 737.9
    f4 = 12194.0
    zeros = np.array([0, 0, -2*np.pi*f4, -2*np.pi*f4], dtype=np.float64)
    poles = np.array([-2*np.pi*f1, -2*np.pi*f1, -2*np.pi*f2, -2*np.pi*f3, -2*np.pi*f4, -2*np.pi*f4], dtype=np.float64)
    gain = (2*np.pi*f4)**4 / np.sqrt((2*np.pi*f2)*(2*np.pi*f3))
    
    try:
        b, a = signal.zpk2tf(zeros, poles, gain)
        sos = signal.tf2sos(b, a, analog=False)
        w, h = sosfreqz(sos, worN=2048, fs=fs)
        freq_1khz = np.argmin(np.abs(w - 1000))
        gain_1khz = np.abs(h[freq_1khz])
        sos = signal.tf2sos(b, a, analog=False, gain=1/gain_1khz)
    except Exception as e:
        print(f"Error designing filter: {e}")
        sos = butter(4, [20, 20000], btype='band', fs=fs, output='sos')
        print("Using fallback Butterworth filter")
    
    return sos

def calculate_leq_a(file_path, sr=44100, p_scale=1.5):
    """Calculate A-weighted Leq for a WAV file."""
    try:
        # Load WAV file
        y, sr = librosa.load(file_path, sr=sr, mono=True)
        p = y * p_scale
        p0 = 2e-5  # Reference pressure (20 µPa)
        
        # Create A-weighting filter
        sos = a_weighting_filter(sr)
        
        # Apply A-weighting filter to full signal
        p_a_weighted = sosfilt(sos, p)
        signal_a_weighted = Signal(p_a_weighted, fs=sr)
        
        # Calculate Leq using acoustics
        leq = signal_a_weighted.leq()
        if not np.isfinite(leq):
            raise ValueError("Invalid Leq value")
        
        return leq
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def process_wav_files(folder_path, output_csv="leq_results.csv"):
    """Process WAV files in folder and save Leq results to CSV."""
    results = []
    
    # Check if folder exists
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder {folder_path} does not exist")
    
    # Iterate through WAV files in the folder
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.wav'):
            file_path = os.path.join(folder_path, filename)
            leq = calculate_leq_a(file_path)
            if leq is not None:
                results.append({"Filename": filename, "Leq_dBA": f"{leq:.2f}"})
                print(f"Processed {filename}: Leq = {leq:.2f} dB(A)")
    
    # Save results to CSV
    if results:
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        print(f"Results saved to {output_csv}")
    else:
        print("No WAV files processed successfully")

if __name__ == "__main__":
    folder_path = "data\\noiseData"  # Adjust to your folder path
    process_wav_files(folder_path)
