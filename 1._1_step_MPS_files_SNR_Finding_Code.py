import os
import csv
import numpy as np
import wave
import soundfile as sf
import librosa
from pathlib import Path

# --- SNR algorithm ---

def complete_silence_check(audio):
    max_value_in_wav_file = 127
    max_value_16_bit = 32768
    silence_threshold = max_value_in_wav_file / max_value_16_bit
    window_size = 8000
    hop_size = 4000

    for start in range(0, len(audio) - window_size + 1, hop_size):
        window = audio[start:start + window_size]
        if np.average(np.abs(window)) > silence_threshold:
            return True
    return False

db_vals = np.arange(-20, 101)
g_vals = np.array([0.40974774, 0.40986926, 0.40998566, 0.40969089, 0.40986186, 0.40999006, 0.41027138, 0.41052627, 0.41101024, 0.41143264, 0.41231718, 0.41337272, 0.41526426, 0.4178192 , 0.42077252, 0.42452799, 0.42918886, 0.43510373, 0.44234195, 0.45161485, 0.46221153, 0.47491647, 0.48883809, 0.50509236, 0.52353709, 0.54372088, 0.56532427, 0.58847532, 0.61346212, 0.63954496, 0.66750818, 0.69583724, 0.72454762, 0.75414799, 0.78323148, 0.81240985, 0.84219775, 0.87166406, 0.90030504, 0.92880418, 0.95655449, 0.9835349 , 1.01047155, 1.0362095 , 1.06136425, 1.08579312, 1.1094819 , 1.13277995, 1.15472826, 1.17627308, 1.19703503, 1.21671694, 1.23535898, 1.25364313, 1.27103891, 1.28718029, 1.30302865, 1.31839527, 1.33294817, 1.34700935, 1.3605727 , 1.37345513, 1.38577122, 1.39733504, 1.40856397, 1.41959619, 1.42983624, 1.43958467, 1.44902176, 1.45804831, 1.46669568, 1.47486938, 1.48269965, 1.49034339, 1.49748214, 1.50435106, 1.51076426, 1.51698915, 1.5229097 , 1.528578  , 1.53389835, 1.5391211 , 1.5439065 , 1.54858517, 1.55310776, 1.55744391, 1.56164927, 1.56566348, 1.56938671, 1.57307767, 1.57654764, 1.57980083, 1.58304129, 1.58602496, 1.58880681, 1.59162477, 1.5941969 , 1.59693155, 1.599446  , 1.60185011, 1.60408668, 1.60627134, 1.60826199, 1.61004547, 1.61192472, 1.61369656, 1.61534074, 1.61688905, 1.61838916, 1.61985374, 1.62135878, 1.62268119, 1.62390423, 1.62513143, 1.62632463, 1.6274027 , 1.62842767, 1.62945532, 1.6303307 , 1.63128026, 1.63204102])

def wada_snr(wav, epsilon=1e-10):
    wav -= wav.mean()
    energy = (wav**2).sum()
    wav = wav / np.abs(wav).max()
    abs_wav = abs(wav)
    abs_wav[abs_wav < epsilon] = epsilon
    v1 = max(epsilon, abs_wav.mean())
    v2 = np.log(abs_wav).mean()
    v3 = np.log(v1) - v2
    wav_snr_idx = None
    if any(g_vals < v3):
        wav_snr_idx = np.where(g_vals < v3)[0].max()
    if wav_snr_idx is None:
        wav_snr = db_vals[0]
    elif wav_snr_idx == len(db_vals) - 1:
        wav_snr = db_vals[-1]
    else:
        wav_snr = db_vals[wav_snr_idx + 1]
    factor = 10 ** (wav_snr / 10)
    noise_energy = energy / (1 + factor)
    signal_energy = energy * factor / (1 + factor)
    snr = 10 * np.log10(signal_energy / noise_energy)
    return snr

def window_snr(audio_path):
    with wave.open(audio_path, 'rb') as wav:
        if wav.getsampwidth() != 2:
            print(f"{audio_path}: not a 16-bit file")
            return None

    audio, sample_rate = sf.read(audio_path)
    if not complete_silence_check(audio):
        return None

    duration = len(audio) / sample_rate
    if int(np.floor(duration)) - 6 <= 0:
        print(audio_path + " is too short")
        return None

    snr_list = []
    for j in range(0, int(duration - 6) + 1):
        offset = j
        a, sr = librosa.load(audio_path, offset=offset, duration=6.0, sr=None)
        snr_moving = wada_snr(a)
        snr_list.append(np.nanmean(snr_moving))

    if snr_list:
        snr_list.pop(0)
        avg_snr = np.nanmean(snr_list).round(2)
        return avg_snr
    else:
        return None

# --- Processing function ---

def process_mps_dataset(mps_root, output_csv_path):
    results = []
    for root, _, files in os.walk(mps_root):
        for fname in files:
            if fname.lower().endswith(".wav"):
                file_path = os.path.join(root, fname)
                speaker_id = Path(root).name
                try:
                    snr_value = window_snr(file_path)
                    if snr_value is not None:
                        results.append([speaker_id, fname, snr_value])
                        print(f"{speaker_id}/{fname} → SNR: {snr_value} dB")
                    else:
                        print(f"{speaker_id}/{fname} skipped (silent or too short)")
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

    with open(output_csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["SpeakerID", "Filename", "SNR_dB"])
        writer.writerows(results)

    print(f"\n✅ Done. Output saved to: {output_csv_path}")

# --- Run the processing ---

if __name__ == "__main__":
    mps_dataset_path = "/home/drsandipan/Desktop/VTLN-Experiment/mps_dataset/Final_MPS_Dataset_EAAI/"
    output_csv = "MPS_shortlisted_files_based_on_SNR_15db.csv"
    process_mps_dataset(mps_dataset_path, output_csv)

