#Extracts 10 speakers (age 6–10)

#Gets their gender

#Computes SNR for each .wav file

#Selects 15 files with SNR ≥ 15 dB per speaker

#Copies them to a new folder

#Extracts corresponding transcription

#Cleans text and stores everything into a CSV with SNR


import os
import random
import shutil
import csv
import re
import numpy as np
import wave
import soundfile as sf
import librosa
import pandas as pd

# ------------------------------------------------------------------------------------
# (1) Load speaker age info, filter 10 speakers aged 6–10
# ------------------------------------------------------------------------------------
spk2age_path = "/home/drsandipan/Desktop/VTLN-Experiment/Speech_Ocean_Dataset/test/spk2age"
spk2gender_path = "/home/drsandipan/Desktop/VTLN-Experiment/Speech_Ocean_Dataset/test/spk2gender"
text_file_path = "/home/drsandipan/Desktop/VTLN-Experiment/Speech_Ocean_Dataset/test/text"
source_root_dir = "/home/drsandipan/Desktop/VTLN-Experiment/Speech_Ocean_Dataset/WAVE"
destination_root_dir = "/home/drsandipan/Desktop/VTLN-Experiment/Speech_Ocean_Dataset/EAAI_Final_Dataset/"
output_csv_path = os.path.join(destination_root_dir, "Speech_1_Ocean_EAAI_test_speaker_data_10_spk_15_uttr.csv")

def clean_text(text):
    return re.sub(r"[\'\"\?\&\*\!]", "", text)

age_6_10_speakers = []
with open(spk2age_path, "r") as f:
    for line in f:
        spk_id, age = line.strip().split()
        if 6 <= int(age) <= 10:
            age_6_10_speakers.append((spk_id, int(age)))
        if len(age_6_10_speakers) == 10:
            break

selected_spk_ids = {spk for spk, _ in age_6_10_speakers}
spk_age_map = dict(age_6_10_speakers)

# ------------------------------------------------------------------------------------
# (2) Load gender information for selected speakers
# ------------------------------------------------------------------------------------
spk_gender_map = {}
with open(spk2gender_path, "r") as f:
    for line in f:
        spk_id, gender = line.strip().split()
        if spk_id in selected_spk_ids:
            spk_gender_map[spk_id] = gender

# ------------------------------------------------------------------------------------
# (3) Load transcription mapping (utterance ID → text)
# ------------------------------------------------------------------------------------
utt_text_map = {}
with open(text_file_path, "r") as f:
    for line in f:
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2:
            utt_id, transcript = parts
            utt_text_map[utt_id] = clean_text(transcript)

# ------------------------------------------------------------------------------------
# (4) SNR Utility Functions
# ------------------------------------------------------------------------------------
db_vals = np.arange(-20, 101)
g_vals = np.array([0.40974774, 0.40986926, 0.40998566, 0.40969089, 0.40986186, 0.40999006, 0.41027138, 0.41052627, 0.41101024, 0.41143264, 0.41231718, 0.41337272, 0.41526426, 0.4178192 , 0.42077252, 0.42452799, 0.42918886, 0.43510373, 0.44234195, 0.45161485, 0.46221153, 0.47491647, 0.48883809, 0.50509236, 0.52353709, 0.54372088, 0.56532427, 0.58847532, 0.61346212, 0.63954496, 0.66750818, 0.69583724, 0.72454762, 0.75414799, 0.78323148, 0.81240985, 0.84219775, 0.87166406, 0.90030504, 0.92880418, 0.95655449, 0.9835349 , 1.01047155, 1.0362095 , 1.06136425, 1.08579312, 1.1094819 , 1.13277995, 1.15472826, 1.17627308, 1.19703503, 1.21671694, 1.23535898, 1.25364313, 1.27103891, 1.28718029, 1.30302865, 1.31839527, 1.33294817, 1.34700935, 1.3605727 , 1.37345513, 1.38577122, 1.39733504, 1.40856397, 1.41959619, 1.42983624, 1.43958467, 1.44902176, 1.45804831, 1.46669568, 1.47486938, 1.48269965, 1.49034339, 1.49748214, 1.50435106, 1.51076426, 1.51698915, 1.5229097 , 1.528578  , 1.53389835, 1.5391211 , 1.5439065 , 1.54858517, 1.55310776, 1.55744391, 1.56164927, 1.56566348, 1.56938671, 1.57307767, 1.57654764, 1.57980083, 1.58304129, 1.58602496, 1.58880681, 1.59162477, 1.5941969 , 1.59693155, 1.599446  , 1.60185011, 1.60408668, 1.60627134, 1.60826199, 1.61004547, 1.61192472, 1.61369656, 1.61534074, 1.61688905, 1.61838916, 1.61985374, 1.62135878, 1.62268119, 1.62390423, 1.62513143, 1.62632463, 1.6274027 , 1.62842767, 1.62945532, 1.6303307 , 1.63128026, 1.63204102])  # Use full g_vals array here (same as previous version)

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

def compute_window_snr(audio_path, min_duration=0.5):
    try:
        with wave.open(audio_path, 'rb') as wav:
            if wav.getsampwidth() != 2:
                return None
    except:
        return None

    audio, sample_rate = sf.read(audio_path)
    if not complete_silence_check(audio):
        return None

    audio_duration = len(audio) / sample_rate
    if audio_duration < min_duration:
        return None

    snr_list = []
    for j in range(0, int(audio_duration)):
        try:
            a, sr = librosa.load(audio_path, offset=j, duration=6.0, sr=None)
            snr_moving = wada_snr(a)
            snr_list.append(np.nanmean(snr_moving))
        except:
            continue

    if len(snr_list) > 1:
        snr_list.pop(0)
    if not snr_list:
        return None
    return round(np.nanmean(snr_list), 2)

# ------------------------------------------------------------------------------------
# (5) For each speaker, select 15 .wav files with SNR ≥ 15 and copy them
# ------------------------------------------------------------------------------------
if not os.path.exists(destination_root_dir):
    os.makedirs(destination_root_dir)

final_records = []

for spk_id in selected_spk_ids:
    folder_name = f"SPEAKER{spk_id}"
    folder_path = os.path.join(source_root_dir, folder_name)
    if not os.path.exists(folder_path):
        print(f"Skipping missing speaker folder: {folder_path}")
        continue

    wav_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".wav")]
    random.shuffle(wav_files)

    selected_files = []
    for f in wav_files:
        if len(selected_files) == 15:
            break
        audio_path = os.path.join(folder_path, f)
        snr_val = compute_window_snr(audio_path)
        if snr_val is not None and snr_val >= 15:
            selected_files.append((f, snr_val))

    if len(selected_files) < 15:
        print(f"Speaker {spk_id} has only {len(selected_files)} good SNR files, skipping")
        continue

    dest_spk_folder = os.path.join(destination_root_dir, folder_name)
    os.makedirs(dest_spk_folder, exist_ok=True)

    for filename, snr_val in selected_files:
        src = os.path.join(folder_path, filename)
        dst = os.path.join(dest_spk_folder, filename)
        shutil.copyfile(src, dst)

        utt_id = os.path.splitext(filename)[0]
        text = utt_text_map.get(utt_id, "")
        final_records.append([
            folder_name,
            spk_age_map[spk_id],
            spk_gender_map.get(spk_id, "NA"),
            filename,
            text,
            snr_val
        ])

# ------------------------------------------------------------------------------------
# (6) Write final CSV with speaker_name, age, gender, audio_file, text, snr
# ------------------------------------------------------------------------------------
with open(output_csv_path, "w", newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["speaker_name", "age", "gender", "audio_file", "text", "snr"])
    writer.writerows(final_records)

print(f"\n✅ Done. Final CSV saved at:\n{output_csv_path}")

