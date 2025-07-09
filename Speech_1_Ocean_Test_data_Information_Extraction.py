# ------------------------------------------------------------------------------------
# Speech_Ocean_Test_Data_Analysis
# ------------------------------------------------------------------------------------
# 1. From `spk2age`, extract 10 speakers with age in range 6–10.
# 2. For those speakers, get corresponding gender from `spk2gender`.
# 3. Prefix their numeric IDs with "SPEAKER" → these are speaker folder names.
# 4. For each matching speaker folder:
#     a. Randomly select 15 `.WAV` files.
#     b. Copy them to a new directory, converting the extension to `.wav`.
# 5. For each copied `.wav` file:
#     a. Extract the utterance ID from the filename.
#     b. Use the `text` file to get the corresponding transcription.
#     c. Clean the text (remove special symbols: ' " ? & * !).
# 6. Store: speaker name, age, gender, audio file name, text → in a CSV file.
# ------------------------------------------------------------------------------------

import os
import random
import shutil
import csv
import re

# ----------- Input file paths --------------
spk2age_path = "/home/drsandipan/Desktop/VTLN-Experiment/Speech_Ocean_Dataset/test/spk2age"
spk2gender_path = "/home/drsandipan/Desktop/VTLN-Experiment/Speech_Ocean_Dataset/test/spk2gender"
text_file_path = "/home/drsandipan/Desktop/VTLN-Experiment/Speech_Ocean_Dataset/test/text"
source_root_dir = "/home/drsandipan/Desktop/VTLN-Experiment/Speech_Ocean_Dataset/WAVE"
destination_root_dir = "/home/drsandipan/Desktop/VTLN-Experiment/Speech_Ocean_Dataset/EAAI-Speech_Ocean_10_spk_15_uttr/"
output_csv_path = "/home/drsandipan/Desktop/VTLN-Experiment/Speech_Ocean_Dataset/test/EAAI_test_speaker_data_10_spk_15_uttr.csv"

# ----------- Helper function: Clean text --------------
def clean_text(text):
    return re.sub(r"[\'\"\?\&\*\!]", "", text)

# ----------- Step 1: Get 10 speaker IDs (age 6–10) ----------
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

# ----------- Step 2: Get gender of those speakers ----------
spk_gender_map = {}
with open(spk2gender_path, "r") as f:
    for line in f:
        spk_id, gender = line.strip().split()
        if spk_id in selected_spk_ids:
            spk_gender_map[spk_id] = gender

# ----------- Step 3: Form full speaker folder names ----------
speaker_folders = [f"SPEAKER{spk}" for spk in selected_spk_ids]

# ----------- Step 4: Randomly select 15 WAV files per speaker ----------
if not os.path.exists(destination_root_dir):
    os.makedirs(destination_root_dir)

copied_files = []

for spk in selected_spk_ids:
    folder_name = f"SPEAKER{spk}"
    folder_path = os.path.join(source_root_dir, folder_name)
    if not os.path.isdir(folder_path):
        print(f"Skipping {folder_path}, not found")
        continue

    all_wavs = [f for f in os.listdir(folder_path) if f.lower().endswith(".wav")]
    selected_wavs = random.sample(all_wavs, min(15, len(all_wavs)))

    spk_dest_dir = os.path.join(destination_root_dir, folder_name)
    os.makedirs(spk_dest_dir, exist_ok=True)

    for wav_file in selected_wavs:
        src_path = os.path.join(folder_path, wav_file)
        new_name = os.path.splitext(wav_file)[0] + ".wav"
        dest_path = os.path.join(spk_dest_dir, new_name)
        shutil.copyfile(src_path, dest_path)
        copied_files.append((folder_name, spk, new_name))  # SPEAKERxxxx, numeric id, file name

# ----------- Step 5: Load transcription mapping from `text` ----------
utt_text_map = {}
with open(text_file_path, "r") as f:
    for line in f:
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2:
            utt_id, transcript = parts
            utt_text_map[utt_id] = clean_text(transcript)

# ----------- Step 6: Write final CSV ----------
with open(output_csv_path, "w", newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["speaker_name", "age", "gender", "audio_file", "text"])

    for folder_name, spk_id, audio_file in copied_files:
        utt_id = os.path.splitext(audio_file)[0]
        text = utt_text_map.get(utt_id, "")
        writer.writerow([
            folder_name, 
            spk_age_map.get(spk_id, "NA"), 
            spk_gender_map.get(spk_id, "NA"), 
            audio_file, 
            text
        ])

print(f"✅ Done. Output CSV written to: {output_csv_path}")

