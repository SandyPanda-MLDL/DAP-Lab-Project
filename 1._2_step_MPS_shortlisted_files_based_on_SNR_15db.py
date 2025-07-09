import os
import csv
import shutil
from pathlib import Path

# Paths
csv_file = '/home/drsandipan/Desktop/VTLN-Experiment/mps_dataset/MPS_SNR_Results.csv'
source_root = Path('/home/drsandipan/Desktop/VTLN-Experiment/MPS_Dataset/')
target_root = Path('/home/drsandipan/Desktop/VTLN-Experiment/mps_dataset/Final_MPS_Dataset_EAAI/')

# Ensure the target root exists
target_root.mkdir(parents=True, exist_ok=True)

# Track which folders are created
created_speakers = set()

# Read and process CSV
with open(csv_file, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        speaker_id = row['SpeakerID']
        filename = row['Filename']
        try:
            snr = float(row['SNR_dB'])
        except ValueError:
            print(f"Invalid SNR for {filename}, skipping.")
            continue

        if snr >= 15.0:
            source_file = source_root / speaker_id / filename
            target_speaker_folder = target_root / speaker_id

            # Create target speaker folder if not already done
            if speaker_id not in created_speakers:
                target_speaker_folder.mkdir(parents=True, exist_ok=True)
                created_speakers.add(speaker_id)

            if source_file.exists():
                target_file = target_speaker_folder / filename
                shutil.copy2(source_file, target_file)
                print(f"Copied: {source_file} -> {target_file}")
            else:
                print(f"File not found: {source_file}")

