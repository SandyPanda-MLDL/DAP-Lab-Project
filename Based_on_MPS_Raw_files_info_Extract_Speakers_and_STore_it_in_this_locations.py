import os
import csv
import shutil
from pathlib import Path

# === Paths (Update these) ===
csv_file = '/home/Sharedata/sandipan/Voice_Editing_VTLN/VTLN-Experiment/EAAI-Final-Dataset/MPS_Raw_files_shortlisted_files_based_on_SNR_15db.csv'
source_root = Path('/home/Sharedata/sandipan/Voice_Editing_VTLN/VTLN-Experiment/SNR_Experiment/MPS_Enhanced_DNS_64/')  # <-- No speaker_id folder now
target_root = Path('/home/Sharedata/sandipan/Voice_Editing_VTLN/VTLN-Experiment/EAAI-Final-Dataset/Final_MPS_Speakers')  # Update this to desired copy location

# Create target root if it doesn't exist
target_root.mkdir(parents=True, exist_ok=True)

# Read the CSV and process each row
with open(csv_file, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        speaker_id = row['SpeakerID']
        filename = row['Filename']

        # Determine top-level folder based on first character of SpeakerID
        top_folder = speaker_id[0]

        # Updated source path: /source_root/top_folder/PCM/filename
        source_path = source_root / top_folder / 'PCM' / filename

        # Target folder: /target_root/top_folder/
        target_folder = target_root / top_folder
        target_path = target_folder / filename

        # Create target subfolder if needed
        target_folder.mkdir(parents=True, exist_ok=True)

        # Copy the file
        if source_path.exists():
            shutil.copy2(source_path, target_path)
            print(f"Copied: {source_path} -> {target_path}")
        else:
            print(f"File not found: {source_path}")

