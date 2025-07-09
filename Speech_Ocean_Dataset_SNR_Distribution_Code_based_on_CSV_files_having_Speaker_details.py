import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set path to your dataset file
csv_path = '/home/drsandipan/Desktop/VTLN-Experiment/Speech_Ocean_Dataset/Final_Speech_Ocean_Dataset/Speech_3_Ocean_Final_Dataset_EAAI_test_speaker_data_10_spk_15_uttr.csv'  # <-- UPDATE THIS

# Load the dataset
df = pd.read_csv(csv_path, sep=None, engine='python')  # Auto-detects comma or tab separator

# Ensure 'snr' column is present and convert to numeric
df['snr'] = pd.to_numeric(df['snr'], errors='coerce')
df = df.dropna(subset=['snr'])

# Extract SNR values
snr_values = df['snr'].tolist()

# Set bins (you can adjust the range or width as needed)
bins = np.arange(0, max(snr_values) + 10, 10)

# Plot the histogram
plt.figure(figsize=(10, 6))
counts, _, patches = plt.hist(snr_values, bins=bins, edgecolor='black', alpha=0.7)

# Annotate each bin with its count
for count, patch in zip(counts, patches):
    if count > 0:
        plt.text(patch.get_x() + patch.get_width() / 2, count + 0.5,
                 f'{int(count)}', ha='center', va='bottom', fontsize=10)

# Title and labels
plt.title(f'SNR Distribution in Speech Ocean Dataset (Total: {len(snr_values)} Samples)', fontsize=14)
plt.xlabel('SNR (dB)', fontsize=12)
plt.ylabel('Number of Samples', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

# Save the figure
output_file = '/home/drsandipan/Desktop/VTLN-Experiment/Speech_Ocean_Dataset/SNR_Distribution_SpeechOcean.png'  # <-- UPDATE THIS
plt.savefig(output_file)
plt.close()

print(f"Plot saved to: {output_file}")

