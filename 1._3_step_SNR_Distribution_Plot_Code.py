import csv
import matplotlib.pyplot as plt
import numpy as np

# Path to your CSV file
csv_file = '/home/drsandipan/Desktop/VTLN-Experiment/mps_dataset/MPS_Raw_files_shortlisted_files_based_on_SNR_15db.csv'

# Read SNR values
snr_values = []
with open(csv_file, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            snr = float(row['SNR_dB'])
            snr_values.append(snr)
        except ValueError:
            continue

# Define histogram bins
bins = np.arange(0, max(snr_values) + 5, 5)

# Plot histogram
plt.figure(figsize=(10, 6))
counts, _, patches = plt.hist(snr_values, bins=bins, edgecolor='black', alpha=0.7)

# Annotate bar counts
for count, patch in zip(counts, patches):
    if count > 0:
        plt.text(patch.get_x() + patch.get_width() / 2, count + 0.5, int(count),
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

# Plot settings
plt.title(f'Distribution of SNR Values for MPS Dataset (Total: {len(snr_values)} Samples)', fontsize=14)
plt.xlabel('SNR (dB)', fontsize=12)
plt.ylabel('Number of Samples', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# Save the plot
output_path = '/home/drsandipan/Desktop/VTLN-Experiment/mps_dataset/SNR_Distribution_Plot_Code.png'
plt.savefig(output_path)
plt.close()

print(f"SNR distribution plot saved to: {output_path}")

