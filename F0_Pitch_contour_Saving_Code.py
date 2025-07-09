import os
import csv
import numpy as np
import parselmouth

def extract_f0_parselmouth(wav_path,
                           time_step=0.01,
                           f0_min=150,
                           f0_max=500,
                           silence_threshold=0.03,
                           voicing_threshold=0.5,
                           octave_cost=0.01,
                           octave_jump_cost=0.35,
                           voiced_unvoiced_cost=0.14,
                           max_candidates=15):
    snd = parselmouth.Sound(wav_path)
    pitch = parselmouth.praat.call(
        snd, "To Pitch (ac)...",
        time_step, f0_min, max_candidates, f0_max,
        silence_threshold, voicing_threshold,
        octave_cost, octave_jump_cost, voiced_unvoiced_cost, f0_max
    )
    return pitch.selected_array['frequency']

def extract_all_f0_to_csv(input_folder, output_csv):
    """
    Walks through input_folder, finds all .wav files, extracts pitch contour,
    and writes a CSV with columns [filename, f0_contour].
    f0_contour is stored as semicolon-separated floats.
    """
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['filename', 'f0_contour'])
        
        for root, _, files in os.walk(input_folder):
            for fname in files:
                if not fname.lower().endswith('.wav'):
                    continue
                wav_path = os.path.join(root, fname)
                try:
                    f0 = extract_f0_parselmouth(wav_path)
                except Exception as e:
                    print(f"Error extracting F0 from {wav_path}: {e}")
                    continue
                # convert to string: "f0_1;f0_2;...;f0_n"
                f0_str = ';'.join(f"{v:.2f}" for v in f0)
                writer.writerow([fname, f0_str])
                print(f"Processed {fname}, {len(f0)} frames")

if __name__ == "__main__":
    input_folder = "/home/drsandipan/Desktop/VTLN-Experiment/Speech_Ocean_Dataset/Final_Speech_Ocean_Dataset/"       # update this
    output_csv   = "/home/drsandipan/Desktop/VTLN-Experiment/Speech_Ocean_Dataset/Speech_Ocean_pitch_contours.csv" 
    extract_all_f0_to_csv(input_folder, output_csv)
    print(f"All contours saved to {output_csv}")

