import os
import re
import numpy as np
from jiwer import process_words
from difflib import SequenceMatcher

# === File Paths ===
gt_file = "/home/Sharedata/sandipan/Voice_Editing_VTLN/VTLN-Experiment/EAAI-Final-Dataset/CSV_Files/GT_Manual_Transcript_of_MPS_Data.txt"
original_decoded_file = "/home/Sharedata/sandipan/Voice_Editing_VTLN/VTLN-Experiment/EAAI-Final-Dataset/CSV_Files/MPS-Decoded-Text/Whisper_large_V3/Decoded-Texts/20-40dB-Decoded-Text/Original_Decoded/Original_Audios_Decoded_Text.txt"
modified_decoded_root = "/home/Sharedata/sandipan/Voice_Editing_VTLN/VTLN-Experiment/EAAI-Final-Dataset/CSV_Files/MPS-Decoded-Text/Whisper_large_V3/Decoded-Texts/20-40dB-Decoded-Text"
modified_output_root = os.path.join(modified_decoded_root, "Best_WER_Output_Experiment_Basis")

os.makedirs(modified_output_root, exist_ok=True)

# === Utilities ===
def extract_story_id(utt_id):
    match = re.search(r"EN-OL-RC-(\d+_\d+)", utt_id)
    return match.group(1) if match else None

def extract_speaker_id(utt_id):
    return utt_id.split("_EN-OL-RC")[0] if "_EN-OL-RC" in utt_id else utt_id

def load_file_to_dict(path):
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                parts = line.strip().split("\t", 1)
                if len(parts) == 2:
                    utt_id, text = parts
                    data[utt_id] = text.strip().lower()
    return data

def analyze_edits(ref, hyp):
    ref_words = ref.split()
    hyp_words = hyp.split()
    matcher = SequenceMatcher(None, ref_words, hyp_words)

    substitutions, deletions, insertions = [], [], []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace':
            len_ref = i2 - i1
            len_hyp = j2 - j1
            min_len = min(len_ref, len_hyp)

            for k in range(min_len):
                substitutions.append((ref_words[i1 + k], hyp_words[j1 + k]))

            # Extra deletions from ref
            deletions.extend(ref_words[i1 + min_len:i2])
            # Extra insertions from hyp
            insertions.extend(hyp_words[j1 + min_len:j2])

        elif tag == 'delete':
            deletions.extend(ref_words[i1:i2])

        elif tag == 'insert':
            insertions.extend(hyp_words[j1:j2])

    return substitutions, deletions, insertions

# === Compute WER Dict and Return Detailed Info ===
def compute_wer_detailed(ref_dict, hyp_dict):
    total_S = total_D = total_I = total_H = total_N = 0
    details = {}

    for utt_id in ref_dict:
        if utt_id not in hyp_dict:
            continue
        ref = ref_dict[utt_id]
        hyp = hyp_dict[utt_id]
        measures = process_words(ref, hyp)
        subs, dels, ins = analyze_edits(ref, hyp)

        S = measures.substitutions
        D = measures.deletions
        I = measures.insertions
        H = measures.hits
        N = S + D + H
        wer = (S + D + I) / N if N > 0 else 0

        details[utt_id] = {
            'ref': ref, 'hyp': hyp, 'S': S, 'D': D, 'I': I, 'H': H, 'N': N, 'WER': wer,
            'subs': subs, 'dels': dels, 'ins': ins
        }

        total_S += S
        total_D += D
        total_I += I
        total_H += H
        total_N += N

    avg_wer = (total_S + total_D + total_I) / total_N if total_N > 0 else 0

    summary = {
        'total_S': total_S, 'total_D': total_D, 'total_I': total_I,
        'total_H': total_H, 'total_N': total_N, 'avg_WER': avg_wer
    }

    return details, summary

# === Save comparison for modified file ===
def save_comparison_report(mod_name, orig_details, mod_details, orig_summary, mod_summary, output_path):
    with open(output_path, "w", encoding="utf-8") as out_f:
        for utt_id in sorted(mod_details):
            if utt_id not in orig_details:
                continue

            mod = mod_details[utt_id]
            orig = orig_details[utt_id]

            story_id = extract_story_id(utt_id)
            speaker_id = extract_speaker_id(utt_id)

            out_f.write(f"Utterance ID: {utt_id}\n")
            out_f.write(f"Story ID: {story_id}\n")
            out_f.write(f"Speaker ID: {speaker_id}\n\n")

            # Original
            out_f.write("== Original vs GT ==\n")
            out_f.write(f"WER: {orig['WER']:.4f}\nSubstitutions: {orig['S']}, Deletions: {orig['D']}, Insertions: {orig['I']}, Hits: {orig['H']}\n")
            out_f.write("Substituted Words:\n" + "\n".join(f"  {r} => {h}" for r, h in orig['subs']) + "\n")
            out_f.write("Deleted Words:\n" + "\n".join(f"  {w}" for w in orig['dels']) + "\n")
            out_f.write("Inserted Words:\n" + "\n".join(f"  {w}" for w in orig['ins']) + "\n\n")

            # Modified
            out_f.write("== Modified vs GT ==\n")
            out_f.write(f"WER: {mod['WER']:.4f}\nSubstitutions: {mod['S']}, Deletions: {mod['D']}, Insertions: {mod['I']}, Hits: {mod['H']}\n")
            out_f.write("Substituted Words:\n" + "\n".join(f"  {r} => {h}" for r, h in mod['subs']) + "\n")
            out_f.write("Deleted Words:\n" + "\n".join(f"  {w}" for w in mod['dels']) + "\n")
            out_f.write("Inserted Words:\n" + "\n".join(f"  {w}" for w in mod['ins']) + "\n\n")

            wer_diff = mod['WER'] - orig['WER']
            out_f.write(f"Relative WER Difference (Modified - Original): {wer_diff:+.4f}\n")
            out_f.write("=" * 60 + "\n\n")

        # === Final Summary ===
        out_f.write("### Summary Over All Utterances ###\n")
        out_f.write(f"Original Avg WER: {orig_summary['avg_WER']:.4f}\n")
        out_f.write(f"Modified Avg WER: {mod_summary['avg_WER']:.4f}\n")
        out_f.write(f"Relative Avg WER Change: {(mod_summary['avg_WER'] - orig_summary['avg_WER']):+.4f}\n")
        out_f.write("=" * 60 + "\n")

    print(f"✅ Saved comparison report: {output_path}")

# === Load all files ===
gt_dict = load_file_to_dict(gt_file)
original_dict = load_file_to_dict(original_decoded_file)

# === Compute original WER once ===
original_details, original_summary = compute_wer_detailed(gt_dict, original_dict)

# === For each modified file ===
for mod_file in os.listdir(modified_decoded_root):
    if mod_file.endswith(".txt") and "Original_Audios_Decoded_Text.txt" not in mod_file:
        mod_path = os.path.join(modified_decoded_root, mod_file)
        mod_dict = load_file_to_dict(mod_path)
        mod_name = os.path.splitext(mod_file)[0]

        # Compute WER details for modified
        mod_details, mod_summary = compute_wer_detailed(gt_dict, mod_dict)

        # Save detailed comparison report
        output_path = os.path.join(modified_output_root, f"WER_Comparison_{mod_name}.txt")
        save_comparison_report(mod_name, original_details, mod_details, original_summary, mod_summary, output_path)
