import os
import re
import torch
import whisper
import numpy as np
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier

# === Setup ===
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"🖥️ Using device: {device}")

# === Load Models ===
print("🔄 Loading Whisper large-v3 model...")
whisper_model = whisper.load_model("large-v3").to(device)

print("🔄 Loading ECAPA-TDNN model...")
ecapa_model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

# === Paths ===
input_dir = "/home/Sharedata/sandipan/Voice_Editing_VTLN/VTLN-Experiment/EAAI-Final-Dataset/Final_Speech_Ocean_Dataset_All/Final__Speech_Ocean_McAdam_with_different_alpha_different_pitch_shift"
output_text_dir = "/home/Sharedata/sandipan/Voice_Editing_VTLN/VTLN-Experiment/EAAI-Final-Dataset/CSV_Files/Executed_Again_Speech_Ocean_McAdams_all_alpha_all_pitch"
output_embed_dir = "/home/Sharedata/sandipan/Voice_Editing_VTLN/VTLN-Experiment/EAAI-Final-Dataset/ECAPA-TDNN/Executed_Again_Speech_Ocean_McAdams_all_alpha_all_pitch"

# === Regex for punctuation removal ===
punctuation_pattern = r'[?\/&$#@!&*:"<>.,;\'\-_=+()\[\]{}\\]'
num_to_word = {str(i): word for i, word in enumerate(["ZERO", "ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT", "NINE"])}

def convert_numbers_to_words(text):
    return re.sub(r'\d+', lambda m: ' '.join(num_to_word[d] for d in m.group()), text)

def clean_and_normalize_text(text):
    text = re.sub(punctuation_pattern, "", text)
    text = convert_numbers_to_words(text)
    return text.upper().strip()

def extract_embedding(wav_path):
    signal, fs = torchaudio.load(wav_path)
    emb = ecapa_model.encode_batch(signal).detach().cpu().numpy()
    return emb.flatten()

# === Processing ===
for shift_folder in sorted(os.listdir(input_dir)):
    shift_path = os.path.join(input_dir, shift_folder)
    if not os.path.isdir(shift_path):
        continue

    for warp_folder in sorted(os.listdir(shift_path)):
        warp_path = os.path.join(shift_path, warp_folder)
        if not os.path.isdir(warp_path):
            continue

        # Output text and embeddings path
        output_txt_path = os.path.join(output_text_dir, shift_folder, warp_folder, "decoded.txt")
        output_embed_base = os.path.join(output_embed_dir, shift_folder, warp_folder)
        os.makedirs(os.path.dirname(output_txt_path), exist_ok=True)
        os.makedirs(output_embed_base, exist_ok=True)

        # Load previously decoded utterance IDs
        decoded_utterances = set()
        if os.path.exists(output_txt_path):
            with open(output_txt_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        utt_id = line.strip().split("\t")[0]
                        decoded_utterances.add(utt_id)

        with open(output_txt_path, "a", encoding="utf-8") as out_file:

            for speaker_folder in sorted(os.listdir(warp_path)):
                speaker_path = os.path.join(warp_path, speaker_folder)
                if not os.path.isdir(speaker_path):
                    continue

                for fname in sorted(os.listdir(speaker_path)):
                    if not fname.lower().endswith(".wav"):
                        continue

                    wav_path = os.path.join(speaker_path, fname)
                    base_name = os.path.splitext(fname)[0]

                    embed_path = os.path.join(output_embed_base, base_name + ".npy")
                    already_decoded = base_name in decoded_utterances
                    already_embedded = os.path.exists(embed_path)

                    if already_decoded and already_embedded:
                        continue  # Skip if both outputs already exist

                    try:
                        if not already_decoded:
                            # === Whisper ===
                            result = whisper_model.transcribe(wav_path, language="en", task="transcribe")
                            raw_text = result["text"].strip()
                            normalized_text = clean_and_normalize_text(raw_text)

                            if normalized_text:
                                out_file.write(f"{base_name}\t{normalized_text}\n")
                                out_file.flush()
                                print(f"📝 [{shift_folder}/{warp_folder}] {base_name} → {normalized_text}")
                            else:
                                print(f"⚠️ Skipped (empty after cleaning): {base_name}")

                    except Exception as e:
                        print(f"❌ Whisper failed for {base_name}: {e}")

                    try:
                        if not already_embedded:
                            embedding = extract_embedding(wav_path)
                            np.save(embed_path, embedding)
                            print(f"🎯 Embedding saved → {embed_path}")

                    except Exception as e:
                        print(f"❌ ECAPA embedding failed for {base_name}: {e}")

print("\n✅ Resumable decoding & ECAPA embedding extraction completed.")
