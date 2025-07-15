import os
import numpy as np
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

# ========== CONFIG ==========
original_audio_path = "/home/Sharedata/sandipan/Voice_Editing_VTLN/VTLN-Experiment/EAAI-Final-Dataset/Final_MPS_Dataset_All/MPS-Raw-Data-20-40dB-150-files"
original_embedding_path = "/home/Sharedata/sandipan/Voice_Editing_VTLN/VTLN-Experiment/EAAI-Final-Dataset/ECAPA-TDNN/MPS-Raw-Data-20-40dB-150-files"

variant_paths = {
    "McAdams": "/home/Sharedata/sandipan/Voice_Editing_VTLN/VTLN-Experiment/EAAI-Final-Dataset/Final_MPS_Dataset_All/MPS-McAdams-0.80_1_1.15_Raw-Data-20-40dB-150-files",
    "VTLN": "/home/Sharedata/sandipan/Voice_Editing_VTLN/VTLN-Experiment/EAAI-Final-Dataset/Final_MPS_Dataset_All/MPS-VTLN-Raw-Data-20-40dB-150-files_0.80_1.0_1.15"
}

output_embedding_root = "/home/Sharedata/sandipan/Voice_Editing_VTLN/VTLN-Experiment/EAAI-Final-Dataset/ECAPA-TDNN/MPS_McAdams_VTLN_-20-40dB-150-files"
eer_output_root = "/home/Sharedata/sandipan/Voice_Editing_VTLN/VTLN-Experiment/EAAI-Final-Dataset/CSV_Files/MPS_McAdams_VTLN_-20-40dB-150-files"

# ========== Load ECAPA-TDNN ==========
classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

def extract_embedding(wav_path):
    signal, fs = torchaudio.load(wav_path)
    emb = classifier.encode_batch(signal).detach().cpu().numpy()
    return emb.flatten()

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)

def extract_all_embeddings(audio_root, embedding_root):
    for root, _, files in os.walk(audio_root):
        for file in files:
            if file.endswith(".wav"):
                wav_path = os.path.join(root, file)
                rel_path = os.path.relpath(wav_path, audio_root)
                out_path = os.path.join(embedding_root, rel_path.replace(".wav", ".npy"))
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                if not os.path.exists(out_path):
                    emb = extract_embedding(wav_path)
                    np.save(out_path, emb)
                    print(f"✅ Saved: {out_path}")

def load_embeddings_from_folder(folder):
    embeddings = {}
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith(".npy"):
                path = os.path.join(root, file)
                speaker_id = os.path.basename(file).split("_")[0]
                utt_id = file.replace(".npy", "")
                emb = np.load(path)
                embeddings[(speaker_id, utt_id)] = emb
    return embeddings

def compute_eer(original_embeddings, variant_embeddings):
    scores = []
    labels = []
    results = []

    for (mod_spk, mod_utt), mod_emb in variant_embeddings.items():
        for (orig_spk, orig_utt), orig_emb in original_embeddings.items():
            sim = cosine_similarity(mod_emb, orig_emb)
            label = 1 if mod_spk == orig_spk else 0
            results.append(f"{mod_spk}\t{mod_utt}\t{orig_spk}\t{orig_utt}\t{sim:.4f}\t{label}")
            scores.append(sim)
            labels.append(label)

    fpr, tpr, _ = roc_curve(labels, scores)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer, results

# ========== Process Original ==========
print("\n🔍 Extracting original embeddings...")
extract_all_embeddings(original_audio_path, original_embedding_path)
original_embs = load_embeddings_from_folder(original_embedding_path)

# ========== Process Variants ==========
for algo, path in variant_paths.items():
    for warp in sorted(os.listdir(path)):
        warp_path = os.path.join(path, warp)
        if not os.path.isdir(warp_path):
            continue

        print(f"\n🔁 Extracting embeddings for {algo} | Warp: {warp}")
        variant_emb_path = os.path.join(output_embedding_root, f"{algo}_wpf_{warp}")
        extract_all_embeddings(warp_path, variant_emb_path)

        print(f"📥 Loading variant embeddings for {algo}-{warp}")
        variant_embs = load_embeddings_from_folder(variant_emb_path)

        print("📊 Computing EER...")
        eer, result_lines = compute_eer(original_embs, variant_embs)

        out_txt = os.path.join(eer_output_root, f"{algo}_Warp_{warp}_EER_cosine.txt")
        os.makedirs(os.path.dirname(out_txt), exist_ok=True)
        with open(out_txt, "w", encoding="utf-8") as f:
            f.write("modified_speaker\tmodified_utt\toriginal_speaker\toriginal_utt\tcosine_similarity\tlabel\n")
            f.write("\n".join(result_lines))
            f.write(f"\n\n=== Summary ===\nEqual Error Rate (EER): {eer:.4f}\n")
        print(f"✅ EER saved for {algo} warp={warp}: {eer:.4f}")

#print("\n🎉 All done.")
