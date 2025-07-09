#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import librosa
import soundfile as sf
import scipy.signal


def anonym_v2(freq, samples, winLengthinms=20, shiftLengthinms=10, lp_order=20, mcadams=0.8):
    eps = np.finfo(np.float32).eps
    samples = samples + eps

    winlen = int(np.floor(winLengthinms * 0.001 * freq))
    shift = int(np.floor(shiftLengthinms * 0.001 * freq))
    length_sig = len(samples)

    wPR = np.hanning(winlen)
    K = np.sum(wPR) / shift
    win = np.sqrt(wPR / K)

    frames = librosa.util.frame(samples, frame_length=winlen, hop_length=shift).T
    windowed_frames = frames * win
    nframe = windowed_frames.shape[0]

    lpc_coefs = librosa.core.lpc(windowed_frames + eps, order=lp_order, axis=1)
    ar_poles = np.array([scipy.signal.tf2zpk(np.array([1]), x)[1] for x in lpc_coefs])

    def _mcadam_angle(poles, mcadams):
        old_angles = np.angle(poles)
        new_angles = np.copy(old_angles)
        real_idx = ~np.isreal(poles)
        neg_idx = np.bitwise_and(real_idx, old_angles < 0.0)
        pos_idx = np.bitwise_and(real_idx, old_angles > 0.0)
        new_angles[neg_idx] = -((-old_angles[neg_idx]) ** mcadams)
        new_angles[pos_idx] = old_angles[pos_idx] ** mcadams
        return new_angles

    def _new_poles(old_poles, new_angles):
        return np.abs(old_poles) * np.exp(1j * new_angles)

    def _lpc_ana_syn(old_lpc_coef, new_lpc_coef, data):
        res = scipy.signal.lfilter(old_lpc_coef, np.array(1), data)
        return scipy.signal.lfilter(np.array([1]), new_lpc_coef, res)

    pole_new_angles = np.array([_mcadam_angle(poles, mcadams) for poles in ar_poles])
    poles_new = np.array([_new_poles(ar_poles[i], pole_new_angles[i]) for i in range(nframe)])

    recon_frames = [
        _lpc_ana_syn(lpc_coefs[i], np.real(np.poly(poles_new[i])), windowed_frames[i])
        for i in range(nframe)
    ]
    recon_frames = np.stack(recon_frames, axis=0) * win

    anonymized_data = np.zeros(length_sig)
    for i in range(nframe):
        start = i * shift
        end = start + winlen
        if end > length_sig:
            break
        anonymized_data[start:end] += recon_frames[i, :end - start]

    return anonymized_data


def apply_mcadams_to_file(input_path, output_path, mcadams=0.8, winLengthinms=20, shiftLengthinms=10, lp_order=20):
    samples, sr = librosa.load(input_path, sr=None)
    anon = anonym_v2(
        freq=sr,
        samples=samples,
        winLengthinms=winLengthinms,
        shiftLengthinms=shiftLengthinms,
        lp_order=lp_order,
        mcadams=mcadams,
    )
    anon = anon / np.max(np.abs(anon)) * 0.99
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sf.write(output_path, anon, sr)


def process_folder_recursively(input_dir, output_dir, mcadams=0.8):
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(".wav"):
                input_path = os.path.join(root, file)
                rel_path = os.path.relpath(input_path, input_dir)
                output_path = os.path.join(output_dir, rel_path)
                print(f"Processing {input_path} -> {output_path}")
                apply_mcadams_to_file(input_path, output_path, mcadams=mcadams)
    print("✅ All files processed.")


if __name__ == "__main__":
    # === CHANGE THESE PATHS ===
    input_base_dir = "/path/to/input/folder"     # <-- change to your source
    output_base_dir = "/path/to/output/folder"   # <-- same structure, same names
    mcadams_value = 0.8                          # Try 0.5 to 1.2

    process_folder_recursively(input_base_dir, output_base_dir, mcadams=mcadams_value)

