#!/usr/bin/env python3
"""
Interactive CLI for Real-Time Speaker Identification & Voice Biometrics
========================================================================
Identifies speakers in audio files or streaming sessions against the
enrolled voiceprint gallery. Automatically detects unauthorized impostors.

Usage:
    # 1. Identify speaker in an audio file
    python identify_speaker.py --audio outputs/sample_alice.wav
    python identify_speaker.py --audio outputs/sample_intruder.wav

    # 2. Real-time streaming session monitoring with live decisions
    python identify_speaker.py --stream outputs/sample_session.wav

    # 3. Enroll a new speaker voiceprint into the gallery
    python identify_speaker.py --enroll "Frank" --audio outputs/sample_intruder.wav
"""

import os
import sys
import argparse
import json
import numpy as np
import soundfile as sf
import torch

from main import (
    VoiceIDNet,
    LogMelExtractor,
    SpeakerGallery,
    SAMPLE_RATE,
    WINDOW_SEC,
    STEP_SEC,
    WINDOW_FRAMES,
    AUTH_THRESHOLD,
    simulate_streaming_identification
)


def load_audio(audio_path: str) -> np.ndarray:
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    data, sr = sf.read(audio_path, dtype='float32')
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    if sr != SAMPLE_RATE:
        import torchaudio
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
        tensor_data = torch.from_numpy(data).unsqueeze(0)
        data = resampler(tensor_data).squeeze(0).numpy()
    return data


def load_gallery(gallery_path: str) -> SpeakerGallery:
    gallery = SpeakerGallery()
    if not os.path.exists(gallery_path):
        raise FileNotFoundError(f"Gallery not found at: {gallery_path}. Run main.py first!")

    with open(gallery_path, 'r') as f:
        data = json.load(f)
    for name, vec in data['voiceprints'].items():
        gallery.voiceprints[name] = np.array(vec, dtype=np.float32)
    return gallery


def identify_single_audio(audio_path: str, model, extractor, gallery: SpeakerGallery, threshold: float):
    audio = load_audio(audio_path)
    dur_sec = len(audio) / SAMPLE_RATE

    win_samples = int(WINDOW_SEC * SAMPLE_RATE)
    # Take center or first window
    if len(audio) >= win_samples:
        start = max(0, (len(audio) - win_samples) // 2)
        chunk = audio[start:start + win_samples]
    else:
        chunk = np.pad(audio, (0, win_samples - len(audio)), mode='constant')

    spec = extractor(torch.from_numpy(chunk))[:, :, :WINDOW_FRAMES]
    with torch.no_grad():
        emb, _ = model(spec.unsqueeze(0))
        emb_np = emb.squeeze(0).numpy()

    best_spk, score, is_auth, all_scores = gallery.match(emb_np, threshold=threshold)

    print("\n" + "=" * 62)
    print("      VOICEIDNET BIOMETRIC SPEAKER IDENTIFICATION")
    print("=" * 62)
    print(f"Audio File       : {audio_path}")
    print(f"Duration         : {dur_sec:.2f} seconds ({len(audio):,} samples)")
    print(f"Auth Threshold   : {threshold:.2f} (Cosine Similarity)")
    print("-" * 62)

    status_tag = "✓ ACCESS GRANTED" if is_auth else "✗ ACCESS DENIED (IMPOSTOR)"
    print(f"\nIDENTIFIED SPEAKER : {best_spk.upper()}")
    print(f"BIOMETRIC STATUS   : {status_tag} (Score: {score:.3f})\n")

    print(f"{'Rank':<6}{'Enrolled Voice':<18}{'Cosine Sim':<14}{'Confidence Meter'}")
    print("-" * 62)
    sorted_scores = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
    for rank, (name, s_val) in enumerate(sorted_scores, 1):
        bar_len = int(max(0.0, s_val) * 28)
        bar = "█" * bar_len + "░" * (28 - bar_len)
        flag = " [MATCH]" if (name == best_spk and is_auth) else ""
        print(f"#{rank:<5}{name:<18}{s_val:>7.3f}       |{bar}|{flag}")

    print("=" * 62 + "\n")


def stream_session_monitor(stream_path: str, model, extractor, gallery: SpeakerGallery):
    audio = load_audio(stream_path)
    dur_sec = len(audio) / SAMPLE_RATE

    print("\n" + "=" * 68)
    print("      REAL-TIME STREAMING VOICE IDENTIFICATION MONITOR")
    print("=" * 68)
    print(f"Stream Audio     : {stream_path}")
    print(f"Stream Duration  : {dur_sec:.2f} seconds")
    print(f"Decision Rate    : 4 decisions/sec (Every {STEP_SEC*1000:.0f} ms)")
    print("-" * 68)
    print(f"{'Time Window':<20}{'Speaker':<18}{'Score':<10}{'Access Status':<16}")
    print("-" * 68)

    decisions, latencies = simulate_streaming_identification(audio, model, extractor, gallery)
    for d in decisions[::2]:  # Print alternating for readability
        time_str = f"[{d['start_sec']:05.2f}s -> {d['end_sec']:05.2f}s]"
        spk_str = d['speaker']
        score_str = f"{d['confidence']:>5.3f}"
        status_str = "AUTHORIZED" if d['is_authorized'] else ("SILENCE" if spk_str == "SILENCE" else "DENIED / UNKNOWN")
        print(f"{time_str:<20}{spk_str:<18}{score_str:<10}{status_str:<16}")

    mean_lat = np.mean(latencies)
    print("-" * 68)
    print(f"[✓] Average Decision Latency : {mean_lat:.2f} ms (Budget: {STEP_SEC*1000:.0f} ms)")
    print(f"[✓] Real-Time Factor (RTF)   : {(mean_lat / (STEP_SEC * 1000.0)):.4f}x")
    print("=" * 68 + "\n")


def enroll_speaker(name: str, audio_path: str, model, extractor, gallery_path: str):
    audio = load_audio(audio_path)
    win_samples = int(WINDOW_SEC * SAMPLE_RATE)
    embeddings = []

    for start in range(0, len(audio) - win_samples + 1, win_samples // 2):
        chunk = audio[start:start + win_samples]
        spec = extractor(torch.from_numpy(chunk))[:, :, :WINDOW_FRAMES]
        with torch.no_grad():
            emb, _ = model(spec.unsqueeze(0))
            embeddings.append(emb.squeeze(0).numpy())

    if not embeddings:
        print("[!] Audio too short for enrollment.")
        return

    embeddings = np.array(embeddings)
    centroid = np.mean(embeddings, axis=0)
    centroid = centroid / np.linalg.norm(centroid)

    # Update gallery
    with open(gallery_path, 'r') as f:
        gallery_data = json.load(f)

    gallery_data['voiceprints'][name] = centroid.tolist()
    if name not in gallery_data['enrolled_speakers']:
        gallery_data['enrolled_speakers'].append(name)

    with open(gallery_path, 'w') as f:
        json.dump(gallery_data, f, indent=2)

    print(f"[✓] Successfully enrolled '{name}' ({len(embeddings)} utterances analyzed).")
    print(f"[✓] Updated gallery saved to: {gallery_path}")


def main():
    parser = argparse.ArgumentParser(description="VoiceIDNet: Speaker Identification CLI")
    parser.add_argument('--audio', type=str, default=None, help='Single audio file to identify')
    parser.add_argument('--stream', type=str, default=None, help='Continuous stream audio to monitor')
    parser.add_argument('--enroll', type=str, default=None, help='Enroll speaker name')
    parser.add_argument('--model', type=str, default='outputs/voice_id_best.pth', help='Model checkpoint')
    parser.add_argument('--gallery', type=str, default='outputs/speaker_gallery.json', help='Gallery JSON')
    parser.add_argument('--threshold', type=float, default=AUTH_THRESHOLD, help='Verification threshold')
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"[!] Model not found: {args.model}. Run main.py first!")
        sys.exit(1)

    device = torch.device('cpu')
    ckpt = torch.load(args.model, map_location=device)
    model = VoiceIDNet().to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    extractor = LogMelExtractor(sample_rate=SAMPLE_RATE)
    gallery = load_gallery(args.gallery)

    if args.enroll:
        if not args.audio:
            print("[!] Please provide --audio file with speaker voice sample to enroll.")
            sys.exit(1)
        enroll_speaker(args.enroll, args.audio, model, extractor, args.gallery)
    elif args.stream:
        stream_session_monitor(args.stream, model, extractor, gallery)
    elif args.audio:
        identify_single_audio(args.audio, model, extractor, gallery, args.threshold)
    else:
        # Default run demo on Alice sample
        demo_sample = 'outputs/sample_alice.wav'
        if os.path.exists(demo_sample):
            identify_single_audio(demo_sample, model, extractor, gallery, args.threshold)
        else:
            parser.print_help()


if __name__ == '__main__':
    main()

