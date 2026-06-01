# src/audio_emotion.py
import librosa, numpy as np
from pathlib import Path
import torch
from silero_vad import get_speech_timestamps, read_audio, VADIterator
# Simple mapping: compute pitch/energy/stats -> map to emotion (placeholder)
def map_audio_features_to_emotion(wav_path):
    # returns dict {"emotion": "sad"|"happy"|... , "features": {...}}
    try:
        y, sr = librosa.load(wav_path, sr=16000)
    except Exception:
        # fallback
        y, sr = librosa.load(wav_path, sr=None)
    # compute RMS energy and zero-crossing rate and tempo
    rms = float(np.mean(librosa.feature.rms(y=y)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    # simplistic mapping
    if rms < 0.01 and zcr < 0.05:
        em = "sad"
    elif rms > 0.05 and tempo > 100:
        em = "happy"
    elif rms > 0.05 and tempo < 70:
        em = "angry"
    else:
        em = "neutral"
    return {"emotion": em, "rms": rms, "zcr": zcr, "tempo": float(tempo)}

# high-level analyze_audio_emotion (keeps compatibility)
def analyze_audio_emotion(wav_path):
    return map_audio_features_to_emotion(wav_path)

# speech-to-text (optional): tries to use SpeechRecognition/pocketsphinx/google
def transcribe_audio(wav_path):
    try:
        import speech_recognition as sr
    except Exception:
        return None
    r = sr.Recognizer()
    with sr.AudioFile(wav_path) as source:
        audio = r.record(source)
    try:
        # try Google (requires internet), else fallback to Sphinx if installed
        text = r.recognize_google(audio)
        return text
    except Exception:
        try:
            text = r.recognize_sphinx(audio)
            return text
        except Exception:
            return None
def extract_prosody(wav_path):
    import librosa, numpy as np
    y, sr = librosa.load(wav_path, sr=16000)
    # RMS energy
    energy = float(np.mean(librosa.feature.rms(y=y)))
    # Pitch estimation using pyin
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=50, fmax=500)
    f0 = f0[~np.isnan(f0)]
    f0_std = float(np.std(f0)) if len(f0) > 0 else 0.0
    return {"energy": energy, "f0_std": f0_std}



# # src/audio_emotion.py
# import numpy as np, librosa, os

# def extract_prosody(wav_path, sr=16000):
#     # load
#     y, sr = librosa.load(wav_path, sr=sr)
#     # energy
#     energy = np.mean(np.abs(y))
#     # zero crossing rate
#     zcr = np.mean(librosa.feature.zero_crossing_rate(y))
#     # pitch (f0) using YIN (robust)
#     try:
#         f0 = librosa.yin(y, fmin=50, fmax=500, sr=sr)
#         f0 = f0[~np.isnan(f0)]
#         f0_mean = float(np.mean(f0)) if len(f0)>0 else 0.0
#         f0_std = float(np.std(f0)) if len(f0)>0 else 0.0
#     except Exception:
#         f0_mean, f0_std = 0.0, 0.0
#     return {"energy": energy, "zcr": zcr, "f0_mean": f0_mean, "f0_std": f0_std}

# def map_audio_features_to_emotion(wav_path):
#     feats = extract_prosody(wav_path)
#     # simple heuristics:
#     # high energy + high pitch variability -> anger/excited
#     # low energy + low pitch -> sad/low
#     # stable mid-range pitch + medium energy -> neutral/happy
#     e = feats["energy"]
#     f0 = feats["f0_mean"]
#     f0s = feats["f0_std"]
#     z = feats["zcr"]
#     emotion = "neutral"
#     if e > 0.06 and f0s > 40:
#         emotion = "angry"
#     elif e < 0.01 and f0 < 120:
#         emotion = "sad"
#     elif 0.01 <= e <= 0.06 and f0s < 30 and f0>120:
#         emotion = "happy"
#     elif f0s > 60 and e>0.04:
#         emotion = "fear"
#     else:
#         emotion = "neutral"
#     out = {"emotion": emotion, "features": feats}
#     return out

# # convenience wrapper for earlier endpoint
# def analyze_audio_emotion(wav_path):
#     if not os.path.exists(wav_path):
#         return {"error": "file not found"}
#     return map_audio_features_to_emotion(wav_path)
