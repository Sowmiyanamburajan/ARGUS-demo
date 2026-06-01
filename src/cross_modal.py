# src/cross_modal.py
# Cross-modal consistency checks. Lightweight heuristics + hooks for model-based checks.
import os, numpy as np, json
from pathlib import Path
from .audio_emotion import extract_prosody
import cv2
from scipy.signal import correlate

def mouth_motion_series(frames_dir, sample_rate=4):
    """Estimate mouth openness per frame using a simple heuristic:
       compute average brightness around mouth region using a Haar cascade (fallback to center lower face).
       Returns time-series of values normalized [0,1].
    """
    files = sorted([os.path.join(frames_dir,f) for f in os.listdir(frames_dir) if f.endswith('.jpg')])
    vals = []
    # low-dependency approach: use center-lower-face crop
    for p in files:
        im = cv2.imread(p)
        if im is None:
            vals.append(0.0); continue
        h,w = im.shape[:2]
        y1 = int(h*0.55); y2 = int(h*0.85)
        x1 = int(w*0.25); x2 = int(w*0.75)
        crop = im[y1:y2, x1:x2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        # mouth openness heuristic: mean intensity (rough)
        vals.append(float(gray.mean()))
    if len(vals)==0: return np.array([])
    arr = np.array(vals)
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    return arr

def audio_energy_series(wav_path, hop_seconds=0.25):
    import librosa
    y, sr = librosa.load(wav_path, sr=16000)
    hop = int(sr * hop_seconds)
    energies = []
    for i in range(0, len(y), hop):
        seg = y[i:i+hop]
        energies.append(float((seg**2).mean()))
    arr = np.array(energies)
    if arr.sum()==0: return arr
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    return arr

def lip_sync_score(frames_dir, wav_path):
    """Compute a correlation between mouth motion series and audio energy series.
       Returns score in [0,1] where low score = possible lip-sync mismatch.
    """
    mouth = mouth_motion_series(frames_dir)
    aud = audio_energy_series(wav_path)
    if len(mouth)==0 or len(aud)==0:
        return {'score': 0.5, 'reason': 'insufficient_data'}
    # align lengths (resample shorter to longer)
    import numpy as np
    if len(mouth) > len(aud):
        aud = np.interp(np.linspace(0,len(aud)-1,len(mouth)), np.arange(len(aud)), aud)
    elif len(aud) > len(mouth):
        mouth = np.interp(np.linspace(0,len(mouth)-1,len(aud)), np.arange(len(mouth)), mouth)
    # cross-correlation as similarity
    c = correlate(mouth - mouth.mean(), aud - aud.mean(), mode='valid')
    peak = c.max() if c.size>0 else 0.0
    # normalize peak by possible energy
    denom = (np.sqrt(np.sum((mouth-mouth.mean())**2))*np.sqrt(np.sum((aud-aud.mean())**2))+1e-8)
    corr = float(peak/denom) if denom>0 else 0.0
    # map correlation [-1,1] to score [0,1]
    score = max(0.0, min(1.0, (corr+1)/2))
    return {'score': score, 'corr': corr}

def cross_modal_consensus(results):
    """
    results: list of per-file dicts (each with 'type' and 'score' keys)
    Returns weighted consensus and issues list.
    """
    if not results: return {'consensus': None, 'issues': []}
    # simple weighting: video>image>audio
    weight = {'video': 0.5, 'image': 0.3, 'audio': 0.2}
    total_w = 0.0; s = 0.0
    issues=[]
    for r in results:
        t = r.get('type','unknown')
        w = weight.get(t, 0.1)
        sc = float(r.get('score',0.5))
        s += w*sc; total_w += w
        # issue heuristics
        if t=='video' and r.get('lip_sync') and r['lip_sync'].get('score',1.0) < 0.4:
            issues.append('lip-sync-mismatch')
        if t=='audio' and r.get('stress_score',0) > 0.6:
            issues.append('voice-stress-suspicious')
    consensus = s/total_w if total_w>0 else None
    return {'consensus': consensus, 'issues': list(set(issues))}
