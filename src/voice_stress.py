# src/voice_stress.py
from .audio_emotion import extract_prosody
def voice_stress_score(wav_path):
    feats = extract_prosody(wav_path)
    # heuristic: high energy + high pitch variance -> stress
    score = 0.0
    e = feats['energy']; pstd = feats['f0_std']
    score = min(1.0, (e*10) + (pstd/100.0))
    return {'stress_score': float(score), 'features': feats}
