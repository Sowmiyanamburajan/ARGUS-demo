# import numpy as np, os
# from PIL import Image
# import imagehash
# import matplotlib.pyplot as plt
# import librosa
# from pathlib import Path

# def compute_phash(img_path):
#     im = Image.open(img_path).convert('RGB')
#     ph = imagehash.phash(im)
#     return str(ph)

# def phash_match(ph, ph_db, max_dist=6):
#     # ph_db: dict {filename: phash}
#     from imagehash import hex_to_hash
#     h = imagehash.hex_to_hash(ph)
#     matches=[]
#     for fn,ph2 in ph_db.items():
#         h2 = imagehash.hex_to_hash(ph2)
#         if h - h2 <= max_dist:
#             matches.append(fn)
#     return matches

# def save_plot_rppg(greens, filtered, out_path):
#     plt.figure(figsize=(6,2.5))
#     plt.plot(greens, label='green mean')
#     plt.plot(filtered, label='filtered')
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(out_path)
#     plt.close()

# def save_spectrogram(wav_path, out_img):
#     y, sr = librosa.load(wav_path, sr=16000)
#     S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
#     S_db = librosa.power_to_db(S, ref=np.max)
#     plt.figure(figsize=(6,3))
#     plt.imshow(S_db, aspect='auto', origin='lower')
#     plt.axis('off')
#     plt.tight_layout()
#     plt.savefig(out_img, bbox_inches='tight', pad_inches=0)
#     plt.close()
import matplotlib.pyplot as plt
from PIL import Image
import imagehash

def compute_phash(img_path):
    return imagehash.phash(Image.open(img_path))

def save_plot(arr, out_path, title="Graph"):
    plt.figure(figsize=(5,3))
    plt.plot(arr)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
