import hashlib, os
from pathlib import Path
def sha256_file(path):
    h = hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            h.update(chunk)
    return h.hexdigest()

def save_json(obj, path):
    import json
    with open(path,'w') as f:
        json.dump(obj, f, indent=2)
