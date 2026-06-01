# simple helper: compute phash of all files in dataset/known_fakes and save DB
import os, json
from src.utils import compute_phash
DB_OUT = 'outputs/phash_db.json'
def build_phash_db(folder='dataset/known_fakes'):
    db = {}
    if not os.path.exists(folder):
        print("Create dataset/known_fakes and add images of known fakes to build DB.")
        return
    for fn in os.listdir(folder):
        p = os.path.join(folder, fn)
        try:
            ph = compute_phash(p)
            db[fn] = ph
            print("Added", fn, ph)
        except Exception as e:
            print("skip", fn, e)
    with open(DB_OUT,'w') as f:
        json.dump(db, f, indent=2)
    print("Saved phash DB to", DB_OUT)

if __name__ == '__main__':
    build_phash_db()
