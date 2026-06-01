# src/registry.py
import sqlite3, os
DB = "outputs/registry.db"
os.makedirs("outputs", exist_ok=True)
def append_registry_entry(path, metadata=None):
    conn = sqlite3.connect(DB, check_same_thread=False)
    conn.execute("""CREATE TABLE IF NOT EXISTS uploads (id INTEGER PRIMARY KEY AUTOINCREMENT, path TEXT, metadata TEXT, ts TEXT)""")
    conn.execute("INSERT INTO uploads (path, metadata, ts) VALUES (?,?,datetime('now'))", (path, str(metadata)))
    conn.commit(); conn.close()

# import json, os
# from pathlib import Path
# REG = Path("outputs/registry.json")
# def append_registry_entry(filepath, metadata=None):
#     entry = {'file': filepath, 'sha256': None, 'metadata': metadata or {}, 'timestamp': None}
#     try:
#         from .common import sha256_file
#         entry['sha256'] = sha256_file(filepath)
#     except Exception:
#         entry['sha256'] = None
#     import datetime
#     entry['timestamp'] = datetime.datetime.utcnow().isoformat()+'Z'
#     if not REG.exists():
#         REG.parent.mkdir(parents=True, exist_ok=True)
#         json.dump([entry], open(REG,'w'), indent=2)
#     else:
#         arr = json.load(open(REG))
#         arr.append(entry)
#         json.dump(arr, open(REG,'w'), indent=2)
#     return entry
