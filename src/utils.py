from pathlib import Path
import json
import os

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def save_json(obj, path):
    ensure_dir(Path(path).parent)
    with open(path, 'w') as f:
        json.dump(obj, f, default=str, indent=2)
