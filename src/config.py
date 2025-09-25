import yaml
from pathlib import Path

_cfg = None

def load(cfg_path="config.yaml"):
    global _cfg
    if _cfg is None:
        cfg_file = Path(cfg_path)
        if not cfg_file.exists():
            raise FileNotFoundError(f"{cfg_path} not found")
        _cfg = yaml.safe_load(open(cfg_path))
    return _cfg
