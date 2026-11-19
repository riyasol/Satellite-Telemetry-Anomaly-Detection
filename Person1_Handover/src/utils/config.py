from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]

def load_config():
    path = ROOT / "config" / "config.yaml"
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def resolve_path(value):
    p = Path(value)
    return p if p.is_absolute() else ROOT / p

def ensure_dirs(cfg):
    for key in ("processed_dir", "model_dir", "results_dir", "plot_dir"):
        resolve_path(cfg["output"][key]).mkdir(parents=True, exist_ok=True)
