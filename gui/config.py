"""
Persistent configuration for VGLGui.
Reads / writes gui/config.json next to this file.
"""

import json
import os

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")

_DEFAULTS: dict = {
    "language": "pt",
}


def load_config() -> dict:
    """Returns the config dict, merged with defaults."""
    if os.path.isfile(_CONFIG_PATH):
        try:
            with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {**_DEFAULTS, **data}
        except Exception:
            pass
    return dict(_DEFAULTS)


def save_config(cfg: dict):
    """Persists the config dict to config.json."""
    try:
        with open(_CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[config] Could not save config: {e}")


_MAX_RECENT = 10

def get_recent_files() -> list[str]:
    """Returns list of recent file paths (most recent first)."""
    cfg = load_config()
    return cfg.get("recent_files", [])

def add_recent_file(path: str):
    """Adds path to recent files list and saves config."""
    cfg = load_config()
    recent = cfg.get("recent_files", [])
    # Remove if already present, then prepend
    recent = [p for p in recent if p != path]
    recent.insert(0, path)
    cfg["recent_files"] = recent[:_MAX_RECENT]
    save_config(cfg)
