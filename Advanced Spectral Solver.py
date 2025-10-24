# -*- coding: utf-8 -*-
"""
ASS stater
Author: Martin Jindra
"""
import os, json
from pathlib import Path
from ASS.gui import MainWindow

DEFAULT_PLOT_CONFIG = {
    "X axis": None,
    "Y axis": None,
    "Raw label": None,
    "Compare offset": 0,
    "Compare label": None,
    "Title": None,
}

DEFAULT_USER_LOADER = {
    "file_type": "txt",
    "separator": "\\s+",
    "skip_rows": 40,
    "usecols": [0, 1],
    "encoding": "ansi",
}

def ensure_user_config():
    # 1. Where configs live
    config_dir = Path(os.getenv("APPDATA", Path.home())) / "ASS"
    config_dir.mkdir(parents=True, exist_ok=True)

    # 2. Define file paths
    plot_path = config_dir / "plot_config.json"
    loader_path = config_dir / "user_loader.json"

    # 3. If missing, create them
    if not plot_path.exists():
        with open(plot_path, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_PLOT_CONFIG, f, indent=4)
        print(f"🆕 Created default {plot_path}")

    if not loader_path.exists():
        with open(loader_path, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_USER_LOADER, f, indent=4)
        print(f"🆕 Created default {loader_path}")

    print(f"✅ Config folder ready: {config_dir}")
    
if __name__ == "__main__":
    ensure_user_config()
    app = MainWindow()
    app.run()