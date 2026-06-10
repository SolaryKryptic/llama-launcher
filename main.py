import tkinter as tk
sys = __import__('sys')  # import sys for platform detection after path manipulation
sys.path.insert(0, '.')
import updater
from gui_builder import LlamaServerGUI

root = tk.Tk()
gui = LlamaServerGUI(root)

# Disable mouse scroll wheel to prevent accidental value changes
def _no_scroll(event):
    return "break"
root.bind("<MouseWheel>", _no_scroll)

# UPDATED: Use the dynamic version from updater.py so it matches your GitHub file tags
root.title(f"llama-launcher - v{updater.CURRENT_VERSION}")

# Restore saved window geometry or default to centered
try:
    import json as _json, os as _os, sys as _sys
    if getattr(_sys, 'frozen', False):
        _base_dir = _os.path.dirname(_sys.executable)
    else:
        _base_dir = _os.path.dirname(_os.path.abspath(__file__))
    _cfg_path = _os.path.join(_base_dir, "llama_gui_data.json")
    with open(_cfg_path, "r", encoding="utf-8") as _f:
        _saved = _json.load(_f)
    geom = _saved.get("window_geometry", "764x693")
    root.geometry(geom)
    root.update_idletasks()
    state = _saved.get("window_state", "normal")
    if state == "zoomed":
        root.state("zoomed")
except Exception:
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    x_pos = max(0, (screen_w - 764) // 2)
    y_pos = max(0, (screen_h - 593) // 2)
    root.geometry(f"764x693+{x_pos}+{y_pos}")

# ADDED: Automatically launch the silent background update check 1 second after startup
root.after(1000, lambda: updater.check_for_updates(root))

root.mainloop()