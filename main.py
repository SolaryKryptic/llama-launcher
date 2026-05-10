import tkinter as tk
sys = __import__('sys')  # import sys for platform detection after path manipulation
sys.path.insert(0, '.')

from gui_builder import LlamaServerGUI

root = tk.Tk()
gui = LlamaServerGUI(root)

# Disable mouse scroll wheel — prevents accidental value changes.
def _no_scroll(event):
    return "break"
root.bind("<MouseWheel>", _no_scroll)  # Windows/Linux
root.bind("<Button-4>", _no_scroll)   # macOS trackpad
root.bind("<Button-5>", _no_scroll)   # macOS trackpad

root.title("llama-server CLI Generator")
# Grid layout with sticky="nswe" makes all widgets fill the window.
if sys.platform == "darwin":  # macOS: native WM supports -zoomed natively
    root.attributes('-zoomed', True)
elif not sys.platform.startswith("win"):  # Linux / X11 via xdotool if available
    import subprocess, time as _time_mod
    try:
        root.update_idletasks()
        w = max(root.winfo_screenwidth(), root.winfo_reqwidth())
        h = max(root.winfo_screenheight(), root.winfo_reqheight())
        subprocess.run(["xdotool", "windowsize", str(root.winfo_id()), str(w), str(h)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (FileNotFoundError, Exception):
        pass  # xdotool not available — normal window is fine.

root.mainloop()
