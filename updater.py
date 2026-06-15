import os
import sys
import re
import threading
import tkinter as tk
from tkinter import messagebox
import requests

def get_current_version():
    """Extracts the version number directly from the running .exe filename"""
    running_filename = os.path.basename(sys.argv[0])
    match = re.search(r"v?(\d+\.\d+\.\d+)", running_filename)
    if match:
        return match.group(1)
    # Fallback version for when you run the raw .py file in your IDE
    return "1.3.3"

CURRENT_VERSION = get_current_version()

GITHUB_OWNER = "SolaryKryptic"
GITHUB_REPO = "llama-launcher"


def check_for_updates(root_window):
    """Runs the version check in a background thread to keep Tkinter UI responsive"""
    thread = threading.Thread(target=updater_worker, args=(root_window,), daemon=True)
    thread.start()


def updater_worker(root_window):
    api_url = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/releases/latest"
    headers = {"User-Agent": "TkinterAppUpdater"}

    try:
        response = requests.get(api_url, headers=headers, timeout=10)
        if response.status_code != 200:
            return

        data = response.json()
        latest_version = data.get("tag_name", "").strip("v")

        if latest_version > CURRENT_VERSION:
            root_window.after(0, lambda: prompt_update(root_window, data, latest_version))

    except requests.RequestException:
        pass


def prompt_update(root_window, release_data, latest_version):
    """Asks the user if they want to download the new version"""
    msg = f"A new version ({latest_version}) is available!\nWould you like to download it now?\n\nThe new exe will be saved to the same folder. You can delete the old one after."
    if not messagebox.askyesno("Update Available", msg):
        return

    download_url = None
    for asset in release_data.get("assets", []):
        asset_name = asset.get("name", "")
        if asset_name.startswith("llama-launcher-") and asset_name.endswith(".exe"):
            download_url = asset.get("browser_download_url")
            break

    if download_url:
        threading.Thread(
            target=download_new_exe,
            args=(download_url, latest_version, root_window),
            daemon=True
        ).start()
    else:
        messagebox.showerror("Error", "Could not find the updated .exe in the release assets. Please download it manually from GitHub.")


def download_new_exe(download_url, latest_version, root_window):
    """Downloads the new exe directly with the correct versioned filename"""
    try:
        if getattr(sys, 'frozen', False):
            exe_dir = os.path.dirname(os.path.abspath(sys.executable))
        else:
            exe_dir = os.path.dirname(os.path.abspath(sys.argv[0]))

        new_exe_name = f"llama-launcher-v{latest_version}.exe"
        new_exe_path = os.path.join(exe_dir, new_exe_name)

        response = requests.get(download_url, stream=True, timeout=60)
        with open(new_exe_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        root_window.after(0, lambda: messagebox.showinfo(
            "Download Complete",
            f"New version saved as:\n{new_exe_name}\n\nYou can now close this app and delete the old exe."
        ))

    except Exception as e:
        err_msg = str(e)
        root_window.after(0, lambda: messagebox.showerror("Download Failed", f"Error: {err_msg}"))