import os
import sys
import re  # FIX 1: Added missing import for regular expressions
import threading
import subprocess
import tkinter as tk
from tkinter import messagebox
import requests

def get_current_version():
    """Extracts the version number directly from the running .exe filename"""
    running_filename = os.path.basename(sys.argv[0])
    
    # Looks for numbers separated by dots (e.g., 1.0.3)
    match = re.search(r"v?(\d+\.\d+\.\d+)", running_filename)
    if match:
        return match.group(1)
    return "1.0.5"
    # Fallback version for when you run the raw .py file in your IDE

# Automatically sets itself based on the file name
CURRENT_VERSION = get_current_version() 

GITHUB_OWNER = "SolaryKryptic"
GITHUB_REPO = "llama-launcher"


def check_for_updates(root_window):
    """Runs the version check in a background thread to keep Tkinter UI responsive"""
    thread = threading.Thread(target=updater_worker, args=(root_window,), daemon=True)
    thread.start()


def updater_worker(root_window):
    # FIX 2: Corrected the broken API endpoint syntax
    api_url = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/releases/latest"
    headers = {"User-Agent": "TkinterAppUpdater"}
    
    try:
        response = requests.get(api_url, headers=headers, timeout=10)
        if response.status_code != 200:
            return # Silent fail if API limit reached or repo is private

        data = response.json()
        latest_version = data.get("tag_name", "").strip("v")
        
        # Simple semantic version string comparison
        if latest_version > CURRENT_VERSION:
            # Safely schedule the GUI popup on Tkinter's main thread
            root_window.after(0, lambda: prompt_update(root_window, data, latest_version))
            
    except requests.RequestException:
        pass # Handle offline or connection drops gracefully


def prompt_update(root_window, release_data, latest_version):
    """Asks the user if they want to update and dynamically finds the asset"""
    msg = f"A new version ({latest_version}) is available!\nWould you like to update now?"
    if messagebox.askyesno("Update Available", msg):
        
        download_url = None
        # Loop through assets and find the one that matches your application format
        for asset in release_data.get("assets", []):
            asset_name = asset.get("name", "")
            if asset_name.startswith("llama-launcher-") and asset_name.endswith(".exe"):
                download_url = asset.get("browser_download_url")
                break
        
        if download_url:
            threading.Thread(target=download_and_install, args=(download_url, root_window), daemon=True).start()
        else:
            messagebox.showerror("Error", "Could not find the updated .exe file in assets. Please check the release page manually.")


def download_and_install(download_url, root_window):
    """Downloads the new binary and triggers the swap sequence"""
    try:
        # UPDATED: Targets the true path of the running executable file on disk
        if getattr(sys, 'frozen', False):
            # Running as PyInstaller exe
            current_exe_path = os.path.abspath(sys.executable)
        else:
            # Running as .py script during development
            current_exe_path = os.path.abspath(sys.argv[0])
        temp_exe_path = current_exe_path + ".new"

        # Download the file to a temporary location
        response = requests.get(download_url, stream=True, timeout=30)
        with open(temp_exe_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        # Trigger the file swap script execution
        root_window.after(0, lambda: apply_update_and_restart(current_exe_path, temp_exe_path, root_window))
        
    except Exception as e:
        err_msg = str(e)
        root_window.after(0, lambda: messagebox.showerror("Update Failed", f"Error: {err_msg}"))



def apply_update_and_restart(current_path, temp_path, root_window):
    """Executes a detached script to swap the running binary, then exits"""
    if sys.platform == "win32":
        # Windows Batch command line sequence:
        # Wait 2 seconds for app to exit -> delete old app -> rename new app -> launch updated app
        batch_cmd = (
            f'timeout /t 2 > nul && '
            f'del "{current_path}" && '
            f'move "{temp_path}" "{current_path}"'
        )
        subprocess.Popen(batch_cmd, shell=True, creationflags=subprocess.CREATE_NEW_CONSOLE)
    else:
        # macOS / Linux Bash command line sequence:
        bash_cmd = (
            f'sleep 2 && '
            f'mv "{temp_path}" "{current_path}" && '
            f'chmod +x "{current_path}" && '
            f'"{current_path}" &'
        )
        subprocess.Popen(bash_cmd, shell=True, start_new_session=True)

    # Immediately close the current instance so the updater script can write over the file
    messagebox.showinfo("Update Complete", "Update downloaded. Please relaunch the app.")
    root_window.destroy()
    sys.exit()

