# llama-launcher

A Python GUI application for generating [llama.cpp](https://github.com/ggerganov/llama.cpp) server commands. You must have llama.cpp already built on your device, this project does not come with llama.cpp!

## Overview

Point-and-click interface to configure and generate `llama-server` command-line arguments without typing them manually. Select your GGUF model, adjust settings, run the command.

<img width="765" height="894" alt="image" src="https://github.com/user-attachments/assets/84888536-f92f-488e-bc87-10e53cb829d8" />


Gemma 12B MTP results after optimisation
<img width="603" height="527" alt="image" src="https://github.com/user-attachments/assets/f83397b1-c1f8-4a3d-8c7e-1858216613e3" />


Qwen3.5 4B MTP results after optimisation
<img width="606" height="551" alt="image" src="https://github.com/user-attachments/assets/e1e1e759-33db-49d8-ae0b-d0c7afbcdb74" />



## Features

### System Hardware

- **Hardware Detection**: Displays CPU model (with core/thread count), GPU model, VRAM, and total system RAM on launch
- **Hardware Scanner**: Uses WMI (Windows) to query CPU cores/threads, GPU details, and memory without third-party libraries

### Model Loading

- **GGUF Browse**: Open file dialog to select `.gguf` model files
- **HuggingFace Link**: Opens huggingface.co/models pre-filtered by GGUF format

### Auto-Optimiser (WIP)

- **Sequential Greedy Search**: Sweeps threads → batch size → FITT target → cache type K independently
- **Neighbourhood Verification**: Tests adjacent combinations of best batch × FITT to confirm global optimum
- **Live Progress Window**: Progress bar, current step label, real-time score, and ETA
- **Results Dialog**: Shows best config with "Apply Settings" and "Copy Flags" buttons

### Output Actions

- **Live Command Preview**: Updates in real-time as you change any setting
- **Copy**: Copies the full command to clipboard
- **Run in CMD**: Launches `cmd.exe /k` with the generated command in a new window
- **Save as .bat**: Dialog to pick folder and filename, saves command as executable `.bat` file

### Persistence

- **Auto-Save/Restore**: All flag values saved to `llama_gui_data.json` on close, restored on launch

## Requirements

- Python 3.7+
- Windows (hardware scanner uses WMI / PowerShell)
- No third-party dependencies required

## Usage

Download and run the `.exe` file (click "More info" then "Run anyway" if Windows SmartScreen pops up), app window will open shortly.

Configure settings then copy the generated command from the bottom panel, run it in a separate CMD window straight from the app, or save it as a `.bat` for later one-click loading.

## File Structure

```
llama-launcher/
├── main.py              # Entry point — creates Tk root and GUI instance
├── gui_builder.py       # Full GUI logic, sections, command generation, optimiser
├── hardwarescanner.py   # WMI-based hardware detection (CPU/GPU/VRAM/RAM)
├── optimiser_script.py  # Benchmark runner and result parser
├── llama_gui_data.json  # Auto-generated config (saved settings)
└── updater.py

```
