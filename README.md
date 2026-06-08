# llama-server Command Generator

A Python GUI application for generating [llama.cpp](https://github.com/ggerganov/llama.cpp) server commands. You must have llama.cpp already built on your device, this project does not come with llama.cpp!

## Overview

Point-and-click interface to configure and generate `llama-server` command-line arguments without typing them manually. Select your GGUF model, adjust settings, copy the generated command — done.

## Features

### System Hardware

- **Auto-Detected Specs**: Displays CPU model (with core/thread count), GPU model, VRAM, and total system RAM on launch
- **Hardware Scanner**: Uses WMI (Windows) to query CPU cores/threads, GPU details, and memory without third-party libraries

### Model Loading

- **GGUF Browse**: Open file dialog to select `.gguf` model files
- **HuggingFace Link**: Opens huggingface.co/models pre-filtered by GGUF format

### Performance Tuning

- **GPU Layers**: Spinbox (-1=auto-detect, 0–99 explicit layers offloaded via `-ngl`)
- **Context Size**: Numeric input (2–999999999, passed as `-c`)
- **Flash Attention**: Enable with `-fa on` flag
- **Fit On**: GPU VRAM fit mode via `--fit on`
- **Batch Size**: Spinbox 1–8192 (`-b`)
- **Micro-Batch Size**: Memory splitting via `-ub` (1–8192)
- **Threads**: Override CPU thread count (-1=auto, 1–128 via `-t`)
- **Thread Batch**: Per-thread batch size (0=unset, 1–512 via `-tb`)
- **Cache Type K/V**: Dropdown selectors — f16, f32, q8_0, q4_0, q4_1, iq4_nl

### Speculative Decoding

- **Toggle Enable/Disable**: Master switch for speculative decoding
- **Strategy Selection**: `ngram-mod` and/or `draft-mtp` (multi-select)
- **Draft Parameters**: `--spec-draft-n-max` (0–64) and `--spec-draft-n-min` (0–64) spinboxes

### Network & Server

- **Host Binding**: Configurable bind address (default `0.0.0.0` — all interfaces/LAN access)
- **Port Selection**: Spinbox 1–65535 (default `8080`)

### Sampling Parameters

- **Temperature**: 0.05 – 2.0 (step 0.05)
- **Min-P**: -1.0 – 1.0 (step 0.01)
- **Top-K**: 1 – 9999 (step 1)
- **Presence Penalty**: -2.0 – 2.0 (step 0.1)
- **Top-P**: 0.05 – 1.0 (step 0.05)
- **Repeat Penalty**: 1.0 – 3.0 (step 0.1)

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
- **Window Geometry**: Window size, position, and maximized state preserved between sessions
- **Last Folder**: `.bat` save folder remembered across sessions

## Requirements

- Python 3.7+ (uses `tkinter` — included with standard Python installations)
- Windows (hardware scanner uses WMI / PowerShell)
- No third-party dependencies required

## Usage

Download and run the `.exe` file, app window will open shortly.

Configure settings then copy the generated command from the bottom panel, run it in a separate CMD window straight from the app, or save it as a `.bat` for later one-click loading.

## File Structure

```
llama-launcher/
├── main.py              # Entry point — creates Tk root and GUI instance
├── gui_builder.py       # Full GUI logic, sections, command generation, optimiser
├── hardwarescanner.py   # WMI-based hardware detection (CPU/GPU/VRAM/RAM)
├── optimiser_script.py  # Benchmark runner and result parser
└── llama_gui_data.json  # Auto-generated config (saved settings)
```
