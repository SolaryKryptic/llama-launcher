# llama-server Command Generator

A Python GUI application for generating [llama.cpp](https://github.com/ggerganov/llama.cpp) server commands. You must have llama.cpp already built on your device, this project does not come with llama.cpp!

## Overview

Point-and-click interface to configure and generate `llama-server` command-line arguments without typing them manually. Select your GGUF model, adjust settings, copy the generated command — done.

## Features

### Model Loading

- **GGUF Browse**: Open file dialog to select `.gguf` models
- **No-MMAP / MLock**: Mutually exclusive memory loading options (only one active at a time)

### Performance Tuning

- **GPU Layers**: Spinbox (-1=auto-detect, 0–99 explicit), always included in generated command
- **Context Size**: Toggle + numeric input (2–999999999)
- **Flash Attention**: Enable with `-fa on` flag
- **Fit On**: GPU VRAM fit mode via `--fit-on`
- **Batch Size**: Spinbox 1–8192 (`-b`)
- **Micro-Batch Size**: Memory splitting via `-ub` (1–8192)
- **Threads**: Override CPU thread count (-1=auto, up to 128)
- **Cache Type K/V**: Dropdown selectors — f16, f32, q8_0, q4_0, q4_1, iq4_nl

### Network & Server

- **Host Binding**: Configurable bind address (default `0.0.0.0`)
- **Port Selection**: Spinbox 1–65535

### Sampling Parameters

- **Temperature**: 0.05 – 2.0
- **Min-P**: -1.0 – 1.0
- **Top-K**: 1 – 9999
- **Presence Penalty**: -2.0 – 2.0
- **Top-P**: 0.05 – 1.0
- **Repeat Penalty**: 1.0 – 3.0

### Output Actions

- **Live Command Preview**: Updates in real-time as you change any setting
- **Copy**: Copies the full command to clipboard
- **Run in CMD**: Launches `cmd.exe` with the generated command
- **Save as .bat**: Saves to a user-specified folder and filename

### Other

- **Config Persistence**: All settings saved/restored automatically between sessions (`~/.llama_server_gui.json`)
- **Cross-platform**: Works on Windows, Linux (xdotool), and macOS

## Requirements

- Python 3.7+ (uses `tkinter` — included with standard Python installations)
- No third-party dependencies required

## Usage

```bash
python main.py
```

The application window opens centered on your screen. Configure settings then copy the generated command from the bottom panel, run it in a seperate CMD window straight from the app, or save it as a .bat for one-click loading.

### Example Output

```
llama-server.exe -m "C:\models\mistral.gguf" -ngl auto --no-mmap -b 2048 -ub 512 -ctk f16 -ctv f16 --ctx-size 4096 --host 0.0.0.0 --port 8080 --temp 0.80 --min-p 0.00 --top-k 40 --presence-penalty 0.00 --top-p 0.950 --repeat-penalty 1.10
```

## Screenshots

![App Interface](screenshots/interface.png)

*(Add screenshot when available)*
