# llama-launcher

A Python GUI application for generating [llama.cpp](https://github.com/ggerganov/llama.cpp) server commands. You must have llama.cpp already built on your device, this project does not come with llama.cpp!

## Overview

Point-and-click interface to configure and generate `llama-server` command-line arguments without typing them manually. Select your GGUF model, adjust settings, run the command.

<img width="765" height="894" alt="image" src="https://github.com/user-attachments/assets/84888536-f92f-488e-bc87-10e53cb829d8" />


- Gemma 12B MTP results after optimisation
<img width="579" height="552" alt="image" src="https://github.com/user-attachments/assets/e10f067d-9ba7-4353-8a33-22d14b5c5759" />


- Gemma E4B MTP results after optimisation
<img width="604" height="551" alt="image" src="https://github.com/user-attachments/assets/a55af24c-3cfd-4b07-be91-acb5935d12e1" />


- Gemma E4B no MTP results after optimisation (optimisation seems to have the best gains with MTP)
<img width="602" height="549" alt="image" src="https://github.com/user-attachments/assets/c11fd7be-3778-44f4-8613-ae76a34ae051" />


## Features

### System Hardware

- **Hardware Detection**: Displays CPU model (with core/thread count), GPU model, VRAM, and total system RAM on launch
- **Hardware Scanner**: Uses WMI (Windows) to query CPU cores/threads, GPU details, and memory without third-party libraries

### Model Loading

- **GGUF Browse**: Open file dialog to select `.gguf` model files
- **HuggingFace Link**: Opens huggingface.co/models pre-filtered by GGUF format

### Auto-Optimiser

- **Optimiser Settings**: Choose score weighting, context size, Bayesian trial count, average runs per trial, and seed
- **Bayesian Optimisation**: Uses Optuna TPE search across threads, thread batch, batch size, micro-batch, FITT, KV cache types, draft-MTP cache types, and speculative draft settings
- **MTP / Draft-MTP Support**: Can optimise MTP without a separate draft model, or with a selected draft model; `--model-draft` is only added when a draft model path is selected
- **Speculative Draft Controls**: Configure draft-MTP draft model, draft token max/min, and `--spec-draft-p-min`
- **Live Progress Window**: Shows progress, current step, ETA, baseline score, last score, and best PPL-validated score while benchmarking
- **Results Dialog**: Shows the final method, context size, tuned flags, baseline score/speeds, Trial 0 baseline values, best PPL-validated score/speeds, and improvement percentage
- **Apply Settings**: Applies tuned `-t`, `-tb`, `-b`, `-ub`, `-fitt`, `-ctk`, `-ctv`, `-ctkd`, `-ctvd`, flash attention, fit-on, and draft-MTP settings back into the main command generator
- **Copy Flags**: Copies the final recommended `llama-server` flags, including MTP/speculative flags when enabled

### Output Actions

- **Live Command Preview**: Updates in real-time as you change any setting
- **Copy**: Copies the full command to clipboard
- **Run in CMD**: Launches `cmd.exe /k` with the generated command in a new window
- **Save as .bat**: Dialog to pick folder and filename, saves command as executable `.bat` file

### Persistence

- **Auto-Save/Restore**: All flag values saved to `llama_gui_data.json` on close, restored on launch

## Requirements

- Python 3.10+ recommended
- Windows (hardware scanner uses WMI / PowerShell)
- `llama.cpp` built with `llama-server.exe` available
- `requests`
- `optuna` required for optimisation

## Usage

Download and run the `.exe` file (click "More info" then "Run anyway" if Windows SmartScreen pops up), app window will open shortly.

Configure settings then copy the generated command from the bottom panel, run it in a separate CMD window straight from the app, or save it as a `.bat` for later one-click loading.

## File Structure

```
llama-launcher/
├── main.py              # Entry point — creates Tk root and GUI instance
├── gui_builder.py       # Tk GUI, command generation, settings persistence, optimisation UI
├── hardwarescanner.py   # WMI-based hardware detection (CPU/GPU/VRAM/RAM)
├── optimisation_service.py  # Dispatches optimisation requests
├── optimiser_script.py  # Shared benchmark, PPL, cache, and server utilities
├── bayesian.py          # Optuna Bayesian optimisation harness
├── llama_gui_data.json  # Auto-generated config (saved settings)
└── updater.py

```
