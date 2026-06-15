# llama-launcher

A Python GUI application for generating [llama.cpp](https://github.com/ggerganov/llama.cpp) server commands. You must have llama.cpp already built on your device, this project does not come with llama.cpp!

## Overview

Point-and-click interface to configure and generate `llama-server` command-line arguments without typing them manually. Select your GGUF model, adjust settings, run the command.

<img width="765" height="894" alt="image" src="https://github.com/user-attachments/assets/84888536-f92f-488e-bc87-10e53cb829d8" />


## - Qwen 4B MTP results after optimisation
<img width="578" height="589" alt="image" src="https://github.com/user-attachments/assets/f5597e44-4fa5-4963-9c6b-2ab60c1555f0" />

## - Gemma E4B MTP results after optimisation
<img width="604" height="551" alt="image" src="https://github.com/user-attachments/assets/a55af24c-3cfd-4b07-be91-acb5935d12e1" />

## - Qwen 3.5 0.8b MTP CPU ONLY results after optimisation
<img width="578" height="589" alt="image" src="https://github.com/user-attachments/assets/b1ec95c8-c9ce-4866-820f-faac7cdf861f" />


## Features

### System Hardware

- **Hardware Detection**: Displays CPU model (with core/thread count), GPU model, VRAM, and total system RAM on launch
- **Hardware Scanner**: Uses WMI (Windows) to query CPU cores/threads, GPU details, and memory without third-party libraries

### Model Loading

- **GGUF Browse**: Open file dialog to select `.gguf` model files
- **HuggingFace Link**: Opens huggingface.co/models pre-filtered by GGUF format

### Auto-Optimiser

- **Bayesian-Only Optimisation**: Sequential optimisation has been removed; Bayesian/Optuna TPE is the only optimisation method.
- **Optimiser Settings Persistence**: Score weighting, context size, trial count, average runs, seed, PPL threshold, and corpus path are saved and restored between optimiser settings windows.
- **Search Space**: Uses Optuna TPE across threads, effective thread batch, batch size, micro-batch, FITT, KV cache types, draft-MTP cache types, and speculative draft settings.
- **PPL Validation**: Changed-cache trials are validated against the original baseline PPL using `llama-perplexity`. Same-cache trials skip PPL because their KV quantisation matches the baseline.
- **PPL Command Flags**: Perplexity runs include `-fit on`, `-fitc`, `-c`, `-fitt 50`, `-s`, `--chunks 3`, `-ctk`, and `-ctv`; draft cache flags are not passed to PPL because no draft model is supplied.
- **Benchmark Command Flags**: Server benchmark runs still include full cache flags, including `-ctkd` and `-ctvd`, for speculative/MTP speed testing.
- **Failure Handling**: Benchmark failures and PPL failures return low finite penalty scores so Optuna can learn from bad regions; user cancellation/interrupt still prunes.
- **Baseline Fallback**: If no PPL-validated trial beats the baseline, the results dialog applies the baseline command instead of returning no result.
- **Live Progress Window**: Shows progress, current step, ETA, baseline score, last score, and best PPL-validated score while benchmarking.
- **Results Dialog**: Shows the final method, context size, tuned fields, baseline score/speeds/PPL, best PPL-validated score/speeds/PPL, PPL threshold, improvement percentage, and whether the baseline command is being used.
- **Apply Settings Sync**: Apply Settings now updates the main GUI controls as well as the command, including context size, spinboxes, checkboxes, cache dropdowns, speculative sub-options, draft model label, and draft-MTP cache visibility.
- **Copy Flags**: Copies the final recommended `llama-server` flags, including MTP/speculative flags when enabled.

### Output Actions

- **Live Command Preview**: Updates in real-time as you change any setting
- **Copy**: Copies the full command to clipboard
- **Run in CMD**: Launches `cmd.exe /k` with the generated command in a new window
- **Save as .bat**: Dialog to pick folder and filename, saves command as executable `.bat` file

### Persistence

- **Auto-Save/Restore**: Flag values and optimiser settings are saved to `llama_gui_data.json` on close/start/cancel and restored on launch.
- **Saved Setting Normalisation**: Old or incomplete saved settings with `null` values are normalised to safe defaults so startup and command generation do not crash.
- **Corpus Path Handling**: `Moby Dick.txt` is bundled as the default PPL corpus; relative corpus paths resolve from the `.exe` folder, while externally selected corpus paths remain absolute.

## Requirements

- Python 3.10+ recommended
- Windows (hardware scanner uses WMI / PowerShell)
- `llama.cpp` built with `llama-server.exe` and `llama-perplexity.exe` available

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
├── Moby Dick.txt        # Bundled default perplexity corpus
└── updater.py

```
