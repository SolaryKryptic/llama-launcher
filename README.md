# llama-server CLI Generator

A Python GUI application for generating [llama.cpp](https://github.com/ggerganov/llama.cpp) server commands.

## Overview

Point-and-click interface to configure and generate `llama-server` command-line arguments without typing them manually. Select your GGUF model, adjust settings, copy the generated command — done.

## Features

- **Model Selection**: Browse for `.gguf` files
- **Performance Settings**: GPU layers (-1=auto), context size toggle
- **Low VRAM / MLock**: Mutually exclusive memory options
- **Server Configuration**: Host binding and port selection
- **Thread Control**: Optional CPU thread count override
- **Sampling Parameters**: Temperature, top-k, top-p, repeat penalty, seed
- **Live Command Preview**: Generated command updates in real-time
- **One-Click Copy**: Copies the full command to clipboard

## Requirements

- Python 3.7+ (uses `tkinter` — included with standard Python installations)
- No third-party dependencies required

## Usage

```bash
python main.py
```

The application window opens centered on your screen. Configure settings then copy the generated command from the bottom panel.

### Example Output

```
llama-server \
    --model "C:\models\mistral.gguf" \
    --n-gpu-layers auto \
    --host 0.0.0.0 \
    --port 8080
```

## Screenshots

![App Interface](screenshots/interface.png)

*(Add screenshot when available)*
