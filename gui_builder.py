"""Core GUI builder for llama-server CLI command generation.

Pure stdlib tkinter — no third-party dependencies required.
Uses ttk widgets with the 'clam' theme (available on Windows 10/11).
"""

import os
import json
import subprocess
from tkinter import filedialog, messagebox, Tk, Toplevel, StringVar
import tkinter as tk
from tkinter import ttk

_CONFIG_PATH = os.path.join(os.path.expanduser("~"), ".llama_server_gui.json")

def _load_config():
    """Load the full config (last_folder + all flag values) from disk."""
    try:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _save_config(data):
    """Persist the full config to disk."""
    try:
        with open(_CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass

def _load_last_folder():
    """Return the previously saved destination folder, or empty string."""
    return _load_config().get("last_folder", "")

def _save_last_folder(folder):
    """Persist the given folder path for future sessions."""
    data = _load_config()
    data["last_folder"] = folder
    _save_config(data)


# ---------------------------------------------------------------------------
# Data model & command generation
# ---------------------------------------------------------------------------

class FlagConfig:
    """Holds all mutable flag state and builds the generated CLI string."""

    def __init__(self):
        self.model_path = ""          # set via browse dialog
        self.no_mmap = False          # disables memory-mapped file loading
        self.mlock = False            # mutually exclusive with no-mmap
        self.ctx_size_enabled = False  # toggle to include --ctx-size in command
        self.ctx_size_value = 512       # stored ctx-size number when enabled
        self.n_gpu_layers = -1         # -1 means auto-detect by GPU driver; range 0-99

        self.host = "0.0.0.0"          # bind all interfaces (local LAN access)
        self.port = 8080               # HTTP server port
        self.num_threads = os.cpu_count() or 4   # dynamic default from machine CPU count
        self.threads_enabled = False    # toggle to include -t/--threads in command

        self.flash_attention = False    # enable Flash Attention (-fa)
        self.fit_on = False              # enable --fit-on (GPU VRAM fit)

        self.batch_size = 2048           # batch size for KV cache (-b), range 1–8192
        self.micro_batch_size = 512      # micro batch size for memory splitting (-ub), range 1–8192
        self.threads = -1                # thread count (-t), -1 means auto-detect
        self.thread_batch = 0              # thread-batch (-tb), 0 means unset

        self.spec_enabled = False           # toggle to enable speculative decoding
        self.spec_type = "ngram-mod"         # spec strategy: ngram-mod or draft-mtp

        self.cache_type_k = "f16"       # KV cache type K, options: f16, f32, q8_0, q4_0, q4_1, iq4_nl
        self.cache_type_v = "f16"       # KV cache type V, same options

        self.temperature = 0.8          # sampling temperature
        self.min_p = 0.0                # minimum-p sampling value (0.0–1.0)
        self.top_k = 40                 # top-k sampling value
        self.presence_penalty = 0.0     # presence penalty factor (-2.0–2.0)
        self.top_p = 0.95               # nucleus sampling probability threshold
        self.repeat_penalty = 1.1       # repeat penalty factor

    def generate_command(self):
        """Build the full llama-server CLI command string from current state."""
        parts = ["llama-server.exe"]

        model_path = self.model_path.strip()
        if model_path:
            parts.append(f' -m "{model_path}"')

        # Mutually exclusive No-MMAP / MLock — only one can be active at a time
        if self.no_mmap and not self.mlock:
            parts.append(" --no-mmap")
        elif self.mlock and not self.no_mmap:
            parts.append(" --mlock")

        # GPU layers (always included in the generated command)
        gpu_str = "auto" if self.n_gpu_layers == -1 else str(self.n_gpu_layers)
        parts.append(f" -ngl {gpu_str}")

        # Flash Attention & Fit On (only when checked)
        if self.flash_attention:
            parts.append(" -fa on")
        if self.fit_on:
            parts.append(" --fit-on")

        # Batch size, micro batch size, and threads are included when set, with -t skipped for -1
        parts.append(f" -b {self.batch_size}")
        parts.append(f" -ub {self.micro_batch_size}")
        if self.threads != -1:
            parts.append(f" -t {self.threads}")

        # Thread batch (only when set to non-zero)
        if self.thread_batch > 0:
            parts.append(f" -tb {self.thread_batch}")

        # Speculative decoding (only when enabled)
        if self.spec_enabled:
            parts.append(f" --spec-type {self.spec_type}")

        # Cache types are always included
        parts.append(f" -ctk {self.cache_type_k}")
        parts.append(f" -ctv {self.cache_type_v}")

        # Add context size only when the toggle is enabled
        if self.ctx_size_enabled:
            ctx_val = max(2, min(int(str(self.ctx_size_value)), 999999999))
            parts.append(f" --ctx-size {ctx_val}")

        # Server settings always included
        host = str(self.host).strip() or "0.0.0.0"
        port = max(1, min(int(str(self.port)), 65535))
        parts.append(f" --host {host}")
        parts.append(f" --port {port}")

        # Sampling params (always included)
        temp = max(0.05, min(float(str(self.temperature)), 2.0))
        minp = min(max(float(str(self.min_p)), -1.0), 1.0)
        topk = max(1, int(str(self.top_k)))
        pp = min(max(float(str(self.presence_penalty)), -2.0), 2.0)
        topp = min(max(float(str(self.top_p)), 0.05), 1.0)
        rp = min(max(float(str(self.repeat_penalty)), 1.0), 3.0)

        parts.append(f" --temp {temp:.2f}")
        parts.append(f" --min-p {minp:.2f}")
        parts.append(f" --top-k {topk}")
        parts.append(f" --presence-penalty {pp:.2f}")
        parts.append(f" --top-p {topp:.3f}")
        parts.append(f" --repeat-penalty {rp:.2f}")

        return "".join(parts)

    @staticmethod
    def _safe_int(name, default):
        """Safely convert a value to int; fall back on *default*."""
        try:
            val = str(default) if isinstance(default, str) else default
            return int(val)
        except (ValueError, TypeError):
            return default

    def to_dict(self):
        """Return a dict of all mutable flag state for persistence."""
        return {
            "model_path": self.model_path,
            "no_mmap": self.no_mmap,
            "mlock": self.mlock,
            "ctx_size_enabled": self.ctx_size_enabled,
            "ctx_size_value": self.ctx_size_value,
            "n_gpu_layers": self.n_gpu_layers,
            "host": self.host,
            "port": self.port,
            "num_threads": self.num_threads,
            "threads_enabled": self.threads_enabled,
            "flash_attention": self.flash_attention,
            "fit_on": self.fit_on,
            "spec_enabled": self.spec_enabled,
            "spec_type": self.spec_type,
            "batch_size": self.batch_size,
            "micro_batch_size": self.micro_batch_size,
            "threads": self.threads,
            "thread_batch": self.thread_batch,
            "cache_type_k": self.cache_type_k,
            "cache_type_v": self.cache_type_v,
            "temperature": self.temperature,
            "min_p": self.min_p,
            "top_k": self.top_k,
            "presence_penalty": self.presence_penalty,
            "top_p": self.top_p,
            "repeat_penalty": self.repeat_penalty,
        }

    def from_dict(self, d):
        """Restore mutable flag state from a saved dict."""
        for key, val in d.items():
            if hasattr(self, key):
                setattr(self, key, val)


# UI builder — all ttk widget frames returned as methods
# Each section method creates a LabelFrame with its widgets and packs it into *parent*
# Live command generation is triggered by Tk variable traces on every input field

class LlamaServerGUI:
    """Main tkinter GUI for configuring llama-server flags and generating commands."""

    def __init__(self, root):
        self.root = root
        self.config = FlagConfig()
        self._last_folder = _load_last_folder()

        # Save config on window close
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Create all Tk variables (StringVar / IntVar) so every widget change triggers live updates
        sv_model_path = tk.StringVar(value="")                    # model path display var
        iv_auto_gpu = tk.IntVar(value=-1)                         # n-gpu-layers (-1=auto, 0-99)
        lv_bool_no_mmap = tk.BooleanVar(value=False)              # no-mmap toggle state variable (boolean)
        ml_bool_mlock = tk.BooleanVar(value=False)                # mlock toggle state variable (boolean)

        iv_ctx_enabled = tk.BooleanVar(value=False)               # toggle to show ctx-size input

        # Flash Attention / Fit On checkboxes
        iv_flash_attn = tk.BooleanVar(value=False)
        iv_fit_on = tk.BooleanVar(value=False)

        # Speculative decoding toggle
        iv_spec_enabled = tk.BooleanVar(value=False)

        # Batch size, micro batch, threads spinboxes
        iv_batch_size = tk.IntVar(value=2048)
        iv_micro_batch = tk.IntVar(value=512)
        iv_threads_val_new = tk.IntVar(value=-1)
        iv_thread_batch = tk.IntVar(value=0)

        # Cache type K and V dropdowns
        CACHE_TYPES = ["f16", "f32", "q8_0", "q4_0", "q4_1", "iq4_nl"]
        sv_cache_k = tk.StringVar(value="f16")
        sv_cache_v = tk.StringVar(value="f16")

        sv_host = tk.StringVar(value="0.0.0.0")                   # server host binding address
        iv_port = tk.IntVar(value=8080)                           # HTTP port (1-65535)

        iv_threads_enabled = tk.BooleanVar(value=False)           # toggle to show threads input
        iv_threads_val = tk.IntVar(value=os.cpu_count() or 4)     # thread count default from CPU cores

        sv_temp = tk.DoubleVar(value=0.8)                         # temperature (float 0.05–2.0)
        sv_topk = tk.IntVar(value=40)                              # top-k integer value >= 1
        sv_topp = tk.DoubleVar(value=0.95)                        # top-p float between [0.05, 1.0]
        sv_rp = tk.DoubleVar(value=1.1)                           # repeat penalty (float > 1.0)

        self._vars = {
            "model_path": sv_model_path,
            "n_gpu_layers": iv_auto_gpu,
            "no_mmap": lv_bool_no_mmap,
            "mlock": ml_bool_mlock,
            "ctx_size_enabled": iv_ctx_enabled,
            "flash_attention": iv_flash_attn,
            "fit_on": iv_fit_on,
            "spec_enabled": iv_spec_enabled,
            "batch_size": iv_batch_size,
            "micro_batch": iv_micro_batch,
            "threads_val": iv_threads_val_new,
            "thread_batch": iv_thread_batch,
            "cache_type_k": sv_cache_k,
            "cache_type_v": sv_cache_v,
            "host": sv_host,
            "port": iv_port,
            "threads_enabled": iv_threads_enabled,
            "num_threads": iv_threads_val,
            "temperature": sv_temp,
            "top_k": sv_topk,
            "top_p": sv_topp,
            "repeat_penalty": sv_rp,
        }

        # Store all Tk variables on self for cross-method access
        self._tk = {
            "model_path": sv_model_path,
            "n_gpu_layers": iv_auto_gpu,
            "no_mmap": lv_bool_no_mmap,
            "mlock": ml_bool_mlock,
            "ctx_size_enabled": iv_ctx_enabled,
            "flash_attention": iv_flash_attn,
            "fit_on": iv_fit_on,
            "spec_enabled": iv_spec_enabled,
            "batch_size": iv_batch_size,
            "micro_batch": iv_micro_batch,
            "threads_val": iv_threads_val_new,
            "thread_batch": iv_thread_batch,
            "cache_type_k": sv_cache_k,
            "cache_type_v": sv_cache_v,
            "host": sv_host,
            "port": iv_port,
            "threads_enabled": iv_threads_enabled,
            "num_threads": iv_threads_val,
            "temperature": sv_temp,
            "top_k": sv_topk,
            "top_p": sv_topp,
            "repeat_penalty": sv_rp,
        }

        # Load saved config
        saved = _load_config()
        saved_flags = saved.pop("flags", {})
        if saved_flags:
            import sys as _sys2
            print(f"DEBUG from_dict flags keys: {list(saved_flags.keys())}", file=_sys2.stderr)
            print(f"DEBUG spec_type in flags: {'spec_type' in saved_flags}, val={saved_flags.get('spec_type')!r}", file=_sys2.stderr)
            self.config.from_dict(saved_flags)
            print(f"DEBUG after from_dict spec_type={self.config.spec_type!r}", file=_sys2.stderr)
        self._restore_vars(saved_flags)
        print(f"DEBUG after restore spec_type={self.config.spec_type!r}", file=_sys2.stderr)

        # Change handlers — update config state and trigger command rebuild
        def _on_no_mmap_change(*_):
            try:
                if lv_bool_no_mmap.get():
                    ml_bool_mlock.set(False)
                    self.config.mlock = False
                self.config.no_mmap = bool(lv_bool_no_mmap.get())
            except Exception:
                pass

        def _on_mlock_change(*_):
            try:
                if ml_bool_mlock.get():
                    lv_bool_no_mmap.set(False)
                    self.config.no_mmap = False
                self.config.mlock = bool(ml_bool_mlock.get())
            except Exception:
                pass

        def _on_model_change(*_):
            self.config.model_path = sv_model_path.get()

        def _on_gpu_layers_change(*_):
            try:
                val = int(iv_auto_gpu.get())
                if not (-1 <= val <= 99): return
                self.config.n_gpu_layers = val
            except (ValueError, TypeError):
                pass

        def _on_ctx_enabled_change(*_):
            self.config.ctx_size_enabled = bool(iv_ctx_enabled.get())

        def _on_flash_attn_change(*_):
            try:
                self.config.flash_attention = bool(iv_flash_attn.get())
            except Exception:
                pass

        def _on_fit_on_change(*_):
            try:
                self.config.fit_on = bool(iv_fit_on.get())
            except Exception:
                pass

        def _on_batch_size_change(*_):
            try:
                val = int(iv_batch_size.get())
                self.config.batch_size = max(1, min(val, 8192))
            except (ValueError, TypeError):
                pass

        def _on_micro_batch_change(*_):
            try:
                val = int(iv_micro_batch.get())
                self.config.micro_batch_size = max(1, min(val, 8192))
            except (ValueError, TypeError):
                pass

        def _on_threads_new_change(*_):
            try:
                val = int(iv_threads_val_new.get())
                self.config.threads = max(-1, min(val, 128))
            except (ValueError, TypeError):
                pass

        def _on_cache_k_change(*_):
            try:
                val = sv_cache_k.get()
                if val in CACHE_TYPES:
                    self.config.cache_type_k = val
            except Exception:
                pass

        def _on_cache_v_change(*_):
            try:
                val = sv_cache_v.get()
                if val in CACHE_TYPES:
                    self.config.cache_type_v = val
            except Exception:
                pass

        def _on_host_change(*_):
            self.config.host = sv_host.get() or "0.0.0.0"

        def _on_port_change(*_):
            try:
                val = int(iv_port.get()) if iv_port.get() else 8080
                self.config.port = max(1, min(val, 65535))
            except (ValueError, TypeError):
                pass
            self._update_command()

        def _on_threads_enabled_change(*_):
            self.config.threads_enabled = bool(iv_threads_enabled.get())

        def _on_num_threads_change(*_):
            try:
                val = int(iv_threads_val.get())
                if not (1 <= val <= 256): return
                self.config.num_threads = max(1, min(val, 256))
            except (ValueError, TypeError):
                pass





        # Register all traces on the Tk variables so every change triggers live command update
        lv_bool_no_mmap.trace_add("write", lambda *_: (_on_no_mmap_change(), self._update_command()))
        ml_bool_mlock.trace_add("write", lambda *_: (_on_mlock_change(), self._update_command()))
        iv_auto_gpu.trace_add("write", lambda *_: (_on_gpu_layers_change(), self._update_command()))

        # Context size toggle and input
        def _ctx_toggle_wrapper(*_):
            try:
                _on_ctx_enabled_change()
                self._update_command()
            except Exception:
                pass
        iv_ctx_enabled.trace_add("write", lambda *_: (_ctx_toggle_wrapper(),))

        def _host_trace_wrapper(*_):
            try:
                val = sv_host.get() or "0.0.0.0"
                self.config.host = val
                self._update_command()
            except Exception:
                pass
        sv_host.trace_add("write", lambda *_: (_on_host_change(), _host_trace_wrapper()))
        def _port_trace_wrapper(*_):
            try:
                val = int(iv_port.get()) if iv_port.get() else 8080
                self.config.port = max(1, min(val, 65535))
                self._update_command()
            except (ValueError, TypeError):
                pass
        iv_port.trace_add("write", lambda *_: (_on_port_change(), _port_trace_wrapper()))

        # Threads toggle and input
        def _threads_toggle_wrapper(*_):
            try:
                if not self.config.threads_enabled:
                    iv_threads_enabled.set(True)
                _on_threads_enabled_change()
                self._update_command()
            except Exception:
                pass
        iv_threads_enabled.trace_add("write", lambda *_: (_threads_toggle_wrapper(),))

        # Flash Attention / Fit On traces
        iv_flash_attn.trace_add("write", lambda *_: (_on_flash_attn_change(), self._update_command()))
        iv_fit_on.trace_add("write", lambda *_: (_on_fit_on_change(), self._update_command()))

        def _save_on_close(*_):
            try:
                data = _load_config()
                data["flags"] = self.config.to_dict()
                data["last_folder"] = self._last_folder
                _save_config(data)
            except Exception:
                pass
        iv_spec_enabled.trace_add("write", lambda *_: (self.config.__setattr__("spec_enabled", bool(iv_spec_enabled.get())), self._update_command(), _save_on_close()))

        # Batch size, micro batch, and threads traces
        iv_batch_size.trace_add("write", lambda *_: (_on_batch_size_change(), self._update_command()))
        iv_micro_batch.trace_add("write", lambda *_: (_on_micro_batch_change(), self._update_command()))
        iv_threads_val_new.trace_add("write", lambda *_: (_on_threads_new_change(), self._update_command()))

        # Cache type K / V traces
        sv_cache_k.trace_add("write", lambda *_: (_on_cache_k_change(), self._update_command()))
        sv_cache_v.trace_add("write", lambda *_: (_on_cache_v_change(), self._update_command()))

        # Build the full UI
        self._build_ui()


# ---------------------------------------------------------------------------
# Section builders — each returns a ttk.Frame packed into *parent*
# Each section method creates a LabelFrame with its widgets and packs it into *parent*
# Live command generation is triggered by Tk variable traces on every input field
# ---------------------------------------------------------------------------

    def _on_close(self):
        """Save config (flags + folder) before closing."""
        try:
            data = _load_config()
            data["flags"] = self.config.to_dict()
            data["last_folder"] = self._last_folder
            _save_config(data)
        except Exception:
            pass
        self.root.destroy()

    def _restore_vars(self, saved_flags):
        """Set Tk variable values to match saved flag state."""
        tk = self._tk
        if "n_gpu_layers" in saved_flags:
            try:
                tk["n_gpu_layers"].set(int(saved_flags["n_gpu_layers"]))
            except (ValueError, TypeError):
                pass
        if "ctx_size_enabled" in saved_flags:
            try:
                tk["ctx_size_enabled"].set(bool(saved_flags["ctx_size_enabled"]))
            except (ValueError, TypeError):
                pass
        if "flash_attention" in saved_flags:
            try:
                tk["flash_attention"].set(bool(saved_flags["flash_attention"]))
            except (ValueError, TypeError):
                pass
        if "fit_on" in saved_flags:
            try:
                tk["fit_on"].set(bool(saved_flags["fit_on"]))
            except (ValueError, TypeError):
                pass
        if "spec_enabled" in saved_flags:
            try:
                tk["spec_enabled"].set(bool(saved_flags["spec_enabled"]))
            except (ValueError, TypeError):
                pass
        if "spec_type" in saved_flags:
            try:
                self.config.spec_type = str(saved_flags["spec_type"])
            except (ValueError, TypeError):
                pass
        if "batch_size" in saved_flags:
            try:
                tk["batch_size"].set(int(saved_flags["batch_size"]))
            except (ValueError, TypeError):
                pass
        if "micro_batch" in saved_flags:
            try:
                tk["micro_batch"].set(int(saved_flags["micro_batch"]))
            except (ValueError, TypeError):
                pass
        if "threads_val" in saved_flags:
            try:
                tk["threads_val"].set(int(saved_flags["threads_val"]))
            except (ValueError, TypeError):
                pass
        if "cache_type_k" in saved_flags:
            try:
                tk["cache_type_k"].set(str(saved_flags["cache_type_k"]))
            except (ValueError, TypeError):
                pass
        if "cache_type_v" in saved_flags:
            try:
                tk["cache_type_v"].set(str(saved_flags["cache_type_v"]))
            except (ValueError, TypeError):
                pass
        if "host" in saved_flags:
            try:
                tk["host"].set(str(saved_flags["host"]))
            except (ValueError, TypeError):
                pass
        if "port" in saved_flags:
            try:
                tk["port"].set(int(saved_flags["port"]))
            except (ValueError, TypeError):
                pass
        if "temperature" in saved_flags:
            try:
                tk["temperature"].set(float(saved_flags["temperature"]))
            except (ValueError, TypeError):
                pass
        if "top_k" in saved_flags:
            try:
                tk["top_k"].set(int(saved_flags["top_k"]))
            except (ValueError, TypeError):
                pass
        if "top_p" in saved_flags:
            try:
                tk["top_p"].set(float(saved_flags["top_p"]))
            except (ValueError, TypeError):
                pass
        if "repeat_penalty" in saved_flags:
            try:
                tk["repeat_penalty"].set(float(saved_flags["repeat_penalty"]))
            except (ValueError, TypeError):
                pass

    def _build_ui(self):
        """Construct all sections and pack them into a scrollable canvas."""
        root = self.root

        # Use grid so the window layout expands naturally
        root.grid_rowconfigure(0, weight=1)
        root.grid_columnconfigure(0, weight=1)

        outer_frame = ttk.Frame(root, padding=(6, 4))
        outer_frame.grid(row=0, column=0, sticky="nsew")

        # Title label at the top, bold and larger for clarity
        title_label = ttk.Label(
            outer_frame, text="llama-server command generator", anchor="center",
            font=("Segoe UI", 16, "bold")
        )
        title_label.grid(row=0, column=0, sticky="ew", pady=(8, 4))

        # Scrollable canvas area holding the section frames
        inner_frame = ttk.Frame(outer_frame)
        inner_frame.grid(row=1, column=0, sticky="nswe")
        outer_frame.columnconfigure(0, weight=1)

        # Configure the canvas so it resizes when the window changes
        def _on_canvas_resize(event):
            if hasattr(self, 'canvas') and event.widget == inner_frame:
                try: self.canvas.config(width=event.width - 20)
                except Exception: pass
        
        inner_frame.bind("<Configure>", _on_canvas_resize)
        self.canvas = tk.Canvas(inner_frame, highlightthickness=0, bg="#f5f5f5")
        scrollbar = ttk.Scrollbar(
            inner_frame, orient="vertical", command=self.canvas.yview
        )
        self.scrollable = ttk.Frame(self.canvas)
        scroll_content_id = self.canvas.create_window((0, 5), window=self.scrollable, anchor="nw")

        def _on_scroll_configure(event):
            """Update the scroll region whenever section frames change size."""
            bbox = self.canvas.bbox("all")
            if bbox:
                self.canvas.configure(scrollregion=(*bbox[:4],))

        def _update_content_width(event):
            """Resize the scrollable frame to match parent canvas width."""
            w = event.width or root.winfo_reqwidth()
            self.canvas.itemconfigure(scroll_content_id, width=w)
            bbox = self.canvas.bbox("all")
            if bbox:
                self.canvas.configure(scrollregion=(*bbox[:4],))

        def _on_scroll(event):
            """Handle scroll + resize."""
            return (_update_content_width(event), _on_scroll_configure(event))

        self.scrollable.bind("<Configure>", lambda e: _on_scroll(e))

        # Keep canvas and frames expanding properly horizontally
        def _on_window_resize(event):
            if hasattr(self, 'canvas') and event.widget in (outer_frame, inner_frame):
                try:
                    w = max(20, outer_frame.winfo_width() - 40)
                    h = max(100, root.winfo_height() - 80)  # grow height with window
                    self.canvas.config(width=w)
                    self.canvas.itemconfigure(scroll_content_id, width=w - 15)
                    self.canvas.config(height=h)
                except Exception: pass
                # Grow the command box vertically as the window grows
                if hasattr(self, 'cmd_text'):
                    try:
                        new_h = max(5, int((h - 200) / 18))  # ~18px per line, reserve 200 for other UI
                        self.cmd_text.configure(height=new_h)
                    except Exception: pass
        
        outer_frame.bind("<Configure>", _on_window_resize)
        root.grid_rowconfigure(0, weight=1)
        root.grid_columnconfigure(0, weight=1)

        # Keep Copy/Run/Save buttons outside the scrolling area so they stay visible
        copy_frame = ttk.Frame(outer_frame)
        copy_frame.grid(row=2, column=0, sticky="ew")
        self._copy_btn = ttk.Button(
            copy_frame, text="\U0001F4CB Copy", command=self._copy_command
        )
        self._copy_btn.pack(side="right", padx=(0, 4))
        self._run_btn = ttk.Button(
            copy_frame, text="\u25B6 Run in CMD", command=self._run_in_cmd
        )
        self._run_btn.pack(side="right", padx=(0, 4))
        self._save_btn = ttk.Button(
            copy_frame, text="\U0001F4BE Save as .bat", command=self._save_bat_command
        )
        self._save_btn.pack(side="right")

        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True, pady=(0, 4))
        scrollbar.pack(side="right", fill="y")

        # Bind mouse wheel to canvas scrolling
        def _on_mousewheel(event):
            direction = -1 if event.delta > 0 else 1
            self.canvas.yview_scroll(int(1.5 * abs(direction)), "units" if direction < 0 else "pages")

        root.bind_all("<MouseWheel>", lambda e: _on_mousewheel(e))

        # Build each section into the scrollable frame in order
        self._section_model_loading(self.scrollable)
        self._section_context_gpu(self.scrollable)
        self._section_server_settings(self.scrollable)
        self._section_sampling_params(self.scrollable)

        # Place the generated command area at the bottom of the canvas
        cmd_frame = ttk.LabelFrame(
            self.scrollable, text="Generated Command", padding=(10, 8)
        )
        cmd_frame.pack(fill="both", padx=6, pady=(4, 2))

        # Frame containing the command text box and its scrollbar
        txt_row = ttk.Frame(cmd_frame)
        txt_row.pack(fill="both", expand=True)

        self.cmd_text = tk.Text(txt_row, height=5, wrap="word")
        self.cmd_text.pack(side="left", fill="x", expand=True)

        cmd_scrollbar = ttk.Scrollbar(
            txt_row, orient="vertical",
            command=self.cmd_text.yview
        )
        cmd_scrollbar.pack(side="right", fill="y")
        self.cmd_text.configure(yscrollcommand=cmd_scrollbar.set)



        # Render the initial command immediately on startup
        self._update_command()


    def _section_model_loading(self, parent):
        """Model loading section: browse button + No-MMAP / MLock toggles."""
        frame = ttk.LabelFrame(parent, text="Model Loading", padding=(8, 6))
        frame.pack(fill="both", padx=6, pady=4)

        # Browse row with a file dialog so the user can pick a .gguf model
        btn_row = ttk.Frame(frame)
        btn_row.pack(fill="x")

        browse_btn = ttk.Button(btn_row, text="Browse...", command=self._browse_model)
        browse_btn.pack(side="left", padx=(0, 6))

        # Read-only label shows the selected model path and updates automatically
        self.model_path_label = tk.Label(
            btn_row, text="(no model selected)", anchor="w", justify="left"
        )
        self.model_path_label.pack(side="left", fill="x", expand=True)

        # Bind trace so the model path label updates when the path changes
        sv_model_path = self._tk["model_path"]
        def _update_model_label(*_):
            val = sv_model_path.get() or "(no model selected)"
            self.model_path_label.config(text=val)
            self.config.model_path = val

        sv_model_path.trace_add("write", lambda *_: (_update_model_label(),))

        # Place No-MMAP and MLock checkboxes side by side
        check_row = ttk.Frame(frame)
        check_row.pack(fill="x", pady=(4, 0))

        lv_bool_no_mmap = self._tk["no_mmap"]
        ml_bool_mlock = self._tk["mlock"]

        tk.Checkbutton(check_row, text="No-MMAP", variable=lv_bool_no_mmap).pack(side="left")
        tk.Checkbutton(check_row, text="MLock", variable=ml_bool_mlock).pack(side="left", padx=(12, 0))


    def _section_context_gpu(self, parent):
        """Context size & GPU layers section (side-by-side columns)."""
        frame = ttk.LabelFrame(parent, text="Performance", padding=(8, 6))
        frame.pack(fill="both", padx=6, pady=4)

        # Left column is for Context Size options
        ctx_frame = ttk.Frame(frame)
        ctx_frame.grid(row=0, column=0, sticky="nsew")

        iv_ctx_enabled = self._tk["ctx_size_enabled"]

        def _on_ctx_change(*_):
            enabled = bool(iv_ctx_enabled.get())
            if hasattr(self, '_ctx_input_widgets'):
                for w in self._ctx_input_widgets:
                    if enabled and not w.winfo_viewable():
                        w.pack()
                    elif not enabled and w.winfo_viewable():
                        w.pack_forget()

        # Context toggle checkbox
        tk.Checkbutton(ctx_frame, text="Enable context size", variable=iv_ctx_enabled).pack(side="left")

        # Entry and label for ctx-size value, shown only when enabled
        self._ctx_input_widgets = []  # track widgets that need packing or unpacking

        iv_ctx_var = tk.IntVar(value=512)

        def _on_ctx_val(*_):
            val = max(2, min(int(iv_ctx_var.get()), 999999999)) if iv_ctx_var.get() else 512
            iv_ctx_var.set(val)
            self.config.ctx_size_enabled = bool(iv_ctx_enabled.get())
            self.config.ctx_size_value = val

        ctx_label = ttk.Label(ctx_frame, text="Ctx Size")
        entry_c = ttk.Entry(ctx_frame, textvariable=iv_ctx_var, width=8)
        for w in [ctx_label, entry_c]:
            self._ctx_input_widgets.append(w)

        def _ctx_trace_wrapper(*_):
            self.config.ctx_size_enabled = bool(iv_ctx_enabled.get())

        def _ctx_value_trace(*_):
            try:
                val = max(2, min(int(iv_ctx_var.get()), 999999999)) if iv_ctx_var.get() else 512
                self.config.ctx_size_value = val
                self.config.ctx_size_enabled = bool(iv_ctx_enabled.get())
            except Exception:
                pass

        iv_ctx_enabled.trace_add("write", lambda *_: (_on_ctx_change(), _ctx_trace_wrapper(), _ctx_value_trace(), self._update_command()))
        iv_ctx_var.trace_add("write", lambda *_: (_ctx_value_trace(), self._update_command()))

        # Right column holds GPU layer settings
        gpu_frame = ttk.Frame(frame)
        gpu_frame.grid(row=0, column=1, sticky="nsew", padx=(8, 0))

        iv_auto_gpu = self._tk["n_gpu_layers"]

        label = ttk.Label(gpu_frame, text="GPU Layers")
        label.pack(side="left")

        # Spinbox for GPU layers from -1 to 99
        spinvar = tk.IntVar(value=-1)

        def _on_spinval(*_):
            try:
                val = int(spinvar.get())
                if not (-1 <= val <= 99): return
                self.config.n_gpu_layers = max(-1, min(val, 99))
            except (ValueError, TypeError):
                pass

        def _gpu_trace_wrapper(*_):
            try:
                spinvar.set(int(iv_auto_gpu.get()))
                self._update_command()
            except Exception:
                pass

        iv_auto_gpu.trace_add("write", lambda *_: (_on_spinval(), _gpu_trace_wrapper()))

        def _spinvar_safe(*_):
            try:
                val = int(spinvar.get())
                if not (-1 <= val <= 99): return
                self.config.n_gpu_layers = max(-1, min(val, 99))
                iv_auto_gpu.set(max(-1, min(val, 99)))
                spinvar.set(max(-1, min(val, 99)))
            except (ValueError, TypeError):
                pass

        spinvar.trace_add("write", lambda *_: (_spinvar_safe(), _on_gpu_layers_change(), self._update_command()))

        ttk.Spinbox(gpu_frame, from_=-1, to=99, textvariable=spinvar, width=5).pack(side="left")

        # Batch size, micro batch, and thread handlers local to this section
        def _on_batch_size_change(*_):
            try:
                val = int(iv_batch_size.get())
                self.config.batch_size = max(1, min(val, 8192))
            except (ValueError, TypeError):
                pass

        def _on_micro_batch_change(*_):
            try:
                val = int(iv_micro_batch.get())
                self.config.micro_batch_size = max(1, min(val, 8192))
            except (ValueError, TypeError):
                pass

        def _on_threads_new_change(*_):
            try:
                val = int(iv_threads_val_new.get())
                self.config.threads = max(-1, min(val, 128))
            except (ValueError, TypeError):
                pass

        def _on_thread_batch_change(*_):
            try:
                val = int(iv_thread_batch.get())
                if not (-1 <= val <= 512): return
                self.config.thread_batch = max(-1, min(val, 512))
            except (ValueError, TypeError):
                pass

        # --- Flash Attention and Fit On checkboxes (row 1, full width) ---
        fa_row = ttk.Frame(frame)
        fa_row.grid(row=1, column=0, columnspan=2, sticky="w", pady=(6, 0))

        iv_flash_attn = self._tk["flash_attention"]
        iv_fit_on = self._tk["fit_on"]
        tk.Checkbutton(fa_row, text="Flash Attention (-fa)", variable=iv_flash_attn).pack(side="left")
        tk.Checkbutton(fa_row, text="Fit On (--fit-on)", variable=iv_fit_on).pack(side="left", padx=(16, 0))

        # --- Speculative Decoding (row 2, full width) ---
        spec_row = ttk.Frame(frame)
        spec_row.grid(row=2, column=0, columnspan=2, sticky="w", pady=(4, 0))

        iv_spec_enabled = self._tk["spec_enabled"]
        import sys as _sys3
        print(f"DEBUG UI spec_type={self.config.spec_type!r}", file=_sys3.stderr)
        sv_spec_type = tk.StringVar(value=self.config.spec_type)
        def _on_spec_type_change(*_):
            try:
                val = sv_spec_type.get()
                if val in ("ngram-mod", "draft-mtp"):
                    self.config.spec_type = val
            except Exception:
                pass
        def _save_on_change(*_):
            try:
                data = _load_config()
                data["flags"] = self.config.to_dict()
                data["last_folder"] = self._last_folder
                _save_config(data)
            except Exception:
                pass
        sv_spec_type.trace_add("write", lambda *_: (_on_spec_type_change(), self._update_command(), _save_on_change()))

        tk.Checkbutton(spec_row, text="Speculative Decoding", variable=iv_spec_enabled).pack(side="left")
        ttk.OptionMenu(spec_row, sv_spec_type, "ngram-mod", "ngram-mod", "draft-mtp").pack(side="left", padx=(8, 0))

        # --- Batch Size, Micro-Batch, and Threads spinboxes (row 3, three columns) ---
        mb_row = ttk.Frame(frame)
        mb_row.grid(row=3, column=0, columnspan=2, sticky="w", pady=(4, 0))

        iv_batch_size = self._tk["batch_size"]
        iv_micro_batch = self._tk["micro_batch"]
        iv_threads_val_new = self._tk["threads_val"]
        iv_thread_batch = self._tk["thread_batch"]

        def _make_spinbox_factory(var, label, lo, hi, on_change):
            def _spin_safe(s):
                def _inner(*_):
                    try:
                        val = int(s.get())
                        if not (lo <= val <= hi): return
                        var.set(max(lo, min(val, hi)))
                        s.set(max(lo, min(val, hi)))
                    except (ValueError, TypeError):
                        pass
                return _inner
            def _spin_cmd(s):
                def _inner(*_):
                    try: self._update_command()
                    except Exception:
                        pass
                return _inner
            return _spin_safe, _spin_cmd, on_change

        for var, label, lo, hi, handler in [
            (iv_batch_size, "Batch Size", 1, 8192, _on_batch_size_change),
            (iv_micro_batch, "Micro-Batch", 1, 8192, _on_micro_batch_change),
            (iv_threads_val_new, "Threads", -1, 128, _on_threads_new_change),
            (iv_thread_batch, "Thread Batch", 0, 512, _on_thread_batch_change),
        ]:
            col = ttk.Frame(mb_row)
            col.pack(side="left", padx=(0, 12))
            ttk.Label(col, text=label).pack(side="left")

            _spin_safe, _spin_cmd, on_change = _make_spinbox_factory(var, label, lo, hi, handler)
            var.trace_add("write", lambda *_: (_spin_safe(var), on_change(), _spin_cmd(var), self._update_command()))
            ttk.Spinbox(col, from_=lo, to=hi, textvariable=var, width=7).pack(side="left")

        # --- Cache Type K and V dropdowns (row 4, two columns) ---
        ct_row = ttk.Frame(frame)
        ct_row.grid(row=4, column=0, columnspan=2, sticky="w", pady=(4, 0))

        sv_cache_k = self._tk["cache_type_k"]
        sv_cache_v = self._tk["cache_type_v"]

        for var, label in [(sv_cache_k, "Cache K"), (sv_cache_v, "Cache V")]:
            col = ttk.Frame(ct_row)
            col.pack(side="left", padx=(0, 24))
            ttk.Label(col, text=label).pack(side="left")
            cache_menu = ttk.OptionMenu(col, var, var.get(), "f16", "f32", "q8_0", "q4_0", "q4_1", "iq4_nl")
            cache_menu.pack(side="left")

    def _section_server_settings(self, parent):
        """Server settings section (always visible — sensible defaults)."""
        frame = ttk.LabelFrame(parent, text="Network & Server", padding=(8, 6))
        frame.pack(fill="both", padx=6, pady=4)

        # Host and Port row
        net_row = ttk.Frame(frame)
        net_row.pack(fill="x")

        sv_host = self._tk["host"]
        iv_port = self._tk["port"]

        def _on_host_change(*_):
            val = sv_host.get() or "0.0.0.0"
            sv_host.set(val)
            self.config.host = val

        host_frame = ttk.Frame(net_row)
        host_frame.pack(side="left")
        ttk.Label(host_frame, text="Host:").pack(side="left", padx=(0, 4))
        entry_h = ttk.Entry(host_frame, textvariable=sv_host, width=12)
        entry_h.pack(side="left")

        def _on_port_change(*_):
            val = max(1, min(int(iv_port.get()), 65535)) if iv_port.get() else 8080
            iv_port.set(val)
            self.config.port = val
            self._update_command()

        port_frame = ttk.Frame(net_row)
        port_frame.pack(side="left", padx=(24, 0))
        ttk.Label(port_frame, text="Port:").pack(side="left", padx=(0, 4))
        def _port_spin_safe(*_):
            try:
                val = int(iv_port.get())
                if not (1 <= val <= 65535): return
                iv_port.set(max(1, min(val, 65535)))
                self.config.port = max(1, min(val, 65535))
            except (ValueError, TypeError):
                pass
        def _port_spin_cmd(*_):
            try: self._update_command()
            except Exception:
                pass
        iv_port.trace_add("write", lambda *_: (_port_spin_safe(), _port_spin_cmd()))

        entry_p = ttk.Spinbox(port_frame, from_=1, to=65535, textvariable=iv_port, width=8)
        entry_p.pack(side="left")


    def _section_sampling_params(self, parent):
        """Sampling parameters section (always visible, 2-column grid)."""
        samp_frame = ttk.Frame(parent)
        samp_frame.pack(fill="both", padx=6, pady=4)

        # --- Temperature (row 0, col 0) ---
        sv_temp = tk.DoubleVar(value=0.8)
        def _temp_safe(*_):
            try:
                val = float(sv_temp.get()) if sv_temp.get() else 0.8
                self.config.temperature = max(0.05, min(val, 2.0))
                sv_temp.set(max(0.05, min(val, 2.0)))
            except Exception:
                pass
        def _temp_cmd(*_):
            try: self._update_command()
            except Exception:
                pass
        sv_temp.trace_add("write", lambda *_: (_temp_safe(), _temp_cmd()))
        ttk.Label(samp_frame, text="Temperature").grid(row=0, column=0, sticky="w", padx=(4, 0), pady=1)
        ttk.Spinbox(samp_frame, from_=0.05, to=2.0, increment=0.05, width=8,
                    textvariable=sv_temp).grid(row=0, column=0, sticky="w", padx=(100, 0), pady=1)

        # --- Min-P (row 0, col 1) ---
        sv_minp = tk.DoubleVar(value=0.0)
        def _minp_safe(*_):
            try:
                val = float(sv_minp.get()) if sv_minp.get() else 0.0
                self.config.min_p = min(max(val, -1.0), 1.0)
                sv_minp.set(min(max(val, -1.0), 1.0))
            except Exception:
                pass
        def _minp_cmd(*_):
            try: self._update_command()
            except Exception:
                pass
        sv_minp.trace_add("write", lambda *_: (_minp_safe(), _minp_cmd()))
        ttk.Label(samp_frame, text="Min-P").grid(row=0, column=1, sticky="w", padx=(40, 0), pady=1)
        ttk.Spinbox(samp_frame, from_=-1.0, to=1.0, increment=0.01, width=8,
                    textvariable=sv_minp).grid(row=0, column=1, sticky="w", padx=(140, 0), pady=1)

        # --- Top-K (row 1, col 0) ---
        sv_topk = tk.IntVar(value=40)
        def _topk_safe(*_):
            try:
                val = int(sv_topk.get())
                if not (1 <= val <= 9999): return
                self.config.top_k = max(1, min(val, 9999))
                sv_topk.set(max(1, min(val, 9999)))
            except (ValueError, TypeError):
                pass
        def _topk_cmd(*_):
            try: self._update_command()
            except Exception:
                pass
        sv_topk.trace_add("write", lambda *_: (_topk_safe(), _topk_cmd()))
        ttk.Label(samp_frame, text="Top-K").grid(row=1, column=0, sticky="w", padx=(4, 0), pady=1)
        ttk.Spinbox(samp_frame, from_=1, to=9999, increment=1, width=8,
                    textvariable=sv_topk).grid(row=1, column=0, sticky="w", padx=(100, 0), pady=1)

        # --- Presence Penalty (row 1, col 1) ---
        sv_pp = tk.DoubleVar(value=0.0)
        def _pp_safe(*_):
            try:
                val = float(sv_pp.get()) if sv_pp.get() else 0.0
                self.config.presence_penalty = min(max(val, -2.0), 2.0)
                sv_pp.set(min(max(val, -2.0), 2.0))
            except Exception:
                pass
        def _pp_cmd(*_):
            try: self._update_command()
            except Exception:
                pass
        sv_pp.trace_add("write", lambda *_: (_pp_safe(), _pp_cmd()))
        ttk.Label(samp_frame, text="Presence Pen.").grid(row=1, column=1, sticky="w", padx=(40, 0), pady=1)
        ttk.Spinbox(samp_frame, from_=-2.0, to=2.0, increment=0.1, width=8,
                    textvariable=sv_pp).grid(row=1, column=1, sticky="w", padx=(160, 0), pady=1)

        # --- Top-P (row 2, col 0) ---
        sv_topp = tk.DoubleVar(value=0.95)
        def _topp_safe(*_):
            try:
                val = float(sv_topp.get()) if sv_topp.get() else 0.95
                self.config.top_p = min(max(val, 0.05), 1.0)
                sv_topp.set(min(max(val, 0.05), 1.0))
            except Exception:
                pass
        def _topp_cmd(*_):
            try: self._update_command()
            except Exception:
                pass
        sv_topp.trace_add("write", lambda *_: (_topp_safe(), _topp_cmd()))
        ttk.Label(samp_frame, text="Top-P").grid(row=2, column=0, sticky="w", padx=(4, 0), pady=1)
        ttk.Spinbox(samp_frame, from_=0.05, to=1.0, increment=0.05, width=8,
                    textvariable=sv_topp).grid(row=2, column=0, sticky="w", padx=(100, 0), pady=1)

        # --- Repeat Penalty (row 2, col 1) ---
        sv_rp = tk.DoubleVar(value=1.1)
        def _rp_safe(*_):
            try:
                val = float(sv_rp.get()) if sv_rp.get() else 1.1
                self.config.repeat_penalty = min(max(val, 1.0), 3.0)
                sv_rp.set(min(max(val, 1.0), 3.0))
            except Exception:
                pass
        def _rp_cmd(*_):
            try: self._update_command()
            except Exception:
                pass
        sv_rp.trace_add("write", lambda *_: (_rp_safe(), _rp_cmd()))
        ttk.Label(samp_frame, text="Repeat Pen.").grid(row=2, column=1, sticky="w", padx=(40, 0), pady=1)
        ttk.Spinbox(samp_frame, from_=1.0, to=3.0, increment=0.1, width=8,
                    textvariable=sv_rp).grid(row=2, column=1, sticky="w", padx=(150, 0), pady=1)


# ---------------------------------------------------------------------------
# Event handlers & helpers — user interactions and command generation logic
# Each section method creates a LabelFrame with its widgets and packs it into *parent*
# Live command generation is triggered by Tk variable traces on every input field
# ---------------------------------------------------------------------------

    def _browse_model(self):
        """Open file dialog to select a .gguf model file."""
        path = filedialog.askopenfilename(
            title="Select GGUF Model File",
            filetypes=[("GGUF files", "*.gguf"), ("All files", "*.*")],
            initialdir=os.path.expanduser("~"),  # start in the user's home directory
        )
        if path:
            self.config.model_path = path

    def _update_command(self):
        """Rebuild the generated command and update the display."""
        cmd = self.config.generate_command()
        if hasattr(self, 'cmd_text'):
            self.cmd_text.configure(state='normal')
            self.cmd_text.delete('1.0', 'end')
            self.cmd_text.insert('1.0', cmd)
            self.cmd_text.configure(state='disabled')


    def _copy_command(self):
        """Copy current command to clipboard."""
        cmd = self.config.generate_command()
        self.root.clipboard_clear()
        self.root.clipboard_append(cmd)

    def _save_bat_command(self):
        """Prompt user to choose a folder, name the file, and save the command as a .bat file."""
        cmd = self.config.generate_command()

        # Default filename is based on the selected model path
        model_name = os.path.basename(self.config.model_path) if self.config.model_path else "llama-server"
        default_name = model_name + ".bat"

        # Dialog to pick a folder and enter the filename
        dialog = Toplevel(self.root)
        dialog.title("Save as .bat")
        dialog.transient(self.root)
        dialog.grab_set()

        # Center the dialog over the main window
        dialog.update_idletasks()
        pw = dialog.winfo_parent()
        if pw:
            dialog.geometry(f"+{self.root.winfo_x() + (self.root.winfo_width() - 300) // 2}+{self.root.winfo_y() + (self.root.winfo_height() - 140) // 2}")

        sv_folder = tk.StringVar(value=self._last_folder or os.path.expanduser("~"))
        sv_filename = tk.StringVar(value=default_name)
        result = {"folder": None, "filename": None}

        def _browse_folder():
            d = filedialog.askdirectory(title="Select Folder")
            if d:
                sv_folder.set(d)
                self._last_folder = d

        def _ok():
            folder = sv_folder.get().strip()
            fname = sv_filename.get().strip()
            if not folder or not fname:
                messagebox.showwarning("Input", "Please enter both a folder and a filename.")
                return
            if not fname.lower().endswith(".bat"):
                fname += ".bat"
            result["folder"] = folder
            result["filename"] = fname
            dialog.destroy()

        def _cancel():
            result["folder"] = None
            dialog.destroy()

        body = ttk.Frame(dialog, padding=12)
        body.pack(fill="both", expand=True)

        # Folder row
        row = ttk.Frame(body)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text="Folder:").pack(side="left")
        ttk.Entry(row, textvariable=sv_folder, width=35).pack(side="left", fill="x", expand=True, padx=(4, 4))
        ttk.Button(row, text="Browse...", command=_browse_folder).pack(side="left")

        # Filename row
        row = ttk.Frame(body)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text="Filename:").pack(side="left")
        ttk.Entry(row, textvariable=sv_filename, width=35).pack(side="left", fill="x", expand=True, padx=(4, 4))

        # Buttons row
        btn_row = ttk.Frame(body)
        btn_row.pack(fill="x", pady=(8, 0))
        ttk.Button(btn_row, text="Save", command=_ok).pack(side="right", padx=(4, 0))
        ttk.Button(btn_row, text="Cancel", command=_cancel).pack(side="right")

        # Focus on the filename field
        sv_filename_entry = body.winfo_children()[-2].winfo_children()[1]
        sv_filename_entry.focus_set()
        sv_filename_entry.select_range(0, "end")

        dialog.wait_window()

        if not result["folder"] or not result["filename"]:
            return

        filepath = os.path.join(result["folder"], result["filename"])
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write("@echo off\n")
                f.write(f'"{cmd}"\n')
                f.write("pause\n")
            _save_last_folder(result["folder"])
            messagebox.showinfo("Saved", f"Saved as:\n{filepath}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not save file:\n{e}")

    def _run_in_cmd(self):
        """Copy command to clipboard and run it in a new cmd window."""
        cmd = self.config.generate_command()
        self.root.clipboard_clear()
        self.root.clipboard_append(cmd)
        try:
            subprocess.Popen(
                ["cmd.exe", "/k", cmd],
                creationflags=subprocess.DETACHED_PROCESS,
            )
        except Exception as e:
            messagebox.showerror("Error", f"Could not run command:\n{e}")



