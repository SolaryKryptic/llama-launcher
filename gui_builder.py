import os
import json
import re
import subprocess
import webbrowser
from tkinter import filedialog, messagebox, Tk, Toplevel, StringVar
import tkinter as tk
from tkinter import ttk
from hardwarescanner import scan_hardware
from optimisation_service import AVAILABLE_METHODS, DEFAULT_PERPLEXITY_FILE, OptimisationRequest, OptimisationService, resolve_perplexity_file
from optimiser_script import BENCH_PORT, kill_port
import sys as _sys2

import sys
if getattr(sys, 'frozen', False):
    _CONFIG_PATH = os.path.join(os.path.dirname(sys.executable), "llama_gui_data.json")
else:
    _CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "llama_gui_data.json")
"""load config from disk"""
def _load_config():
    try:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _save_config(data):
    """save config to disk"""
    try:
        with open(_CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass

def _load_last_folder():
    """load last used folder path from config for file dialogs"""
    return _load_config().get("last_folder", "")

def _save_last_folder(folder):
    """save last used folder path to config for file dialogs"""
    data = _load_config()
    data["last_folder"] = folder
    _save_config(data)


# data model and command generation

class FlagConfig:
    """holds mutable flag state, builds generated cli string"""

    def __init__(self, default_perplexity_file=None):
        self.model_path = ""          # set by browse dialog
        self.no_mmap = False            # disable memory mapped loading
        self.mlock = False              # lock model in ram to reduce swapping
        self.no_warmup = False          # skip warmup run
        self.ctx_size_value = 512       # context size value
        self.n_gpu_layers = -1          # -1 means auto detect by gpu driver, valid 0 to 99

        self.host = "0.0.0.0"         # bind all interfaces, allow local lan access
        self.port = 8080                # http server port
        self.cache_ram = 8000           # cache ram in gb
        self.num_threads = os.cpu_count() or 4   # default from cpu count
        self.threads_enabled = False    # include threads flag when enabled

        self.flash_attention = False    # enable flash attention
        self.fit_on = False             # enable fit on to fit model to gpu vram
        self.fitt = 1024               # -fitt N value when fit-on enabled

        self.batch_size = 2048          # batch size for kv cache, typical 1 to 8192
        self.micro_batch_size = 512     # micro batch for memory split, 1 to 8192
        self.threads = -1               # thread count, -1 means auto detect
        self.thread_batch = -1          # thread batch value, -1 means unset

        self.spec_enabled = False       # enable speculative decoding
        self.spec_type = "ngram-mod"  # spec strategy, e g ngram mod or draft mtp
        self.spec_draft_n_max = 0       # spec draft max, 0 means unset
        self.spec_draft_n_min = 0       # spec draft min, 0 means unset
        self.spec_draft_p_min = 0.0     # spec draft probability min, 0 means unset
        self.draft_model_path = ""       # draft model path for draft-mtp

        self.cache_type_k = "f16"      # kv cache type k
        self.cache_type_v = "f16"      # kv cache type v
        self.cache_type_kd = "f16"     # draft-mtp cache type k
        self.cache_type_vd = "f16"     # draft-mtp cache type v

        self.perplexity_file = resolve_perplexity_file(default_perplexity_file or DEFAULT_PERPLEXITY_FILE)
        self.ppl_threshold_percent = 3.0

        self.temperature = 0.8          # sampling temperature
        self.min_p = 0.0                # minimum p sampling value
        self.top_k = 40                 # top k sampling value
        self.presence_penalty = 0.0     # presence penalty, negative to penalise
        self.top_p = 0.95               # top p nucleus sampling
        self.repeat_penalty = 1.1       # repeat penalty, >1 to reduce repeats

    def generate_command(self):
        parts = ["llama-server.exe -lv 4"]

        model_path = str(self.model_path or "").strip()
        if model_path:
            parts.append(f' -m "{model_path}"')

        # no mmap and mlock both can be active
        if self.no_mmap:
            parts.append(" --no-mmap")
        if self.mlock:
            parts.append(" --mlock")
        if self.no_warmup:
            parts.append(" --no-warmup")

        # gpu layers only when set to non negative value, -1 means driver default
        if self.n_gpu_layers >= 0:
            parts.append(f" -ngl {self.n_gpu_layers}")

        # flash attention and fit on, only when checked
        if self.flash_attention:
            parts.append(" -fa on")
        if self.fit_on:
            parts.append(" --fit on")
            parts.append(f" -fitt {self.fitt}")

        # batch size, micro batch and threads included when set, -t skipped for -1
        parts.append(f" -b {self.batch_size}")
        parts.append(f" -ub {self.micro_batch_size}")
        if self.threads != -1:
            parts.append(f" -t {self.threads}")

        # thread batch only when non zero
        if self.thread_batch > 0:
            parts.append(f" -tb {self.thread_batch}")

        # speculative decoding only when enabled
        if self.spec_enabled:
            parts.append(f" --spec-type {self.spec_type}")
            if self.spec_draft_n_max > 0:
                parts.append(f" --spec-draft-n-max {self.spec_draft_n_max}")
            if self.spec_draft_n_min > 0:
                parts.append(f" --spec-draft-n-min {self.spec_draft_n_min}")
            if self.spec_draft_p_min > 0:
                parts.append(f" --spec-draft-p-min {float(self.spec_draft_p_min):.1f}")
            draft_path = str(self.draft_model_path or "").strip()
            if draft_path:
                parts.append(f' --model-draft "{draft_path}"')

        # cache types always included
        parts.append(f" -ctk {self.cache_type_k}")
        parts.append(f" -ctv {self.cache_type_v}")
        if "draft-mtp" in (self.spec_type or ""):
            parts.append(f" -ctkd {self.cache_type_kd}")
            parts.append(f" -ctvd {self.cache_type_vd}")

        # add context size value
        ctx_val = max(2, min(int(str(self.ctx_size_value)), 999999999))
        parts.append(f" -c {ctx_val}")

        # server settings always included
        host = str(self.host).strip() or "0.0.0.0"
        port = max(1, min(int(str(self.port)), 65535))
        parts.append(f" --host {host}")
        parts.append(f" --port {port}")
        parts.append(f" --cache-ram {self.cache_ram}"),

        # sampling params always included
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
        parts.append(f" --top-p {topp:.2f}")
        parts.append(f" --repeat-penalty {rp:.2f}")

        return "".join(parts)

    @staticmethod
    def _safe_int(name, default):
        """safely convert value to int, return default on failure"""
        try:
            val = str(default) if isinstance(default, str) else default
            return int(val)
        except (ValueError, TypeError):
            return default

    def to_dict(self):
        """return dict of mutable flag state for persistence"""
        return {
            "model_path": self.model_path,
            "no_mmap": self.no_mmap,
            "mlock": self.mlock,
            "no_warmup": self.no_warmup,
            "ctx_size_value": self.ctx_size_value,
            "n_gpu_layers": self.n_gpu_layers,
            "host": self.host,
            "port": self.port,
            "cache_ram": self.cache_ram,
            "num_threads": self.num_threads,
            "threads_enabled": self.threads_enabled,
            "flash_attention": self.flash_attention,
            "fit_on": self.fit_on,
            "fitt": self.fitt,
            "spec_enabled": self.spec_enabled,
            "spec_type": self.spec_type,
            "spec_draft_n_max": self.spec_draft_n_max,
            "spec_draft_n_min": self.spec_draft_n_min,
            "spec_draft_p_min": self.spec_draft_p_min,
            "draft_model_path": self.draft_model_path,
            "batch_size": self.batch_size,
            "micro_batch_size": self.micro_batch_size,
            "threads": self.threads,
            "thread_batch": self.thread_batch,
            "cache_type_k": self.cache_type_k,
            "cache_type_v": self.cache_type_v,
            "cache_type_kd": self.cache_type_kd,
            "cache_type_vd": self.cache_type_vd,
            "perplexity_file": self.perplexity_file,
            "ppl_threshold_percent": self.ppl_threshold_percent,
            "temperature": self.temperature,
            "min_p": self.min_p,
            "top_k": self.top_k,
            "presence_penalty": self.presence_penalty,
            "top_p": self.top_p,
            "repeat_penalty": self.repeat_penalty,
        }

    def from_dict(self, d):
        """restore mutable flag state from saved dict"""
        string_defaults = {
            "model_path": "",
            "host": "0.0.0.0",
            "spec_type": "",
            "draft_model_path": "",
            "cache_type_k": "f16",
            "cache_type_v": "f16",
            "cache_type_kd": "f16",
            "cache_type_vd": "f16",
            "perplexity_file": "",
        }
        int_defaults = {
            "ctx_size_value": 512,
            "n_gpu_layers": -1,
            "cache_ram": 8000,
            "num_threads": os.cpu_count() or 4,
            "fitt": 1024,
            "spec_draft_n_max": 0,
            "spec_draft_n_min": 0,
            "batch_size": 2048,
            "micro_batch_size": 512,
            "threads": -1,
            "thread_batch": -1,
            "port": 8080,
            "top_k": 40,
        }
        float_defaults = {
            "spec_draft_p_min": 0.0,
            "temperature": 0.8,
            "min_p": 0.0,
            "presence_penalty": 0.0,
            "top_p": 0.95,
            "repeat_penalty": 1.1,
            "ppl_threshold_percent": 3.0,
        }
        bool_defaults = {
            "no_mmap": False,
            "mlock": False,
            "no_warmup": False,
            "threads_enabled": False,
            "flash_attention": False,
            "fit_on": False,
            "spec_enabled": False,
        }
        for key, val in d.items():
            if not hasattr(self, key):
                continue
            if val is None:
                if key in string_defaults:
                    val = string_defaults[key]
                elif key in int_defaults:
                    val = int_defaults[key]
                elif key in float_defaults:
                    val = float_defaults[key]
                elif key in bool_defaults:
                    val = bool_defaults[key]
            elif key in string_defaults:
                val = str(val or string_defaults[key])
            elif key in int_defaults:
                try:
                    val = int(val)
                except (ValueError, TypeError):
                    val = int_defaults[key]
            elif key in float_defaults:
                try:
                    val = float(val)
                except (ValueError, TypeError):
                    val = float_defaults[key]
            elif key in bool_defaults:
                if isinstance(val, str):
                    val = val.strip().lower() in ("1", "true", "yes", "on")
                else:
                    val = bool(val)
            setattr(self, key, val)


# UI builder —> all ttk widget frames returned as methods
# Each section method creates a LabelFrame with its widgets and packs it into *parent*
# Live command generation is triggered by Tk variable traces on every input field

class LlamaServerGUI:
    """main tkinter gui for configuring llama server flags, generating commands"""

    def __init__(self, root, default_perplexity_file=None):
        self.root = root
        self.config = FlagConfig(default_perplexity_file)
        self._last_folder = _load_last_folder()
        self._last_window_geometries = {}
        self._geometry_save_jobs = {}

        # Save config on window close
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Create all Tk variables (StringVar / IntVar) so every widget change triggers live updates
        sv_model_path = tk.StringVar(value="")                    # model path display var
        iv_auto_gpu = tk.IntVar(value=-1)                         # n-gpu-layers (-1=auto, 0-99)
        lv_bool_no_mmap = tk.BooleanVar(value=False)              # no-mmap toggle state variable (boolean)
        ml_bool_mlock = tk.BooleanVar(value=False)                # mlock toggle state variable (boolean)
        nw_bool_no_warmup = tk.BooleanVar(value=False)            # no-warmup toggle state variable (boolean)

        iv_ctx_enabled = tk.BooleanVar(value=False)               # toggle to show ctx-size input

        # Flash Attention / Fit On checkboxes
        iv_flash_attn = tk.BooleanVar(value=False)
        iv_fit_on = tk.BooleanVar(value=False)
        iv_fitt = tk.IntVar(value=1024)

        # Speculative decoding toggle
        iv_spec_enabled = tk.BooleanVar(value=False)

        # Batch size, micro batch, threads spinboxes
        iv_batch_size = tk.IntVar(value=2048)
        iv_micro_batch = tk.IntVar(value=512)
        iv_threads_val_new = tk.IntVar(value=-1)
        iv_thread_batch = tk.IntVar(value=-1)

        # Cache type K and V dropdowns
        CACHE_TYPES = ["f16", "f32", "q8_0", "q5_0", "q4_0"]
        sv_cache_k = tk.StringVar(value="f16")
        sv_cache_v = tk.StringVar(value="f16")
        sv_cache_kd = tk.StringVar(value="f16")
        sv_cache_vd = tk.StringVar(value="f16")

        sv_host = tk.StringVar(value="0.0.0.0")                   # server host binding address
        iv_port = tk.IntVar(value=8080)                           # HTTP port (1-65535)
        iv_cache_ram = tk.IntVar(value=8000)                      # cache RAM in GB (--cache-ram)

        iv_threads_enabled = tk.BooleanVar(value=False)           # toggle to show threads input
        iv_threads_val = tk.IntVar(value=os.cpu_count() or 4)     # thread count default from CPU cores

        sv_temp = tk.DoubleVar(value=0.8)                         # temperature (float 0.05–2.0)
        sv_topk = tk.IntVar(value=40)                              # top-k integer value >= 1
        sv_topp = tk.DoubleVar(value=0.95)                        # top-p float between [0.05, 1.0]
        sv_rp = tk.DoubleVar(value=1.1)                           # repeat penalty (float > 1.0)
        sv_minp = tk.DoubleVar(value=0.0)                         # min-p (float -1.0–1.0)
        sv_pp = tk.DoubleVar(value=0.0)                           # presence penalty (float -2.0–2.0)

        self._vars = {
            "model_path": sv_model_path,
            "n_gpu_layers": iv_auto_gpu,
            "no_mmap": lv_bool_no_mmap,
            "mlock": ml_bool_mlock,
            "no_warmup": nw_bool_no_warmup,
            "ctx_size_enabled": iv_ctx_enabled,
            "flash_attention": iv_flash_attn,
            "fit_on": iv_fit_on,
            "fitt": iv_fitt,
            "spec_enabled": iv_spec_enabled,
            "batch_size": iv_batch_size,
            "micro_batch": iv_micro_batch,
            "threads_val": iv_threads_val_new,
            "thread_batch": iv_thread_batch,
            "cache_type_k": sv_cache_k,
            "cache_type_v": sv_cache_v,
            "cache_type_kd": sv_cache_kd,
            "cache_type_vd": sv_cache_vd,
            "host": sv_host,
            "port": iv_port,
            "cache_ram": iv_cache_ram,
            "threads_enabled": iv_threads_enabled,
            "num_threads": iv_threads_val,
            "temperature": sv_temp,
            "top_k": sv_topk,
            "top_p": sv_topp,
            "repeat_penalty": sv_rp,
            "min_p": sv_minp,
            "presence_penalty": sv_pp,
        }

        # Store all Tk variables on self for cross-method access
        self._tk = {
            "model_path": sv_model_path,
            "n_gpu_layers": iv_auto_gpu,
            "no_mmap": lv_bool_no_mmap,
            "mlock": ml_bool_mlock,
            "no_warmup": nw_bool_no_warmup,
            "ctx_size_enabled": iv_ctx_enabled,
            "flash_attention": iv_flash_attn,
            "fit_on": iv_fit_on,
            "fitt": iv_fitt,
            "spec_enabled": iv_spec_enabled,
            "batch_size": iv_batch_size,
            "micro_batch": iv_micro_batch,
            "threads_val": iv_threads_val_new,
            "thread_batch": iv_thread_batch,
            "cache_type_k": sv_cache_k,
            "cache_type_v": sv_cache_v,
            "cache_type_kd": sv_cache_kd,
            "cache_type_vd": sv_cache_vd,
            "host": sv_host,
            "port": iv_port,
            "cache_ram": iv_cache_ram,
            "threads_enabled": iv_threads_enabled,
            "num_threads": iv_threads_val,
            "temperature": sv_temp,
            "top_k": sv_topk,
            "top_p": sv_topp,
            "repeat_penalty": sv_rp,
            "min_p": sv_minp,
            "presence_penalty": sv_pp,
        }

        # Load saved config
        saved = _load_config()
        saved_flags = saved.pop("flags", {})
        if saved_flags:
            import sys as _sys2
            global _sys2
            self.config.from_dict(saved_flags)
        self._restore_vars(saved_flags)

        # Scan hardware info for display (read-only, not saved to config)
        try:
            self._hw = scan_hardware()
        except Exception:
            self._hw = {"CPU": "Unknown", "GPU": "Unknown", "VRAM": "Unknown", "RAM": "Unknown", "CPU_CORES": 0, "CPU_THREADS": 0}

        # Change handlers — update config state and trigger command rebuild
        def _on_no_mmap_change(*_):
            try:
                self.config.no_mmap = bool(lv_bool_no_mmap.get())
            except Exception:
                pass

        def _on_mlock_change(*_):
            try:
                self.config.mlock = bool(ml_bool_mlock.get())
            except Exception:
                pass

        def _on_no_warmup_change(*_):
            try:
                self.config.no_warmup = bool(nw_bool_no_warmup.get())
            except Exception:
                pass

        def _on_model_change(*_):
            self.config.model_path = sv_model_path.get()

        def _on_gpu_layers_change(*_):
            try:
                raw = iv_auto_gpu.get()
                val = int(raw) if raw else -1
                if not (-1 <= val <= 99): return
                self.config.n_gpu_layers = val
            except (ValueError, TypeError, tk.TclError):
                pass

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

        def _on_fitt_change(*_):
            try:
                raw = iv_fitt.get()
                self.config.fitt = max(1, min(raw, 65536))
            except (ValueError, TypeError, tk.TclError):
                pass

        def _on_batch_size_change(*_):
            try:
                raw = iv_batch_size.get()
                val = int(raw) if raw else 2048
                self.config.batch_size = max(1, min(val, 8192))
            except (ValueError, TypeError, tk.TclError):
                pass

        def _on_micro_batch_change(*_):
            try:
                raw = iv_micro_batch.get()
                val = int(raw) if raw else 512
                self.config.micro_batch_size = max(1, min(val, 8192))
            except (ValueError, TypeError, tk.TclError):
                pass

        def _on_threads_new_change(*_):
            try:
                raw = iv_threads_val_new.get()
                val = int(raw) if raw else -1
                self.config.threads = max(-1, min(val, 128))
            except (ValueError, TypeError, tk.TclError):
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

        def _on_cache_kd_change(*_):
            try:
                val = sv_cache_kd.get()
                if val in CACHE_TYPES:
                    self.config.cache_type_kd = val
            except Exception:
                pass

        def _on_cache_vd_change(*_):
            try:
                val = sv_cache_vd.get()
                if val in CACHE_TYPES:
                    self.config.cache_type_vd = val
            except Exception:
                pass

        def _on_host_change(*_):
            self.config.host = sv_host.get() or "0.0.0.0"

        def _on_port_change(*_):
            try:
                raw = iv_port.get()
                val = int(raw) if raw else 8080
                self.config.port = max(1, min(val, 65535))
            except (ValueError, TypeError, tk.TclError):
                pass
            self._update_command()

        def _on_threads_enabled_change(*_):
            self.config.threads_enabled = bool(iv_threads_enabled.get())

        def _on_num_threads_change(*_):
            try:
                raw = iv_threads_val.get()
                val = int(raw) if raw else os.cpu_count() or 4
                if not (1 <= val <= 256): return
                self.config.num_threads = max(1, min(val, 256))
            except (ValueError, TypeError, tk.TclError):
                pass





        # Register all traces on the Tk variables so every change triggers live command update
        lv_bool_no_mmap.trace_add("write", lambda *_: (_on_no_mmap_change(), self._update_command()))
        ml_bool_mlock.trace_add("write", lambda *_: (_on_mlock_change(), self._update_command()))
        nw_bool_no_warmup.trace_add("write", lambda *_: (_on_no_warmup_change(), self._update_command()))
        iv_auto_gpu.trace_add("write", lambda *_: (_on_gpu_layers_change(), self._update_command()))

        # Context size toggle and input


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
                raw = iv_port.get()
                val = int(raw) if raw else 8080
                self.config.port = max(1, min(val, 65535))
                self._update_command()
            except (ValueError, TypeError, tk.TclError):
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
        iv_fitt.trace_add("write", lambda *_: (_on_fitt_change(), self._update_command()))

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
        sv_cache_kd.trace_add("write", lambda *_: (_on_cache_kd_change(), self._update_command()))
        sv_cache_vd.trace_add("write", lambda *_: (_on_cache_vd_change(), self._update_command()))

        # Build the full UI
        self._build_ui()


# ---------------------------------------------------------------------------
# Section builders, each returns a ttk.Frame packed into *parent*
# Each section method creates a LabelFrame with its widgets and packs it into *parent*
# Live command generation is triggered by Tk variable traces on every input field
# ---------------------------------------------------------------------------

    def _on_close(self):
        """save config flags, folder, window geometry before closing"""
        try:
            data = _load_config()
            data["flags"] = self.config.to_dict()
            data["last_folder"] = self._last_folder
            try:
                data["window_geometry"] = self.root.geometry()
                data["window_state"] = self.root.state()
            except Exception:
                pass
            _save_config(data)
        except Exception:
            pass
        self.root.destroy()

    def _restore_window_geometry(self, win, config_key, default_width, default_height):
        """Restore Toplevel geometry from saved config or center default window."""
        config_data = _load_config()
        saved_geometry = config_data.get(config_key)
        if isinstance(saved_geometry, str) and saved_geometry:
            try:
                win.withdraw()
                win.geometry(saved_geometry)
                win.update_idletasks()
                win.deiconify()
                win.lift()
                return
            except Exception:
                pass

        win.geometry(f"{default_width}x{default_height}")
        win.update_idletasks()
        screen_w = win.winfo_screenwidth()
        screen_h = win.winfo_screenheight()
        x = max(0, (screen_w - default_width) // 2)
        y = max(0, (screen_h - default_height) // 2)
        win.geometry(f"{default_width}x{default_height}+{x}+{y}")

    def _ensure_min_geometry(self, win, min_width, min_height):
        """Force a Toplevel to at least min_width x min_height."""
        geom = win.geometry()
        match = re.match(r"(\d+)x(\d+)", geom)
        if not match:
            return
        width = int(match.group(1))
        height = int(match.group(2))
        if width >= min_width and height >= min_height:
            return
        win.geometry(f"{min_width}x{min_height}")
        win.update_idletasks()
        screen_w = win.winfo_screenwidth()
        screen_h = win.winfo_screenheight()
        x = max(0, (screen_w - min_width) // 2)
        y = max(0, (screen_h - min_height) // 2)
        win.geometry(f"{min_width}x{min_height}+{x}+{y}")

    def _normalise_window_geometry(self, win):
        try:
            if win.winfo_exists():
                geom = f"{win.winfo_width()}x{win.winfo_height()}+{win.winfo_x()}+{win.winfo_y()}"
            else:
                geom = self._last_window_geometries.get(f"__current__:{id(win)}")
            if not geom or geom.startswith("0x0"):
                return None
            return geom
        except Exception:
            return None

    def _save_window_geometry(self, config_key, win):
        """Save current Toplevel geometry to config."""
        try:
            job_id = getattr(self, "_geometry_save_jobs", {}).get(config_key)
            if job_id is not None:
                try:
                    self.root.after_cancel(job_id)
                except Exception:
                    pass
                try:
                    self._geometry_save_jobs.pop(config_key, None)
                except Exception:
                    pass
            geom = self._normalise_window_geometry(win)
            if geom is None:
                return
            self._last_window_geometries[config_key] = geom
            data = _load_config()
            data[config_key] = geom
            _save_config(data)
        except Exception:
            pass

    def _queue_window_geometry_save(self, config_key, win, delay_ms=250):
        if not hasattr(self, "_geometry_save_jobs"):
            self._geometry_save_jobs = {}
        geom = self._normalise_window_geometry(win)
        if geom is not None:
            self._last_window_geometries[f"__current__:{id(win)}"] = geom
            self._last_window_geometries[config_key] = geom
        previous = self._geometry_save_jobs.get(config_key)
        if previous is not None:
            try:
                self.root.after_cancel(previous)
            except Exception:
                pass
        def _save_later():
            self._geometry_save_jobs.pop(config_key, None)
            self._save_window_geometry(config_key, win)
        self._geometry_save_jobs[config_key] = self.root.after(delay_ms, _save_later)

    def _bind_window_geometry_persistence(self, config_key, win):
        def _on_configure(event):
            if event.width > 1 and event.height > 1:
                self._queue_window_geometry_save(config_key, win)
        win.bind("<Configure>", _on_configure)
        win.bind("<Destroy>", lambda *_: self._save_window_geometry(config_key, win))

    def _restore_vars(self, saved_flags):
        """set tk variable values to match saved flag state"""
        tk = self._tk
        if "model_path" in saved_flags:
            tk["model_path"].set(str(saved_flags.get("model_path") or ""))
        if "n_gpu_layers" in saved_flags:
            try:
                tk["n_gpu_layers"].set(int(saved_flags["n_gpu_layers"]))
            except (ValueError, TypeError):
                pass

        if "no_mmap" in saved_flags:
            try:
                tk["no_mmap"].set(bool(saved_flags["no_mmap"]))
            except (ValueError, TypeError):
                pass
        if "mlock" in saved_flags:
            try:
                tk["mlock"].set(bool(saved_flags["mlock"]))
            except (ValueError, TypeError):
                pass
        if "no_warmup" in saved_flags:
            try:
                tk["no_warmup"].set(bool(saved_flags["no_warmup"]))
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
        if "fitt" in saved_flags:
            try:
                tk["fitt"].set(int(saved_flags["fitt"]))
            except (ValueError, TypeError):
                pass
        if "spec_enabled" in saved_flags:
            try:
                tk["spec_enabled"].set(bool(saved_flags["spec_enabled"]))
            except (ValueError, TypeError):
                pass
        if "spec_type" in saved_flags:
            try:
                self.config.spec_type = str(saved_flags.get("spec_type") or "")
            except (ValueError, TypeError):
                pass
        if "spec_draft_n_max" in saved_flags:
            try:
                self.config.spec_draft_n_max = int(saved_flags["spec_draft_n_max"])
            except (ValueError, TypeError):
                pass
        if "spec_draft_n_min" in saved_flags:
            try:
                self.config.spec_draft_n_min = int(saved_flags["spec_draft_n_min"])
            except (ValueError, TypeError):
                pass
        if "batch_size" in saved_flags:
            try:
                tk["batch_size"].set(int(saved_flags["batch_size"]))
            except (ValueError, TypeError):
                pass
        if "micro_batch_size" in saved_flags:
            try:
                tk["micro_batch"].set(int(saved_flags["micro_batch_size"]))
            except (ValueError, TypeError):
                pass
        if "threads" in saved_flags:
            try:
                tk["threads_val"].set(int(saved_flags["threads"]))
            except (ValueError, TypeError):
                pass
        if "thread_batch" in saved_flags:
            try:
                tk["thread_batch"].set(int(saved_flags["thread_batch"]))
            except (ValueError, TypeError):
                pass
        if "cache_type_k" in saved_flags:
            try:
                tk["cache_type_k"].set(str(saved_flags.get("cache_type_k") or "f16"))
            except (ValueError, TypeError):
                pass
        if "cache_type_v" in saved_flags:
            try:
                tk["cache_type_v"].set(str(saved_flags.get("cache_type_v") or "f16"))
            except (ValueError, TypeError):
                pass
        if "cache_type_kd" in saved_flags:
            try:
                tk["cache_type_kd"].set(str(saved_flags.get("cache_type_kd") or "f16"))
            except (ValueError, TypeError):
                pass
        if "cache_type_vd" in saved_flags:
            try:
                tk["cache_type_vd"].set(str(saved_flags.get("cache_type_vd") or "f16"))
            except (ValueError, TypeError):
                pass
        if "perplexity_file" in saved_flags:
            self.config.perplexity_file = resolve_perplexity_file(str(saved_flags.get("perplexity_file") or DEFAULT_PERPLEXITY_FILE))
        if "ppl_threshold_percent" in saved_flags:
            try:
                self.config.ppl_threshold_percent = float(saved_flags["ppl_threshold_percent"])
            except (ValueError, TypeError):
                pass
        if "host" in saved_flags:
            try:
                tk["host"].set(str(saved_flags.get("host") or "0.0.0.0"))
            except (ValueError, TypeError):
                pass
        if "port" in saved_flags:
            try:
                tk["port"].set(int(saved_flags["port"]))
            except (ValueError, TypeError):
                pass
        if "cache_ram" in saved_flags:
            try:
                tk["cache_ram"].set(int(saved_flags["cache_ram"]))
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
        if "min_p" in saved_flags:
            try:
                tk["min_p"].set(float(saved_flags["min_p"]))
            except (ValueError, TypeError):
                pass
        if "presence_penalty" in saved_flags:
            try:
                tk["presence_penalty"].set(float(saved_flags["presence_penalty"]))
            except (ValueError, TypeError):
                pass
        if "repeat_penalty" in saved_flags:
            try:
                tk["repeat_penalty"].set(float(saved_flags["repeat_penalty"]))
            except (ValueError, TypeError):
                pass

    def _build_ui(self):
        """construct all sections and pack into a scrollable canvas"""
        root = self.root

        # Use grid so the window layout expands naturally
        root.grid_rowconfigure(0, weight=1)
        root.grid_columnconfigure(0, weight=1)

        outer_frame = ttk.Frame(root, padding=(6, 4))
        outer_frame.grid(row=0, column=0, sticky="nsew")
        # Ensure the middle (scrollable) row grows but leave space for bottom buttons
        outer_frame.grid_rowconfigure(1, weight=1)
        outer_frame.grid_rowconfigure(2, weight=0)

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
            """update scroll region when section frames change size"""
            bbox = self.canvas.bbox("all")
            if bbox:
                self.canvas.configure(scrollregion=(*bbox[:4],))

        def _update_content_width(event):
            """resize scrollable frame to match parent canvas width"""
            w = event.width or root.winfo_reqwidth()
            self.canvas.itemconfigure(scroll_content_id, width=w)
            bbox = self.canvas.bbox("all")
            if bbox:
                self.canvas.configure(scrollregion=(*bbox[:4],))

        def _on_scroll(event):
            """handle scroll and resize"""
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
            # Only scroll canvas if there's more to show; otherwise let child widgets handle it
            bbox = self.canvas.bbox("all")
            if not bbox:
                pass
            can_scroll_up   = self.canvas.yview()[0] > 0
            can_scroll_down = self.canvas.yview()[1] < 1
            if (event.delta > 0 and can_scroll_up) or (event.delta <= 0 and can_scroll_down):
                direction = -1 if event.delta > 0 else 1
                self.canvas.yview_scroll(int(1.5 * abs(direction)), "units" if direction < 0 else "pages")
                return "break"

        root.bind_all("<MouseWheel>", lambda e: _on_mousewheel(e))

        # Build each section into the scrollable frame in order
        self._section_hardware_info(self.scrollable)
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
        """model loading section, browse button, no mmap, mlock toggles"""
        frame = ttk.LabelFrame(parent, text="Model Loading", padding=(8, 6))
        frame.pack(fill="both", padx=6, pady=4)

        # Browse row with a file dialog so the user can pick a .gguf model
        btn_row = ttk.Frame(frame)
        btn_row.pack(fill="x")

        browse_btn = ttk.Button(btn_row, text="Browse local files...", command=self._browse_model)
        browse_btn.pack(side="left", padx=(0, 2))

        find_btn = ttk.Button(btn_row, text="Find a model online...", command=self._open_huggingface)
        find_btn.pack(side="left", padx=(0, 6))

        # Read-only label shows the selected model path and updates automatically
        self.model_path_label = tk.Label(
            btn_row, text="(no model selected)", anchor="w", justify="left"
        )
        self.model_path_label.pack(side="left", fill="x", expand=True)

        # Bind trace so the model path label updates when the path changes
        sv_model_path = self._tk["model_path"]
        def _update_model_label(*_):
            full_path = sv_model_path.get()
            if not full_path:
                display = "(no model selected)"
            else:
                name = os.path.basename(full_path)
                display = "Currently selected: " + (name.rsplit(".gguf", 1)[0] if name.lower().endswith(".gguf") else name)
            self.model_path_label.config(text=display)
            self.config.model_path = full_path

        sv_model_path.trace_add("write", lambda *_: (_update_model_label(), self._update_command()))
        _update_model_label()  # initialise label on startup

        # Place No-MMAP and MLock checkboxes side by side
        check_row = ttk.Frame(frame)
        check_row.pack(fill="x", pady=(4, 0))

        lv_bool_no_mmap = self._tk["no_mmap"]
        ml_bool_mlock = self._tk["mlock"]
        nw_bool_no_warmup = self._tk["no_warmup"]

        tk.Checkbutton(check_row, text="No-MMAP", variable=lv_bool_no_mmap).pack(side="left")
        tk.Checkbutton(check_row, text="MLock", variable=ml_bool_mlock).pack(side="left", padx=(12, 0))
        tk.Checkbutton(check_row, text="No Warmup", variable=nw_bool_no_warmup).pack(side="left", padx=(12, 0))


    def _section_context_gpu(self, parent):
        """context size and gpu layers section, inline on the same row"""
        frame = ttk.LabelFrame(parent, text="Performance", padding=(8, 6))
        frame.pack(fill="both", padx=6, pady=4)
        frame.grid_columnconfigure(0, weight=1)

        ctx_frame = ttk.Frame(frame)
        ctx_frame.grid(row=0, column=0, sticky="w")

        iv_ctx_var = tk.IntVar(value=self.config.ctx_size_value)
        self._tk["ctx_size_value"] = iv_ctx_var

        def _on_ctx_val(*_):
            try:
                raw = iv_ctx_var.get()
                val = max(2, min(int(raw), 999999999)) if raw else 512
                iv_ctx_var.set(val)
                self.config.ctx_size_value = val
            except (ValueError, TypeError, tk.TclError):
                pass

        ttk.Label(ctx_frame, text="Context").pack(side="left")
        ttk.Entry(ctx_frame, textvariable=iv_ctx_var, width=8).pack(side="left", padx=(4, 0))

        def _ctx_value_trace(*_):
            try:
                raw = iv_ctx_var.get()
                val = max(2, min(int(raw), 999999999)) if raw else 512
                self.config.ctx_size_value = val
            except (ValueError, TypeError, tk.TclError):
                pass

        iv_ctx_var.trace_add("write", lambda *_: (_ctx_value_trace(), self._update_command()))

        iv_auto_gpu = self._tk["n_gpu_layers"]

        ttk.Label(ctx_frame, text="GPU Layers").pack(side="left", padx=(16, 0))

        # Spinbox for GPU layers from -1 to 99
        spinvar = tk.IntVar(value=self.config.n_gpu_layers)

        def _on_spinval(*_):
            try:
                val = int(spinvar.get())
                if not (-1 <= val <= 99): return
                self.config.n_gpu_layers = max(-1, min(val, 99))
            except (ValueError, TypeError, tk.TclError):
                pass

        def _gpu_trace_wrapper(*_):
            try:
                raw = iv_auto_gpu.get()
                v = int(raw) if raw else -1
                spinvar.set(v)
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
            except (ValueError, TypeError, tk.TclError):
                pass

        spinvar.trace_add("write", lambda *_: (_spinvar_safe(), self._update_command()))

        spinbox_gpu = ttk.Spinbox(ctx_frame, from_=-1, to=99, textvariable=spinvar, width=5)
        spinbox_gpu.pack(side="left")

        def _gpu_wheel(event):
            try:
                val = int(spinvar.get())
            except (ValueError, TypeError):
                return "break"
            delta = 1 if event.delta > 0 else -1
            new_val = max(-1, min(99, val + delta))
            spinvar.set(new_val)
            return "break"

        spinbox_gpu.bind("<MouseWheel>", _gpu_wheel)

        # Batch size, micro batch, and thread handlers local to this section
        def _on_batch_size_change(*_):
            try:
                raw = iv_batch_size.get()
                val = int(raw) if raw else 2048
                self.config.batch_size = max(1, min(val, 8192))
            except (ValueError, TypeError, tk.TclError):
                pass

        def _on_micro_batch_change(*_):
            try:
                raw = iv_micro_batch.get()
                val = int(raw) if raw else 512
                self.config.micro_batch_size = max(1, min(val, 8192))
            except (ValueError, TypeError, tk.TclError):
                pass

        def _on_threads_new_change(*_):
            try:
                raw = iv_threads_val_new.get()
                val = int(raw) if raw else -1
                self.config.threads = max(-1, min(val, 128))
            except (ValueError, TypeError, tk.TclError):
                pass

        def _on_thread_batch_change(*_):
            try:
                raw = iv_thread_batch.get()
                val = int(raw) if raw else -1
                if not (-1 <= val <= 512): return
                self.config.thread_batch = max(-1, min(val, 512))
            except (ValueError, TypeError, tk.TclError):
                pass

        # --- Flash Attention and Fit On checkboxes (row 1, full width) ---
        fa_row = ttk.Frame(frame)
        fa_row.grid(row=1, column=0, columnspan=2, sticky="w", pady=(6, 0))

        iv_flash_attn = self._tk["flash_attention"]
        iv_fit_on = self._tk["fit_on"]
        iv_fitt = self._tk["fitt"]
        tk.Checkbutton(fa_row, text="Flash Attention (-fa)", variable=iv_flash_attn).pack(side="left")
        tk.Checkbutton(fa_row, text="Fit On (--fit-on)", variable=iv_fit_on).pack(side="left", padx=(16, 0))
        ttk.Label(fa_row, text="Fit Target").pack(side="left", padx=(8, 0))
        ttk.Spinbox(fa_row, from_=1, to=65536, textvariable=iv_fitt, width=5).pack(side="left", padx=(2, 0))

        # --- Speculative Decoding (row 2, full width) ---
        spec_frame = ttk.Frame(frame)
        spec_frame.grid(row=2, column=0, columnspan=2, sticky="w", pady=(4, 0))

        iv_spec_enabled = self._tk["spec_enabled"]
        iv_spec_ngram = tk.BooleanVar(value=("ngram-mod" in self.config.spec_type))
        iv_spec_draft = tk.BooleanVar(value=("draft-mtp" in self.config.spec_type))
        iv_spec_draft_max = tk.IntVar(value=self.config.spec_draft_n_max)
        iv_spec_draft_min = tk.IntVar(value=self.config.spec_draft_n_min)
        iv_spec_draft_p_min = tk.DoubleVar(value=self.config.spec_draft_p_min)
        self._tk["spec_ngram"] = iv_spec_ngram
        self._tk["spec_draft"] = iv_spec_draft
        self._tk["spec_draft_max"] = iv_spec_draft_max
        self._tk["spec_draft_min"] = iv_spec_draft_min
        self._tk["spec_draft_p_min"] = iv_spec_draft_p_min

        spec_row = ttk.Frame(spec_frame)
        spec_row.pack(fill="x")
        tk.Checkbutton(spec_row, text="Speculative Decoding", variable=iv_spec_enabled).pack(side="left")

        # Sub-options row (child of spec_frame same pack master)
        spec_sub_row = ttk.Frame(spec_frame)
        self._spec_sub_row = spec_sub_row
        tk.Checkbutton(spec_sub_row, text="ngram-mod", variable=iv_spec_ngram).grid(row=0, column=0, sticky="w")
        tk.Checkbutton(spec_sub_row, text="draft-mtp", variable=iv_spec_draft).grid(row=0, column=1, sticky="w", padx=(16, 0))

        # Draft model inline block: browse button + label + max/min/p-min spinboxes
        spec_draft_row = ttk.Frame(spec_sub_row)
        self._spec_draft_row = spec_draft_row
        spec_draft_row.grid(row=1, column=0, columnspan=8, sticky="w", pady=(2, 0))
        ttk.Button(spec_draft_row, text="Browse draft models...", command=self._browse_draft_model).pack(side="left", padx=(0, 4))
        self.draft_model_label = tk.Label(spec_draft_row, text="", anchor="w", justify="left", foreground="#666")
        self.draft_model_label.pack(side="left", padx=(0, 8))
        ttk.Label(spec_draft_row, text="Max:").pack(side="left")
        ttk.Spinbox(spec_draft_row, from_=0, to=64, textvariable=iv_spec_draft_max, width=4).pack(side="left", padx=(2, 8))
        ttk.Label(spec_draft_row, text="Min:").pack(side="left")
        ttk.Spinbox(spec_draft_row, from_=0, to=64, textvariable=iv_spec_draft_min, width=4).pack(side="left", padx=(2, 8))
        ttk.Label(spec_draft_row, text="P Min:").pack(side="left")
        ttk.Spinbox(spec_draft_row, from_=0.0, to=0.7, increment=0.1, textvariable=iv_spec_draft_p_min, width=4, format="%.1f").pack(side="left", padx=(2, 0))

        def _update_spec_type():
            """build spec_type string from both checkboxes"""
            types = []
            if iv_spec_ngram.get():
                types.append("ngram-mod")
            if iv_spec_draft.get():
                types.append("draft-mtp")
            self.config.spec_type = ",".join(types) if types else "ngram-mod"

        def _on_spec_toggle(*_):
            try:
                enabled = iv_spec_enabled.get()
                self.config.spec_enabled = enabled
                if enabled:
                    spec_sub_row.pack()
                else:
                    spec_sub_row.pack_forget()
                    iv_spec_ngram.set(False)
                    iv_spec_draft.set(False)
                _update_spec_type()
                self._update_command()
            except Exception:
                pass

        def _on_spec_sub(*_):
            try:
                _update_spec_type()
                if iv_spec_draft.get():
                    spec_draft_row.grid()
                else:
                    spec_draft_row.grid_remove()
                    self.config.draft_model_path = ""
                    self.config.spec_draft_n_max = 0
                    self.config.spec_draft_n_min = 0
                    self.config.spec_draft_p_min = 0.0
                    self.draft_model_label.config(text="")
                    iv_spec_draft_max.set(0)
                    iv_spec_draft_min.set(0)
                    iv_spec_draft_p_min.set(0.0)
                self._update_command()
            except Exception:
                pass

        def _on_spec_draft_spin(*_):
            try:
                self.config.spec_draft_n_max = max(0, iv_spec_draft_max.get())
                self.config.spec_draft_n_min = max(0, iv_spec_draft_min.get())
                raw_p_min = iv_spec_draft_p_min.get()
                p_min = float(raw_p_min) if raw_p_min else 0.0
                self.config.spec_draft_p_min = max(0.0, min(p_min, 0.7))
                iv_spec_draft_p_min.set(self.config.spec_draft_p_min)
                self._update_command()
            except (ValueError, TypeError, tk.TclError):
                pass

        iv_spec_enabled.trace_add("write", lambda *_: _on_spec_toggle())
        iv_spec_ngram.trace_add("write", lambda *_: _on_spec_sub())
        iv_spec_draft.trace_add("write", lambda *_: _on_spec_sub())
        iv_spec_draft_max.trace_add("write", lambda *_: (_on_spec_draft_spin(),))
        iv_spec_draft_min.trace_add("write", lambda *_: (_on_spec_draft_spin(),))
        iv_spec_draft_p_min.trace_add("write", lambda *_: (_on_spec_draft_spin(),))
        # Restore sub-checkbox state from saved spec_type
        spec_type_val = self.config.spec_type or ""
        iv_spec_ngram.set("ngram-mod" in spec_type_val)
        iv_spec_draft.set("draft-mtp" in spec_type_val)
        # Show sub-options if spec was already enabled on load
        if self.config.spec_enabled:
            spec_sub_row.pack()
            if iv_spec_draft.get():
                spec_draft_row.grid()
        # Restore draft model label if saved
        if self.config.draft_model_path:
            display = "Using: " + os.path.basename(self.config.draft_model_path).rsplit(".gguf", 1)[0]
            self.draft_model_label.config(text=display)

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
                        raw = s.get()
                        val = int(raw) if raw else lo
                        if not (lo <= val <= hi): return
                        var.set(max(lo, min(val, hi)))
                        s.set(max(lo, min(val, hi)))
                    except (ValueError, TypeError, tk.TclError):
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
            (iv_thread_batch, "Thread Batch", -1, 512, _on_thread_batch_change),
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
            cache_menu = ttk.OptionMenu(col, var, var.get(), "f16", "q8_0", "q5_0", "q4_0")
            cache_menu.pack(side="left")

        # --- Draft-MTP Cache Type Kd and Vd dropdowns (row 5, hidden by default) ---
        ct_draft_row = ttk.Frame(frame)
        sv_cache_kd = self._tk["cache_type_kd"]
        sv_cache_vd = self._tk["cache_type_vd"]

        for var, label in [(sv_cache_kd, "Cache Kd"), (sv_cache_vd, "Cache Vd")]:
            col = ttk.Frame(ct_draft_row)
            col.pack(side="left", padx=(0, 24))
            ttk.Label(col, text=label).pack(side="left")
            cache_menu_d = ttk.OptionMenu(col, var, var.get(), "f16", "q8_0", "q5_0", "q4_0")
            cache_menu_d.pack(side="left")

        def _on_draft_cache_toggle(*_):
            try:
                if iv_spec_draft.get():
                    ct_draft_row.grid(row=5, column=0, columnspan=2, sticky="w", pady=(4, 0))
                else:
                    ct_draft_row.grid_remove()
            except Exception:
                pass

        iv_spec_draft.trace_add("write", lambda *_: _on_draft_cache_toggle())
        # Show draft cache row if draft-mtp is already enabled
        if iv_spec_draft.get():
            ct_draft_row.grid(row=5, column=0, columnspan=2, sticky="w", pady=(4, 0))

        # --- Optimise button (row 6, left) ---
        opt_btn_frame = ttk.Frame(frame)
        opt_btn_frame.grid(row=6, column=0, sticky="w", pady=(8, 0))
        ttk.Button(opt_btn_frame, text="Optimise", command=self._run_optimiser).pack(side="left")

    def _section_server_settings(self, parent):
        """server settings section, always visible, sensible defaults"""
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
            try:
                raw = iv_port.get()
                val = max(1, min(int(raw), 65535)) if raw else 8080
                iv_port.set(val)
                self.config.port = val
                self._update_command()
            except (ValueError, TypeError, tk.TclError):
                pass

        port_frame = ttk.Frame(net_row)
        port_frame.pack(side="left", padx=(24, 0))
        ttk.Label(port_frame, text="Port:").pack(side="left", padx=(0, 4))
        def _port_spin_safe(*_):
            try:
                raw = iv_port.get()
                val = int(raw) if raw else 8080
                if not (1 <= val <= 65535): return
                iv_port.set(max(1, min(val, 65535)))
                self.config.port = max(1, min(val, 65535))
            except (ValueError, TypeError, tk.TclError):
                pass
        def _port_spin_cmd(*_):
            try: self._update_command()
            except Exception:
                pass
        iv_port.trace_add("write", lambda *_: (_port_spin_safe(), _port_spin_cmd()))

        entry_p = ttk.Spinbox(port_frame, from_=1, to=65535, textvariable=iv_port, width=8)
        entry_p.pack(side="left")

        # Cache RAM spinbox
        iv_cache_ram = self._tk["cache_ram"]
        cache_frame = ttk.Frame(net_row)
        cache_frame.pack(side="left", padx=(24, 0))
        ttk.Label(cache_frame, text="Cache RAM:").pack(side="left", padx=(0, 4))

        def _cache_ram_safe(*_):
            try:
                raw = iv_cache_ram.get()
                val = int(raw) if raw else 8000
                if not (0 <= val <= 999999): return
                iv_cache_ram.set(max(0, min(val, 999999)))
                self.config.cache_ram = max(0, min(val, 999999))
            except (ValueError, TypeError, tk.TclError):
                pass
        def _cache_ram_cmd(*_):
            try: self._update_command()
            except Exception:
                pass
        iv_cache_ram.trace_add("write", lambda *_: (_cache_ram_safe(), _cache_ram_cmd()))

        entry_c = ttk.Spinbox(cache_frame, from_=0, to=999999, textvariable=iv_cache_ram, width=8)
        entry_c.pack(side="left")


    def _section_sampling_params(self, parent):
        """sampling parameters section, always visible, two column grid"""
        samp_frame = ttk.Frame(parent)
        samp_frame.pack(fill="both", padx=6, pady=4)

        sv_temp = self._tk["temperature"]
        sv_minp = self._tk["min_p"]
        sv_topk = self._tk["top_k"]
        sv_pp = self._tk["presence_penalty"]
        sv_topp = self._tk["top_p"]
        sv_rp = self._tk["repeat_penalty"]

        # --- Temperature (row 0, col 0) ---
        def _temp_safe(*_):
            try:
                raw = sv_temp.get()
                val = float(raw) if raw else 0.8
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
        def _minp_safe(*_):
            try:
                raw = sv_minp.get()
                val = float(raw) if raw else 0.0
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
        def _topk_safe(*_):
            try:
                raw = sv_topk.get()
                val = int(raw) if raw else 40
                if not (1 <= val <= 9999): return
                self.config.top_k = max(1, min(val, 9999))
                sv_topk.set(max(1, min(val, 9999)))
            except (ValueError, TypeError, tk.TclError):
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
        def _pp_safe(*_):
            try:
                raw = sv_pp.get()
                val = float(raw) if raw else 0.0
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
        def _topp_safe(*_):
            try:
                raw = sv_topp.get()
                val = float(raw) if raw else 0.95
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
                raw = sv_rp.get()
                val = float(raw) if raw else 1.1
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
# Event handlers & helpers, user interactions and command generation logic
# Each section method creates a LabelFrame with its widgets and packs it into *parent*
# Live command generation is triggered by Tk variable traces on every input field
# ---------------------------------------------------------------------------

    def _section_hardware_info(self, parent):
        """read only hardware info display, cpu, gpu, vram, ram"""
        frame = ttk.LabelFrame(parent, text="System Hardware", padding=(8, 6))
        frame.pack(fill="both", padx=6, pady=4)

        cores = self._hw.get('CPU_CORES', '')
        threads = self._hw.get('CPU_THREADS', '')
        cores_str = f" ({cores} cores / {threads} threads)" if cores and threads else ""

        hw_text = (
            f"CPU:    {self._hw.get('CPU', 'Unknown')}{cores_str}\n"
            f"GPU:    {self._hw.get('GPU', 'Unknown')}\n"
            f"VRAM:   {self._hw.get('VRAM', 'Unknown')}\n"
            f"RAM:    {self._hw.get('RAM', 'Unknown')}"
        )
        info_label = tk.Label(
            frame, text=hw_text, justify="left",
            font=("Consolas", 10), bg="#f5f5f5"
        )
        info_label.pack(fill="x")

    def _browse_model(self):
        """open file dialog to select gguf model file"""
        initialdir = self._last_folder if self._last_folder and os.path.isdir(self._last_folder) else os.path.expanduser("~")
        path = filedialog.askopenfilename(
            title="Select GGUF Model File",
            filetypes=[("GGUF files", "*.gguf"), ("All files", "*.*")],
            initialdir=initialdir,
        )
        if path:
            self.config.model_path = path
            self._tk["model_path"].set(path)
            self._last_folder = os.path.dirname(path)
            _save_last_folder(self._last_folder)
            self._update_command()

    def _browse_draft_model(self):
        """open file dialog to select draft model for draft-mtp"""
        initialdir = self._last_folder if self._last_folder and os.path.isdir(self._last_folder) else os.path.expanduser("~")
        path = filedialog.askopenfilename(
            title="Select Draft Model File",
            filetypes=[("GGUF files", "*.gguf"), ("All files", "*.*")],
            initialdir=initialdir,
        )
        if path:
            self.config.draft_model_path = path
            self._last_folder = os.path.dirname(path)
            _save_last_folder(self._last_folder)
            display = os.path.basename(path).rsplit(".gguf", 1)[0]
            self.draft_model_label.config(text=display)
            self._update_command()

    def _open_huggingface(self):
        """open hugging face models page in default browser"""
        webbrowser.open("https://huggingface.co/models?library=gguf&sort=trending")

    def _validate_prerequisites(self):
        """check model selected, locate llama-server.exe, return dict with ok, model_path, server_exe"""
        if not self.config.model_path:
            return {"ok": False, "message": "Please select a model before optimising."}

        import shutil as _shutil, os as _os
        model_dir = _os.path.dirname(self.config.model_path)
        candidate = _os.path.join(model_dir, "llama-server.exe")
        if _os.path.isfile(candidate):
            server_exe = candidate
        elif _shutil.which("llama-server.exe"):
            server_exe = "llama-server.exe"
        elif _shutil.which("llama-server"):
            server_exe = "llama-server"
        else:
            return {"ok": False, "message": "llama-server.exe not found. Please ensure llama.cpp is on your PATH or in the same folder as your model."}

        perplexity_candidate = _os.path.join(model_dir, "llama-perplexity.exe")
        if _os.path.isfile(perplexity_candidate):
            perplexity_exe = perplexity_candidate
        elif _shutil.which("llama-perplexity.exe"):
            perplexity_exe = "llama-perplexity.exe"
        elif _shutil.which("llama-perplexity"):
            perplexity_exe = "llama-perplexity"
        else:
            return {"ok": False, "message": "llama-perplexity.exe not found. Please ensure llama.cpp is on your PATH or in the same folder as your model."}
        return {"ok": True, "model_path": self.config.model_path, "server_exe": server_exe, "perplexity_exe": perplexity_exe}

    def _show_optimiser_config_dialog(self):
        """Show optimiser settings. Returns dict or None if cancelled."""
        import tkinter as tk
        result = [None]

        config_data = _load_config()
        saved_method = config_data.get("optimiser_method", "bayesian")
        if saved_method not in AVAILABLE_METHODS:
            saved_method = "bayesian"
        saved_weight = config_data.get("optimiser_weight", 0.5)
        saved_ctx = config_data.get("optimiser_context_size", 16384)
        saved_trials = config_data.get("optimiser_trials", 40)
        saved_avg = config_data.get("optimiser_avg_runs", 1)
        saved_seed = config_data.get("optimiser_seed", 42)
        saved_ppl_threshold = config_data.get("optimiser_ppl_threshold_percent", self.config.ppl_threshold_percent)
        saved_perplexity_file = (
            config_data.get("perplexity_file")
            or self.config.perplexity_file
            or resolve_perplexity_file(DEFAULT_PERPLEXITY_FILE)
        )
        saved_perplexity_file = resolve_perplexity_file(saved_perplexity_file)

        dlg = Toplevel(self.root)
        dlg.title("Optimiser Settings")
        dlg.transient(self.root)
        dlg.grab_set()
        dlg.resizable(True, True)

        closed = [False]

        def _close_optimiser_settings(save_current=False):
            if closed[0]:
                return
            closed[0] = True
            try:
                if save_current:
                    payload = _get_optimiser_settings_payload()
                    self.config.ppl_threshold_percent = payload["ppl_threshold_percent"]
                    self.config.perplexity_file = payload["perplexity_file"]
                    _save_optimiser_settings(payload)
                else:
                    perplexity_value = perplexity_file_var.get()
                    if perplexity_value:
                        self.config.perplexity_file = perplexity_value
                        data = _load_config()
                        data["perplexity_file"] = perplexity_value
                        _save_config(data)
            except Exception:
                pass
            self._save_window_geometry("window_geometry_optimiser_settings", dlg)
            try:
                dlg.destroy()
            except Exception:
                pass

        dlg.protocol("WM_DELETE_WINDOW", _close_optimiser_settings)

        ttk.Label(dlg, text="Optimiser Settings", font=("Segoe UI", 11, "bold")).pack(pady=(12, 6))

        def _spin_row(label, var, from_, to, increment, width=8, fmt=None):
            row = ttk.Frame(dlg)
            row.pack(fill="x", padx=20, pady=3)
            ttk.Label(row, text=label).pack(anchor="w", side="left")
            kwargs = {"from_": from_, "to": to, "increment": increment, "textvariable": var, "width": width}
            if fmt is not None:
                kwargs["format"] = fmt
            ttk.Spinbox(row, **kwargs).pack(anchor="w", pady=2)

        def _path_picker_row(label, var):
            row = ttk.Frame(dlg)
            row.pack(fill="x", padx=20, pady=3)
            ttk.Label(row, text=label).pack(anchor="w", side="left")
            entry = ttk.Entry(row, textvariable=var, width=34)
            entry.pack(side="left", fill="x", expand=True, padx=(8, 4))
            ttk.Button(row, text="Browse...", command=lambda: _browse_perplexity_file(var)).pack(side="left")

        def _browse_perplexity_file(var):
            initialdir = os.path.dirname(var.get()) if var.get() and os.path.isdir(os.path.dirname(var.get())) else os.path.dirname(os.path.abspath(__file__))
            path = filedialog.askopenfilename(
                title="Select Perplexity Corpus File",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
                initialdir=initialdir,
            )
            if path:
                var.set(path)
                self.config.perplexity_file = path
                try:
                    data = _load_config()
                    data["perplexity_file"] = path
                    _save_config(data)
                except Exception:
                    pass

        method_var = tk.StringVar(value=saved_method)
        row_method = ttk.Frame(dlg)
        row_method.pack(fill="x", padx=20, pady=4)
        ttk.Label(row_method, text="Method:", width=34).pack(anchor="w")
        ttk.Combobox(row_method, textvariable=method_var, values=AVAILABLE_METHODS, state="readonly", width=22).pack(anchor="w", pady=2)

        weight_var = tk.DoubleVar(value=saved_weight)
        _spin_row("Score Weight: (Value closer to 0 puts more weight on TG, closer to 1 puts more weight on PP)\nRecommend setting weighting to lower end (0.2-0.4) by default due to PP \nusually being x times larger than TG:", weight_var, 0.0, 1.0, 0.05, fmt="%.2f")

        ctx_var = tk.IntVar(value=saved_ctx)
        _spin_row("Context Size:", ctx_var, 512, 1310720, 512, width=10)

        trials_var = tk.IntVar(value=saved_trials)
        _spin_row("Bayesian Trial Count (recommended 40):", trials_var, 1, 500, 1, width=8)

        avg_var = tk.IntVar(value=saved_avg)
        _spin_row("Runs per Trial (recommended 1):", avg_var, 1, 10, 1, width=8)

        seed_var = tk.IntVar(value=saved_seed)
        _spin_row("Seed (keep the same between optimisation runs for reproducibility):", seed_var, 0, 2147483647, 1, width=10)

        ppl_var = tk.DoubleVar(value=saved_ppl_threshold)
        _spin_row("PPL Threshold (% degradation allowed):", ppl_var, 1.0, 10.0, 0.5, width=8, fmt="%.1f")

        perplexity_file_var = tk.StringVar(value=saved_perplexity_file)
        _path_picker_row("Corpus File:", perplexity_file_var)

        def _get_optimiser_settings_payload():
            return {
                "method": method_var.get(),
                "metric_weight": weight_var.get(),
                "context_size": ctx_var.get(),
                "trials": trials_var.get(),
                "avg_runs": avg_var.get(),
                "seed": seed_var.get(),
                "ppl_threshold_percent": ppl_var.get(),
                "perplexity_file": perplexity_file_var.get(),
            }

        def _save_optimiser_settings(payload=None):
            payload = payload or _get_optimiser_settings_payload()
            try:
                data = _load_config()
                data.update({
                    "optimiser_method": payload["method"],
                    "optimiser_weight": payload["metric_weight"],
                    "optimiser_context_size": payload["context_size"],
                    "optimiser_trials": payload["trials"],
                    "optimiser_avg_runs": payload["avg_runs"],
                    "optimiser_seed": payload["seed"],
                    "optimiser_ppl_threshold_percent": payload["ppl_threshold_percent"],
                    "perplexity_file": payload["perplexity_file"],
                })
                _save_config(data)
            except Exception:
                pass

        def _ok():
            payload = _get_optimiser_settings_payload()
            result[0] = payload
            self.config.ppl_threshold_percent = payload["ppl_threshold_percent"]
            self.config.perplexity_file = payload["perplexity_file"]
            _save_optimiser_settings(payload)
            _close_optimiser_settings()

        def _cancel():
            _close_optimiser_settings(save_current=True)

        btn_row = ttk.Frame(dlg)
        btn_row.pack(pady=8)
        ttk.Button(btn_row, text="Start", command=_ok).pack(side="left", padx=6)
        ttk.Button(btn_row, text="Cancel", command=_cancel).pack(side="left", padx=6)

        self._restore_window_geometry(dlg, "window_geometry_optimiser_settings", 650, 460)
        self._ensure_min_geometry(dlg, 650, 460)
        dlg.minsize(650, 460)
        self._bind_window_geometry_persistence("window_geometry_optimiser_settings", dlg)

        try:
            dlg.wait_window()
        finally:
            if not closed[0]:
                self._save_window_geometry("window_geometry_optimiser_settings", dlg)
        return result[0]

    def _show_progress_window(self, request):
        """Show optimisation progress window. Returns final_config dict or None."""
        import threading as _threading
        import time as _time

        win = Toplevel(self.root)
        win.title("Optimisation in Progress")
        win.transient(self.root)
        win.grab_set()

        closed = [False]
        worker_done = [False]

        def _close_progress_window(request_cancel=False):
            if closed[0]:
                return
            if not worker_done[0]:
                if request_cancel and not cancel_flag[0]:
                    cancel_flag[0] = True
                    cancel_button.config(state="disabled")
                    label_var.set("Cancelling optimisation...")
                    kill_port(BENCH_PORT, proc_holder)
                win.after(250, lambda: _close_progress_window(request_cancel=False))
                return
            closed[0] = True
            self._save_window_geometry("window_geometry_optimisation_in_progress", win)
            try:
                win.destroy()
            except Exception:
                pass

        win.protocol("WM_DELETE_WINDOW", lambda: _close_progress_window(request_cancel=True))

        cancel_flag = [False]
        proc_holder = [None]
        final_config_holder = [None]
        error_holder = [None]

        ttk.Label(win, text="Optimisation in Progress", font=("Segoe UI", 12, "bold")).pack(pady=(10, 4))

        label_var     = tk.StringVar(value="Preparing...")
        remaining_var = tk.StringVar(value="Calculating...")
        ttk.Label(win, textvariable=label_var).pack()
        ttk.Label(win, textvariable=remaining_var, foreground="gray").pack()

        progress_var = tk.DoubleVar(value=0.0)
        bar = ttk.Progressbar(win, variable=progress_var, maximum=100, mode="determinate")
        bar.pack(fill="x", padx=20, pady=6)

        # Score display —> baseline / last / best PPL-validated
        scores_frame = ttk.Frame(win)
        scores_frame.pack(pady=4)

        baseline_var = tk.StringVar(value="--")
        last_var     = tk.StringVar(value="--")
        best_var     = tk.StringVar(value="--")

        for col, (lbl, var) in enumerate([
            ("Baseline", baseline_var),
            ("Last", last_var),
            ("Best", best_var),
        ]):
            f = ttk.Frame(scores_frame)
            f.grid(row=0, column=col, padx=20)
            ttk.Label(f, text=lbl, font=("Segoe UI", 8)).pack()
            tk.Label(f, textvariable=var, font=("Consolas", 12, "bold"), fg="#27ae60").pack()

        cancel_button = ttk.Button(win, text="Cancel")

        def _on_cancel():
            _close_progress_window(request_cancel=True)

        cancel_button.config(command=_on_cancel)
        cancel_button.pack(pady=8)

        start_time = [_time.time()]

        def _progress_callback(run_idx, total_runs, step_name, last_score, best_score, base_score):
            pct = max(0.0, min(run_idx / total_runs * 100, 99.9))
            elapsed = _time.time() - start_time[0]
            avg = elapsed / run_idx if run_idx > 0 else 0
            remaining = int(avg * (total_runs - run_idx))

            self.root.after(0, lambda p=pct: progress_var.set(p))
            self.root.after(0, lambda s=step_name: label_var.set(s))
            self.root.after(0, lambda r=remaining: remaining_var.set(f"~{r}s remaining"))
            self.root.after(0, lambda s=f"{base_score:.2f}": baseline_var.set(s))
            self.root.after(0, lambda s=f"{last_score:.2f}": last_var.set(s))
            self.root.after(0, lambda s=f"{best_score:.2f}": best_var.set(s))

        def _run_thread():
            try:
                print(f"[INFO] Starting optimisation method: {request.method}")
                result = OptimisationService().run(
                    request=request,
                    progress_callback=_progress_callback,
                    cancel_flag=cancel_flag,
                    proc_holder=proc_holder,
                )
                final_config_holder[0] = result
            except Exception as e:
                error_holder[0] = e
                final_config_holder[0] = None
            finally:
                # Ensure the server process is properly stopped after optimization completes
                try:
                    if cancel_flag[0]:
                        kill_port(BENCH_PORT, proc_holder)
                    else:
                        proc = proc_holder[0] if proc_holder else None
                        if proc is not None and proc.poll() is None:
                            proc.terminate()
                            proc.wait(timeout=5)
                except Exception:
                    kill_port(BENCH_PORT, proc_holder)
                finally:
                    if proc_holder:
                        proc_holder[0] = None
                    worker_done[0] = True
                    self.root.after(0, lambda: _close_progress_window(request_cancel=False))
        _threading.Thread(target=_run_thread, daemon=True).start()

        self._restore_window_geometry(win, "window_geometry_optimisation_in_progress", 620, 380)
        self._ensure_min_geometry(win, 620, 380)
        win.minsize(620, 380)
        self._bind_window_geometry_persistence("window_geometry_optimisation_in_progress", win)

        try:
            win.wait_window()
        finally:
            if not closed[0]:
                self._save_window_geometry("window_geometry_optimisation_in_progress", win)
        return final_config_holder[0], error_holder[0]

    def _show_results_window(self, final_config):
        """show optimisation results, best config and recommended flags"""
        win = Toplevel(self.root)
        win.title("Optimisation Results")
        win.transient(self.root)
        self._restore_window_geometry(win, "window_geometry_optimisation_results", 580, 440)

        def _close_results_window():
            self._save_window_geometry("window_geometry_optimisation_results", win)
            win.destroy()

        win.protocol("WM_DELETE_WINDOW", _close_results_window)

        ttk.Label(win, text="Optimisation Complete", font=("Segoe UI", 13, "bold")).pack(pady=6)

        text_area = tk.Text(win, width=60, height=16, font=("Consolas", 9), wrap="word")
        text_area.pack(fill="both", expand=True, padx=10, pady=4)

        def _safe_float(value, default=0.0):
            try:
                return float(value)
            except Exception:
                return default

        def _safe_int(value, default):
            try:
                return int(value)
            except Exception:
                return default

        ctx = final_config.get("context_size", 16384)
        method = final_config.get("method", "unknown")
        threads = final_config.get("threads", self.config.threads)
        thread_batch = final_config.get("thread_batch", threads)
        batch = final_config.get("batch", self.config.batch_size)
        micro_batch = final_config.get("micro_batch", self.config.micro_batch_size)
        fitt = final_config.get("fitt", self.config.fitt)
        cache_k = final_config.get("cache_k", self.config.cache_type_k)
        cache_v = final_config.get("cache_v", self.config.cache_type_v)
        baseline_pp = _safe_float(final_config.get("baseline_pp", 0.0))
        baseline_tg = _safe_float(final_config.get("baseline_tg", 0.0))
        baseline = _safe_float(final_config.get("baseline_score", 0.0))
        baseline_ppl = final_config.get("baseline_ppl")
        ppl_threshold = final_config.get("ppl_threshold", 0.03)
        best     = _safe_float(final_config.get("best_quality_score", final_config.get("best_score", 0.0)))
        best_pp  = _safe_float(final_config.get("best_pp", 0.0))
        best_tg  = _safe_float(final_config.get("best_tg", 0.0))
        best_ppl = final_config.get("best_ppl")

        def _safe_bool(value, default=False):
            if value is None:
                return default
            if isinstance(value, str):
                return value.strip().lower() in ("1", "true", "yes", "on")
            return bool(value)

        flash_attention = _safe_bool(final_config.get("flash_attention"), False)
        fit_on = _safe_bool(final_config.get("fit_on"), False)
        use_baseline_command = _safe_bool(final_config.get("use_baseline_command"), False)
        pct_gain = 0.0 if use_baseline_command else ((best - baseline) / baseline * 100) if baseline > 0 else 0.0

        spec_enabled = _safe_bool(final_config.get("spec_enabled"), self.config.spec_enabled) if "spec_enabled" in final_config else bool(final_config.get("mtp") or final_config.get("draft_model_path") or self.config.spec_enabled)
        spec_type = str(final_config.get("spec_type", "draft-mtp" if spec_enabled else self.config.spec_type) or "")
        spec_draft_n = final_config.get("spec_draft_n", self.config.spec_draft_n_max)
        spec_draft_p_min = final_config.get("spec_draft_p_min", self.config.spec_draft_p_min)
        spec_draft_n_min = final_config.get("spec_draft_n_min", self.config.spec_draft_n_min)
        draft_model_path = final_config.get("draft_model_path", self.config.draft_model_path or "")
        draft_model_path = "" if draft_model_path is None else str(draft_model_path)
        cache_type_kd = final_config.get("cache_type_kd", self.config.cache_type_kd)
        cache_type_vd = final_config.get("cache_type_vd", self.config.cache_type_vd)

        threads = _safe_int(threads, threads)
        thread_batch = _safe_int(thread_batch, thread_batch)
        batch = _safe_int(batch, batch)
        micro_batch = _safe_int(micro_batch, micro_batch)
        fitt = _safe_int(fitt, fitt)
        display_threads = "--" if use_baseline_command else threads
        display_thread_batch = "--" if use_baseline_command else thread_batch
        display_batch = "--" if use_baseline_command else batch
        display_micro_batch = "--" if use_baseline_command else micro_batch
        display_fitt = "--" if use_baseline_command else fitt

        extra_flags = []
        if flash_attention:
            extra_flags.append("-fa on")
        if fit_on:
            extra_flags.append("--fit on")
        if spec_enabled:
            extra_flags.append(f"--spec-type {spec_type}")
            if spec_draft_n:
                extra_flags.append(f"--spec-draft-n-max {int(spec_draft_n)}")
            if spec_draft_n_min:
                extra_flags.append(f"--spec-draft-n-min {int(spec_draft_n_min)}")
            if spec_draft_p_min:
                try:
                    extra_flags.append(f"--spec-draft-p-min {float(spec_draft_p_min):.1f}")
                except Exception:
                    pass
            if draft_model_path:
                extra_flags.append(f'--model-draft "{draft_model_path}"')
            if "draft-mtp" in (spec_type or ""):
                extra_flags.extend([f"-ctkd {cache_type_kd}", f"-ctvd {cache_type_vd}"])

        if use_baseline_command:
            recommended_flags = "Baseline command: no tuned -t/-tb/-b/-ub/-fa/--fit/-fitt/-ct flags were applied."
        else:
            recommended_flags = " ".join([
                f"-t {threads}",
                f"-tb {thread_batch}",
                f"-b {batch}",
                f"-ub {micro_batch}",
                f"-c {ctx}",
            ] + extra_flags + [
                f"-fitt {fitt}",
                f"-ctk {cache_k}",
                f"-ctv {cache_v}",
            ])

        lines = [
            f"Method:           {method}",
            f"Context Size:     {ctx}",
            f"Threads:          {display_threads}",
            f"Thread Batch:     {display_thread_batch}",
            f"Batch Size:       {display_batch}",
            f"Micro-Batch:      {display_micro_batch}",
            f"FITT Target:      {display_fitt}",
            f"Cache K:          {cache_k}",
            f"Cache V:          {cache_v}",
            f"Cache Kd:         {cache_type_kd if spec_enabled and 'draft-mtp' in (spec_type or '') else '--'}",
            f"Cache Vd:         {cache_type_vd if spec_enabled and 'draft-mtp' in (spec_type or '') else '--'}",
            f"Speculative:      {'Yes' if spec_enabled else 'No'}",
            f"Spec Type:        {spec_type if spec_enabled else '--'}",
            f"Spec Draft N Max: {spec_draft_n if spec_enabled else '--'}",
            f"Spec Draft P Min: {spec_draft_p_min if spec_enabled else '--'}",
            f"Draft Model:      {os.path.basename(draft_model_path) if draft_model_path and spec_enabled else '--'}",
            f"",
            f"Baseline Score:   {baseline:.2f} (pre-optimisation)",
            f"Baseline PP:      {baseline_pp:.2f} t/s",
            f"Baseline TG:      {baseline_tg:.2f} t/s",
            f"Baseline PPL:     {baseline_ppl if baseline_ppl is not None else '--'}",
            f"Best PPL:         {best_ppl if best_ppl is not None else ('--' if use_baseline_command else '-- (PPL skipped)')}",
            f"PPL Threshold:    {ppl_threshold * 100:.1f}%" if baseline_ppl is not None else "PPL Threshold:    --",
            f"Best Score:       {best:.2f}",
            f"Best PP Speed:    {best_pp:.2f} t/s",
            f"Best TG Speed:    {best_tg:.2f} t/s",
            f"Improvement:      {pct_gain:.2f}%",
            f"Result:           {'No trial beat baseline; using baseline command.' if use_baseline_command else 'Best PPL-validated trial selected.'}",
            f"",
            f"{'Baseline command' if use_baseline_command else 'Recommended flags for llama-server'}:",
            recommended_flags,
        ]
        text_area.insert("1.0", "\n".join(lines))
        text_area.configure(state="disabled")

        def _set_config_attr(name, value):
            if hasattr(self.config, name):
                setattr(self.config, name, value)

        def _set_var(name, value):
            try:
                self._tk[name].set(value)
            except Exception:
                pass

        def _apply():
            _set_config_attr("ctx_size_value", ctx)
            _set_config_attr("batch_size", batch)
            _set_config_attr("micro_batch_size", micro_batch)
            _set_config_attr("fitt", fitt)
            if use_baseline_command:
                _set_config_attr("threads", -1)
                _set_config_attr("thread_batch", -1)
                _set_config_attr("flash_attention", False)
                _set_config_attr("fit_on", False)
                _set_config_attr("cache_type_k", "f16")
                _set_config_attr("cache_type_v", "f16")
                _set_config_attr("cache_type_kd", "f16")
                _set_config_attr("cache_type_vd", "f16")
                _set_var("threads_val", -1)
                _set_var("thread_batch", -1)
                _set_var("batch_size", batch)
                _set_var("micro_batch", micro_batch)
                _set_var("fitt", fitt)
                _set_var("flash_attention", False)
                _set_var("fit_on", False)
                _set_var("cache_type_k", "f16")
                _set_var("cache_type_v", "f16")
                _set_var("cache_type_kd", "f16")
                _set_var("cache_type_vd", "f16")
            else:
                _set_config_attr("threads", threads)
                _set_config_attr("thread_batch", thread_batch)
                _set_config_attr("batch_size", batch)
                _set_config_attr("micro_batch_size", micro_batch)
                _set_config_attr("fitt", fitt)
                _set_config_attr("cache_type_k", cache_k)
                _set_config_attr("cache_type_v", cache_v)
                _set_config_attr("cache_type_kd", cache_type_kd)
                _set_config_attr("cache_type_vd", cache_type_vd)
                _set_config_attr("flash_attention", flash_attention)
                _set_config_attr("fit_on", fit_on)
                _set_var("threads_val", threads)
                _set_var("thread_batch", thread_batch)
                _set_var("batch_size", batch)
                _set_var("micro_batch", micro_batch)
                _set_var("fitt", fitt)
                _set_var("cache_type_k", cache_k)
                _set_var("cache_type_v", cache_v)
                _set_var("cache_type_kd", cache_type_kd)
                _set_var("cache_type_vd", cache_type_vd)
                _set_var("flash_attention", flash_attention)
                _set_var("fit_on", fit_on)
            _set_config_attr("spec_enabled", spec_enabled)
            _set_config_attr("spec_type", spec_type)
            _set_config_attr("spec_draft_n_max", int(spec_draft_n) if spec_enabled and spec_draft_n else 0)
            _set_config_attr("spec_draft_n_min", int(spec_draft_n_min) if spec_enabled and spec_draft_n_min else 0)
            _set_config_attr("spec_draft_p_min", float(spec_draft_p_min) if spec_enabled and spec_draft_p_min else 0.0)
            _set_config_attr("draft_model_path", draft_model_path if spec_enabled and "draft-mtp" in (spec_type or "") and draft_model_path else "")
            _set_var("ctx_size_value", ctx)
            _set_var("spec_enabled", spec_enabled)
            _set_var("spec_ngram", spec_enabled and "ngram-mod" in (spec_type or ""))
            _set_var("spec_draft", spec_enabled and "draft-mtp" in (spec_type or ""))
            _set_var("spec_draft_max", int(spec_draft_n) if spec_enabled and spec_draft_n else 0)
            _set_var("spec_draft_min", int(spec_draft_n_min) if spec_enabled and spec_draft_n_min else 0)
            _set_var("spec_draft_p_min", float(spec_draft_p_min) if spec_enabled and spec_draft_p_min else 0.0)
            if draft_model_path and spec_enabled and "draft-mtp" in (spec_type or ""):
                self.draft_model_label.config(text="Using: " + os.path.basename(draft_model_path).rsplit(".gguf", 1)[0])
            else:
                self.draft_model_label.config(text="")
            try:
                if spec_enabled:
                    self._spec_sub_row.pack()
                else:
                    self._spec_sub_row.pack_forget()
                if spec_enabled and "draft-mtp" in (spec_type or ""):
                    self._spec_draft_row.grid()
                else:
                    self._spec_draft_row.grid_remove()
            except Exception:
                pass
            self._update_command()
            _close_results_window()

        def _copy():
            self.root.clipboard_clear()
            self.root.clipboard_append(recommended_flags)

        btn_frame = ttk.Frame(win)
        btn_frame.pack(fill="x", pady=8)
        ttk.Button(btn_frame, text="Apply Settings", command=_apply).pack(side="right", padx=4)
        ttk.Button(btn_frame, text="Copy Flags", command=_copy).pack(side="right", padx=4)
        ttk.Button(btn_frame, text="Close", command=_close_results_window).pack(side="right")



    def _update_command(self):
        """rebuild generated command, update display"""
        cmd = self.config.generate_command()
        if hasattr(self, 'cmd_text'):
            self.cmd_text.configure(state='normal')
            self.cmd_text.delete('1.0', 'end')
            self.cmd_text.insert('1.0', cmd)
            self.cmd_text.configure(state='disabled')

    def _run_optimiser(self):
        """entry point for modular optimisation service"""
        from tkinter import messagebox

        # Warning dialog (custom Toplevel so geometry persists)
        msg = (
            "Model Optimisation\n\n"
            "Your system will be under high load during this process.\n"
            "Please save your work and close demanding applications.\n"
            "Proceed?"
        )

        confirm = {"value": False}
        dlg = Toplevel(self.root)
        dlg.title("Optimise System")
        dlg.transient(self.root)
        dlg.grab_set()
        dlg.resizable(False, False)
        # Restore last geometry or center default
        self._restore_window_geometry(dlg, "window_geometry_optimise_system", 420, 160)

        def _close_confirm():
            # Save geometry and close, defaulting to False
            try:
                self._save_window_geometry("window_geometry_optimise_system", dlg)
            except Exception:
                pass
            confirm["value"] = False
            try: dlg.destroy()
            except Exception: pass

        def _on_yes():
            try:
                self._save_window_geometry("window_geometry_optimise_system", dlg)
            except Exception:
                pass
            confirm["value"] = True
            try: dlg.destroy()
            except Exception: pass

        def _on_no():
            _close_confirm()

        ttk.Label(dlg, text=msg, wraplength=380, justify="left").pack(padx=16, pady=12)
        btn_row = ttk.Frame(dlg)
        btn_row.pack(pady=8)
        ttk.Button(btn_row, text="Yes", command=_on_yes).pack(side="left", padx=6)
        ttk.Button(btn_row, text="No", command=_on_no).pack(side="left", padx=6)
        dlg.protocol("WM_DELETE_WINDOW", _close_confirm)
        dlg.wait_window()
        if not confirm["value"]:
            return

        # Validate prerequisites
        try:
            prereq_result = self._validate_prerequisites()
            if not prereq_result.get("ok"):
                messagebox.showerror("Prerequisite Error", prereq_result["message"])
                return
        except Exception as e:
            messagebox.showerror("Error", f"Failed to validate prerequisites:\n{e}")
            return

        model_path = prereq_result["model_path"]
        server_exe = prereq_result["server_exe"]
        perplexity_exe = prereq_result["perplexity_exe"]

        # Config dialog ask for weight and context size
        cfg = self._show_optimiser_config_dialog()
        if cfg is None:
            return

        draft_path = None
        mtp_enabled = False
        if self.config.spec_enabled:
            draft_path = self.config.draft_model_path.strip() if self.config.draft_model_path else None
            mtp_enabled = "draft-mtp" in (self.config.spec_type or "")

        request = OptimisationRequest(
            model_path=model_path,
            server_exe=server_exe,
            perplexity_exe=perplexity_exe,
            perplexity_file=cfg["perplexity_file"],
            context_size=cfg["context_size"],
            metric_weight=cfg["metric_weight"],
            ppl_threshold_percent=cfg["ppl_threshold_percent"],
            method=cfg["method"],
            draft_model_path=draft_path,
            mtp=mtp_enabled,
            trials=cfg["trials"],
            avg_runs=cfg["avg_runs"],
            seed=cfg["seed"],
        )

        # Run optimiser
        try:
            final_config, optimisation_error = self._show_progress_window(request)
        except Exception as e:
            messagebox.showerror("Error", f"Optimisation failed:\n{e}")
            return

        if optimisation_error is not None:
            messagebox.showerror("Error", f"Optimisation failed:\n{optimisation_error}")
            return

        if not final_config:
            messagebox.showinfo("Results", "No valid results collected.")
            return

        # Pass request details into config for display in results
        final_config["context_size"] = request.context_size
        final_config["method"] = request.method
        self._show_results_window(final_config)


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
        ttk.Button(row, text="Browse local files...", command=_browse_folder).pack(side="left")

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
                f.write(cmd)
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
                f'cmd.exe /k "{cmd}"',
                creationflags=subprocess.DETACHED_PROCESS,
                shell=True,
            )
        except Exception as e:
            messagebox.showerror("Error", f"Could not run command:\n{e}")