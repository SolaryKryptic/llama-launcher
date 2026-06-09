import os
import json
import subprocess
import webbrowser
from tkinter import filedialog, messagebox, Tk, Toplevel, StringVar
import tkinter as tk
from tkinter import ttk
from hardwarescanner import scan_hardware
import sys as _sys2

_CONFIG_PATH = os.path.join(os.getcwd(), "llama_gui_data.json")
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

    def __init__(self):
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
        self.draft_model_path = ""       # draft model path for draft-mtp

        self.cache_type_k = "f16"      # kv cache type k
        self.cache_type_v = "f16"      # kv cache type v
        self.cache_type_kd = "f16"     # draft-mtp cache type k
        self.cache_type_vd = "f16"     # draft-mtp cache type v

        self.temperature = 0.8          # sampling temperature
        self.min_p = 0.0                # minimum p sampling value
        self.top_k = 40                 # top k sampling value
        self.presence_penalty = 0.0     # presence penalty, negative to penalise
        self.top_p = 0.95               # top p nucleus sampling
        self.repeat_penalty = 1.1       # repeat penalty, >1 to reduce repeats

    def generate_command(self):
        parts = ["llama-server.exe -lv 4"]

        model_path = self.model_path.strip()
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
            draft_path = self.draft_model_path.strip()
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
            "draft_model_path": self.draft_model_path,
            "batch_size": self.batch_size,
            "micro_batch_size": self.micro_batch_size,
            "threads": self.threads,
            "thread_batch": self.thread_batch,
            "cache_type_k": self.cache_type_k,
            "cache_type_v": self.cache_type_v,
            "cache_type_kd": self.cache_type_kd,
            "cache_type_vd": self.cache_type_vd,
            "temperature": self.temperature,
            "min_p": self.min_p,
            "top_k": self.top_k,
            "presence_penalty": self.presence_penalty,
            "top_p": self.top_p,
            "repeat_penalty": self.repeat_penalty,
        }

    def from_dict(self, d):
        """restore mutable flag state from saved dict"""
        for key, val in d.items():
            if hasattr(self, key):
                setattr(self, key, val)


# UI builder — all ttk widget frames returned as methods
# Each section method creates a LabelFrame with its widgets and packs it into *parent*
# Live command generation is triggered by Tk variable traces on every input field

class LlamaServerGUI:
    """main tkinter gui for configuring llama server flags, generating commands"""

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
# Section builders — each returns a ttk.Frame packed into *parent*
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

    def _restore_vars(self, saved_flags):
        """set tk variable values to match saved flag state"""
        tk = self._tk
        if "model_path" in saved_flags:
            tk["model_path"].set(str(saved_flags["model_path"]))
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
                self.config.spec_type = str(saved_flags["spec_type"])
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
                tk["cache_type_k"].set(str(saved_flags["cache_type_k"]))
            except (ValueError, TypeError):
                pass
        if "cache_type_v" in saved_flags:
            try:
                tk["cache_type_v"].set(str(saved_flags["cache_type_v"]))
            except (ValueError, TypeError):
                pass
        if "cache_type_kd" in saved_flags:
            try:
                tk["cache_type_kd"].set(str(saved_flags["cache_type_kd"]))
            except (ValueError, TypeError):
                pass
        if "cache_type_vd" in saved_flags:
            try:
                tk["cache_type_vd"].set(str(saved_flags["cache_type_vd"]))
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
        self._tk["spec_ngram"] = iv_spec_ngram
        self._tk["spec_draft"] = iv_spec_draft
        self._tk["spec_draft_max"] = iv_spec_draft_max
        self._tk["spec_draft_min"] = iv_spec_draft_min

        spec_row = ttk.Frame(spec_frame)
        spec_row.pack(fill="x")
        tk.Checkbutton(spec_row, text="Speculative Decoding", variable=iv_spec_enabled).pack(side="left")

        # Sub-options row (child of spec_frame — same pack master)
        spec_sub_row = ttk.Frame(spec_frame)
        tk.Checkbutton(spec_sub_row, text="ngram-mod", variable=iv_spec_ngram).pack(side="left")
        tk.Checkbutton(spec_sub_row, text="draft-mtp", variable=iv_spec_draft).pack(side="left", padx=(16, 0))

        # Draft model inline block: browse button + label + max/min spinboxes
        spec_draft_row = ttk.Frame(spec_sub_row)
        ttk.Button(spec_draft_row, text="Browse draft models...", command=self._browse_draft_model).pack(side="left", padx=(12, 0))
        self.draft_model_label = tk.Label(spec_draft_row, text="", anchor="w", justify="left", foreground="#666")
        self.draft_model_label.pack(side="left", padx=(4, 0))
        ttk.Label(spec_draft_row, text="Max:").pack(side="left", padx=(12, 0))
        ttk.Spinbox(spec_draft_row, from_=0, to=64, textvariable=iv_spec_draft_max, width=4).pack(side="left", padx=(2, 0))
        ttk.Label(spec_draft_row, text="Min:").pack(side="left", padx=(12, 0))
        ttk.Spinbox(spec_draft_row, from_=0, to=64, textvariable=iv_spec_draft_min, width=4).pack(side="left", padx=(2, 0))

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
                    spec_draft_row.pack()
                else:
                    spec_draft_row.pack_forget()
                    self.config.draft_model_path = ""
                    self.config.spec_draft_n_max = 0
                    self.config.spec_draft_n_min = 0
                    self.draft_model_label.config(text="")
                    iv_spec_draft_max.set(0)
                    iv_spec_draft_min.set(0)
                self._update_command()
            except Exception:
                pass

        def _on_spec_draft_spin(*_):
            try:
                self.config.spec_draft_n_max = max(0, iv_spec_draft_max.get())
                self.config.spec_draft_n_min = max(0, iv_spec_draft_min.get())
                self._update_command()
            except (ValueError, TypeError, tk.TclError):
                pass

        iv_spec_enabled.trace_add("write", lambda *_: _on_spec_toggle())
        iv_spec_ngram.trace_add("write", lambda *_: _on_spec_sub())
        iv_spec_draft.trace_add("write", lambda *_: _on_spec_sub())
        iv_spec_draft_max.trace_add("write", lambda *_: (_on_spec_draft_spin(),))
        iv_spec_draft_min.trace_add("write", lambda *_: (_on_spec_draft_spin(),))
        # Restore sub-checkbox state from saved spec_type
        spec_type_val = self.config.spec_type or ""
        iv_spec_ngram.set("ngram-mod" in spec_type_val)
        iv_spec_draft.set("draft-mtp" in spec_type_val)
        # Show sub-options if spec was already enabled on load
        if self.config.spec_enabled:
            spec_sub_row.pack()
            if iv_spec_draft.get():
                spec_draft_row.pack()
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
        ttk.Button(opt_btn_frame, text="Optimise (WIP)", command=self._run_optimiser).pack(side="left")

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
# Event handlers & helpers — user interactions and command generation logic
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
        """check model selected, check bench exe, return dict with ok, model_path, bench_exe"""
        if not self.config.model_path:
            return {"ok": False, "message": "Please select a model before optimising."}

        # Find llama-bench.exe next to llama-server.exe
        bench_exe = "llama-bench.exe"
        return {"ok": True, "model_path": self.config.model_path, "bench_exe": bench_exe}

    def _build_optimiser_params(self):
        """build sweep parameters for optimiser, return dict of lists and limits"""
        import itertools as _it

        logical_cores = os.cpu_count() or 4
        cap_limit = max(1, int(logical_cores * 0.75))
        step_size = max(1, int(logical_cores * 0.25))
        threads_list = [t for t in range(step_size, cap_limit + 1, step_size)]
        if cap_limit not in threads_list:
            threads_list.append(cap_limit)

        batch_sizes = [512, 1024, 2048]
        fitt_targets = [1024, 512, 256]
        cache_k_types = ["f16", "q8_0", "q4_0"]

        return {
            "threads": threads_list,
            "batch_sizes": batch_sizes,
            "fitt_targets": fitt_targets,
            "cache_k_types": cache_k_types,
            "cap_limit": cap_limit,
            "step_size": step_size,
            "logical_cores": logical_cores,
        }

    def _build_bench_cmd(self, bench_exe, model_path, threads=None, batch=None, micro_batch=None,
                         fitt=None, ctx_size=16384, ctk="f16", ctv="f16", is_base=False):
        """build llama bench command, is_base runs baseline without extra flags, otherwise include parameters"""
        cmd = [bench_exe, "-m", model_path, "-o", "md", "--no-warmup",
               "-fitc", str(ctx_size), "-r", "3"]
        if not is_base:
            cmd += ["-t", str(threads), "-b", str(batch), "-ub", str(micro_batch),
                    "-fa", "auto", "-fitt", str(fitt), "-ctk", ctk, "-ctv", ctv]
        return cmd

    @staticmethod
    def _parse_bench_results(output):
        """parse prompt pp512, text gen tg128 tps from llama bench output"""
        import re as _re
        results = {}
        for line in output.splitlines():
            if "pp512" in line:
                parts = [p.strip() for p in line.split("|")]
                tps = parts[-2].split("\u00b1")[0].strip()
                tps = _re.sub(r"[^\d.]", "", tps)
                results["pp512"] = float(tps)
            elif "tg128" in line:
                parts = [p.strip() for p in line.split("|")]
                tps = parts[-2].split("\u00b1")[0].strip()
                tps = _re.sub(r"[^\d.]", "", tps)
                results["tg128"] = float(tps)
        return results

    @staticmethod
    def _calculate_score(pp, tg, metric_weight=0.5):
        """dual metric weighted score"""
        return (pp * metric_weight) + (tg * (1.0 - metric_weight))

    def _run_optimisation(self, bench_exe, model_path, progress_var,
                          label_var, score_var, remaining_var, cancel_flag):
        """run two stage optimiser, update tk variables via root after, return final config via holder"""
        import subprocess as _sub
        import itertools as _it
        import time as _time

        params = self._build_optimiser_params()
        ctx_size = 16384
        metric_weight = 0.5

        def run_one(t=None, b=None, fitt=None, cache_k=None, is_base=False):
            cmd = self._build_bench_cmd(
                bench_exe, model_path,
                threads=t, batch=b, micro_batch=fitt,
                fitt=fitt, ctx_size=ctx_size,
                ctk=cache_k, ctv="f16", is_base=is_base)
            try:
                proc = _sub.run(
                    cmd, capture_output=True, text=True, errors="ignore",
                    timeout=300  # hard cap per run (5 min)
                )
                if proc.returncode != 0:
                    return None, None
                results = self._parse_bench_results(proc.stdout)
                pp = results.get("pp512", 0.0)
                tg = results.get("tg128", 0.0)
                if pp == 0.0 and tg == 0.0:
                    return None, None
                return pp, tg
            except Exception:
                return None, None

        def calc(pp, tg):
            if pp is None or tg is None:
                return -1.0
            return self._calculate_score(pp, tg, metric_weight)

        # --- STAGE 1: Sequential greedy screening ---
        best_threads = params["threads"][-1]
        best_batch = params["batch_sizes"][0]
        best_fitt = params["fitt_targets"][0]
        best_cache_k = params["cache_k_types"][0]

        global_best_score = -1.0
        global_best_pp = 0.0
        global_best_tg = 0.0

        total_runs = len(params["threads"]) + len(params["batch_sizes"]) + \
                     len(params["fitt_targets"]) + len(params["cache_k_types"])
        elapsed_start = None

        def update_progress(step_name, param_name, value, run_idx):
            total_runs_count = (
                len(params["threads"]) + len(params["batch_sizes"])
                + len(params["fitt_targets"]) + len(params["cache_k_types"])
                + 6  # approximate neighbourhood grid size
            )
            pct = max(0.0, min(run_idx / total_runs_count * 100, 99.9))
            self.root.after(0, lambda p=pct: progress_var.set(p))
            self.root.after(0, lambda msg=f"{step_name}: {param_name}={value}": label_var.set(msg))
            try:
                now = _time.time()
                if elapsed_start is not None:
                    elapsed = now - elapsed_start
                    avg_per_run = elapsed / run_idx
                    remaining = int(avg_per_run * (total_runs_count - run_idx))
                    self.root.after(0, lambda r=remaining: remaining_var.set(f"~{r}s remaining"))
            except Exception:
                pass

        # 1. Threads
        self.root.after(0, lambda: label_var.set("Stage 1 — Optimising threads..."))
        for t in params["threads"]:
            if cancel_flag[0]:
                return None
            pp, tg = run_one(t=t, b=best_batch, fitt=best_fitt, cache_k=best_cache_k)
            score = calc(pp, tg)
            if score > 0:
                if score >= global_best_score:
                    global_best_score = score
                    global_best_pp = pp or 0.0
                    global_best_tg = tg or 0.0
                if score >= calc(None, None) or True:  # always track stage best
                    best_threads = t
                self.root.after(0, lambda s=f"pp={pp:.2f} tg={tg:.2f}": label_var.set(s))
                self.root.after(0, lambda sc=score: score_var.set(sc))

        # 2. Batch size
        self.root.after(0, lambda: label_var.set("Stage 1 — Optimising batch size..."))
        for b in params["batch_sizes"]:
            if cancel_flag[0]:
                return None
            pp, tg = run_one(t=best_threads, b=b, fitt=best_fitt, cache_k=best_cache_k)
            score = calc(pp, tg)
            if score > 0:
                if score >= global_best_score:
                    global_best_score = score
                    global_best_pp = pp or 0.0
                    global_best_tg = tg or 0.0
                best_batch = b
                self.root.after(0, lambda sc=score: score_var.set(sc))

        # 3. FITT target
        self.root.after(0, lambda: label_var.set("Stage 1 — Optimising FITT target..."))
        for fitt in params["fitt_targets"]:
            if cancel_flag[0]:
                return None
            pp, tg = run_one(t=best_threads, b=best_batch, fitt=fitt, cache_k=best_cache_k)
            score = calc(pp, tg)
            if score > 0:
                if score >= global_best_score:
                    global_best_score = score
                    global_best_pp = pp or 0.0
                    global_best_tg = tg or 0.0
                best_fitt = fitt
                self.root.after(0, lambda sc=score: score_var.set(sc))

        # 4. K Cache type
        self.root.after(0, lambda: label_var.set("Stage 1 — Optimising K cache type..."))
        for ck in params["cache_k_types"]:
            if cancel_flag[0]:
                return None
            pp, tg = run_one(t=best_threads, b=best_batch, fitt=best_fitt, cache_k=ck)
            score = calc(pp, tg)
            if score > 0:
                if score >= global_best_score:
                    global_best_score = score
                    global_best_pp = pp or 0.0
                    global_best_tg = tg or 0.0
                best_cache_k = ck
                self.root.after(0, lambda sc=score: score_var.set(sc))

        # --- STAGE 2: Neighbourhood verification (batch × fitt) ---
        def neighbours(current, sweep_list):
            idx = sweep_list.index(current)
            nb = [current]
            if idx > 0: nb.append(sweep_list[idx - 1])
            if idx < len(sweep_list) - 1: nb.append(sweep_list[idx + 1])
            return list(set(nb))

        b_nb = neighbours(best_batch, params["batch_sizes"])
        f_nb = neighbours(best_fitt, params["fitt_targets"])
        grid = list(_it.product(b_nb, f_nb))

        final_best_score = global_best_score
        final_best_pp = global_best_pp
        final_best_tg = global_best_tg
        final_config = {"threads": best_threads, "batch": best_batch,
                        "fitt": best_fitt, "cache_k": best_cache_k}

        self.root.after(0, lambda: label_var.set("Stage 2 — Neighbourhood verification..."))
        for b, ft in grid:
            if cancel_flag[0]:
                return None
            if b == best_batch and ft == best_fitt:
                continue
            pp, tg = run_one(t=best_threads, b=b, fitt=ft, cache_k=best_cache_k)
            score = calc(pp, tg)
            self.root.after(0, lambda sc=score: score_var.set(sc))
            if score > final_best_score:
                final_best_score = score
                final_best_pp = pp or 0.0
                final_best_tg = tg or 0.0
                final_config = {"threads": best_threads, "batch": b,
                                "fitt": ft, "cache_k": best_cache_k}

        # Return via mutable holder
        return final_config

    def _show_progress_window(self, model_path, bench_exe):
        """show optimisation progress window, return final config or none"""
        import threading as _threading

        win = Toplevel(self.root)
        win.title("Optimisation in Progress")
        win.geometry("620x320")
        win.transient(self.root)
        win.grab_set()

        label_var = tk.StringVar(value="Preparing...")
        score_var = tk.DoubleVar(value=0.0)
        remaining_var = tk.StringVar(value="Calculating...")
        cancel_flag = [False]
        final_config_holder = [None]  # mutable container for result

        ttk.Label(win, text="Optimisation in Progress", font=("Segoe UI", 12, "bold")).pack(pady=8)
        ttk.Label(win, textvariable=label_var).pack()
        ttk.Label(win, textvariable=remaining_var).pack()

        progress_var = tk.IntVar(value=0)
        bar = ttk.Progressbar(win, variable=progress_var, maximum=100, mode="determinate")
        bar.pack(fill="x", padx=20, pady=8)

        def _update_bar():
            try:
                val = float(bar.cget("value"))
            except Exception:
                return
            if val < 99.5:
                win.after(500, _update_bar)

        ttk.Label(win, text="Score so far:", font=("Segoe UI", 9)).pack()
        score_label = tk.Label(win, textvariable=score_var,
                               font=("Consolas", 11, "bold"), fg="#27ae60")
        score_label.pack(pady=4)

        def _on_cancel():
            cancel_flag[0] = True
            win.destroy()

        ttk.Button(win, text="Cancel", command=_on_cancel).pack(pady=8)

        def _run_thread():
            result = self._run_optimisation(
                bench_exe, model_path,
                progress_var, label_var, score_var, remaining_var, cancel_flag)
            final_config_holder[0] = result
            win.destroy()

        _threading.Thread(target=_run_thread, daemon=True).start()
        win.wait_window()
        return final_config_holder[0]

        # Collect results from the JSON file (sweep thread writes it)
        try:
            import json as _json
            with open("optimiser_results.json", "r") as f:
                return _json.load(f)
        except Exception:
            return []

    def _show_results_window(self, final_config):
        """show optimisation results, best config and recommended flags"""
        win = Toplevel(self.root)
        win.title("Optimisation Results")
        win.geometry("580x420")
        win.transient(self.root)

        ttk.Label(win, text="Optimisation Complete", font=("Segoe UI", 13, "bold")).pack(pady=6)

        # Result display
        text_area = tk.Text(win, width=60, height=14, font=("Consolas", 9), wrap="word")
        text_area.pack(fill="both", expand=True, padx=10, pady=4)

        lines = [
            f"Threads:     {final_config['threads']}",
            f"Batch Size:  {final_config['batch']}",
            f"FITT Target: {final_config['fitt']}",
            f"Cache K:     {final_config['cache_k']}",
            "",
            "Recommended flags for llama-server:",
            f"-t {final_config['threads']} -b {final_config['batch']} -ub {final_config['batch']} "
            f"-ctx 16384 -fitt {final_config['fitt']} -ctk {final_config['cache_k']}",
        ]
        text_area.insert("1.0", "\n".join(lines))
        text_area.configure(state="disabled")

        def _apply():
            self.config.threads = final_config["threads"]
            self.config.batch_size = final_config["batch"]
            self.config.micro_batch_size = final_config["fitt"]
            self.config.cache_type_k = final_config["cache_k"]
            # Update Tk variables so the command box refreshes
            try:
                self._tk["threads_val"].set(final_config["threads"])
            except Exception: pass
            try:
                self._tk["batch_size"].set(final_config["batch"])
            except Exception: pass
            try:
                self._tk["micro_batch"].set(final_config["fitt"])
            except Exception: pass
            try:
                self._tk["cache_type_k"].set(final_config["cache_k"])
            except Exception: pass
            win.destroy()

        def _copy():
            import tkinter as tk
            flags = (f"-t {final_config['threads']} -b {final_config['batch']} "
                     f"-ub {final_config['batch']} -ctx 16384 -fitt {final_config['fitt']} "
                     f"-ctk {final_config['cache_k']}")
            self.root.clipboard_clear()
            self.root.clipboard_append(flags)

        btn_frame = ttk.Frame(win)
        btn_frame.pack(fill="x", pady=8)
        ttk.Button(btn_frame, text="Apply Settings", command=_apply).pack(side="right", padx=4)
        ttk.Button(btn_frame, text="Copy Flags", command=_copy).pack(side="right", padx=4)
        ttk.Button(btn_frame, text="Close", command=win.destroy).pack(side="right")



    def _update_command(self):
        """rebuild generated command, update display"""
        cmd = self.config.generate_command()
        if hasattr(self, 'cmd_text'):
            self.cmd_text.configure(state='normal')
            self.cmd_text.delete('1.0', 'end')
            self.cmd_text.insert('1.0', cmd)
            self.cmd_text.configure(state='disabled')

    def _run_optimiser(self):
        """entry point for optimiser, sequential greedy and neighbourhood"""
        from tkinter import messagebox

        # Warning dialog
        msg = (
            "Model Optimisation\n\n"
            "Your system will be under high load during this process.\n"
            "Please save your work and close unnecessary applications.\n"
            "Proceed?"
        )
        if not messagebox.askyesno("Optimise System", msg):
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
        bench_exe = prereq_result["bench_exe"]

        # Run optimiser
        try:
            final_config = self._show_progress_window(model_path, bench_exe)
        except Exception as e:
            messagebox.showerror("Error", f"Optimisation failed:\n{e}")
            return

        if not final_config:
            messagebox.showinfo("Results", "No valid results collected.")
            return
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