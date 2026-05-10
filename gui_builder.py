"""Core GUI builder for llama-server CLI command generation.

Pure stdlib tkinter — no third-party dependencies required.
Uses ttk widgets with the 'clam' theme (available on Windows 10/11).
"""

import os
from tkinter import filedialog, messagebox, Tk, Toplevel, StringVar
import tkinter as tk
from tkinter import ttk


# ---------------------------------------------------------------------------
# Data model & command generation
# ---------------------------------------------------------------------------

class FlagConfig:
    """Holds all mutable flag state and builds the generated CLI string."""

    def __init__(self):
        self.model_path = ""          # set via browse dialog.
        self.low_vram = False         # mutually exclusive with mlock.
        self.mlock = False            # mutually exclusive with low-vram.
        self.ctx_size_enabled = False  # toggle to include --ctx-size in command.
        self.ctx_size_value = 512       # stored ctx-size number when enabled.
        self.n_gpu_layers = -1         # -1 means auto-detect by GPU driver; range 0-99.

        self.host = "0.0.0.0"          # bind all interfaces (local LAN access).
        self.port = 8080               # HTTP server port.
        self.num_threads = os.cpu_count() or 4   # dynamic default from machine CPU count.
        self.threads_enabled = False    # toggle to include -t/--threads in command.

        self.sampling_params_enabled = False  # collapsible section; off by default.
        self.seed = -1                  # -1 means auto (random) seed.
        self.temperature = 0.8          # sampling temperature.
        self.top_k = 40                 # top-k sampling value.
        self.top_p = 0.95               # nucleus sampling probability threshold.
        self.repeat_penalty = 1.1       # repeat penalty factor.

    def generate_command(self):
        """Build the full llama-server CLI command string from current state."""
        parts = ["llama-server"]

        model_path = self.model_path.strip()
        if model_path:
            parts.append(f'--model "{model_path}"')

        # Mutually exclusive Low VRAM / MLock — only one can be active at a time.
        if self.low_vram and not self.mlock:
            parts.append("--low-vram")
        elif self.mlock and not self.low_vram:
            parts.append("--mlock")

        # GPU layers (always included in the generated command).
        gpu_str = "auto" if self.n_gpu_layers == -1 else str(self.n_gpu_layers)
        parts.append(f"--n-gpu-layers {gpu_str}")

        # Context size (only when explicitly enabled by user toggle).
        if self.ctx_size_enabled:
            ctx_val = max(2, min(int(str(self.ctx_size_value)), 999999999))
            parts.append(f"--ctx-size {ctx_val}")

        # Server settings — always included.
        host = str(self.host).strip() or "0.0.0.0"
        port = max(1, min(int(str(self._safe_int("port", 8080))), 65535))
        parts.append(f"--host {host}")
        parts.append(f"--port {port}")

        # Threads (only when explicitly enabled).
        if self.threads_enabled:
            threads = max(1, min(int(str(self._safe_int("threads", 4))), 256))
            parts.append(f"-t {threads}")

        # Sampling params — only included when the collapsible section is expanded.
        if self.sampling_params_enabled:
            seed_val = int(self.seed) if str(self.seed).strip() != "-1" else "auto"
            temp = max(0.05, min(float(str(self.temperature)), 2.0))
            topk = max(1, int(str(self.top_k)))
            topp = min(max(float(str(self.top_p)), 0.05), 1.0)
            rp = min(max(float(str(self.repeat_penalty)), 1.0), 3.0)

            parts.append(f"--seed {seed_val}")
            parts.append(f"--temp {temp:.2f}")
            parts.append(f"--top-k {topk}")
            parts.append(f"--top-p {topp:.3f}")
            parts.append(f"--repeat-penalty {rp:.2f}")

        return " \\\n    ".join(parts)

    @staticmethod
    def _safe_int(name, default):
        """Safely convert a value to int; fall back on *default*."""
        try:
            val = str(default) if isinstance(default, str) else default
            return int(val)
        except (ValueError, TypeError):
            return default


# ---------------------------------------------------------------------------
# UI builder — all ttk widget frames returned as methods.
# Each section method creates a LabelFrame with its widgets and packs it into *parent*.
# Live command generation is triggered by Tk variable traces on every input field.
# ---------------------------------------------------------------------------

class LlamaServerGUI:
    """Main tkinter GUI for configuring llama-server flags and generating commands."""

    def __init__(self, root):
        self.root = root
        self.config = FlagConfig()

        # Create all Tk variables (StringVar / IntVar) so every widget change triggers live updates.
        sv_model_path = tk.StringVar(value="")                    # model path display var.
        iv_auto_gpu = tk.IntVar(value=-1)                         # n-gpu-layers (-1=auto, 0-99).
        lv_bool_low_vram = tk.BooleanVar(value=False)             # low-vram toggle state variable (boolean).
        ml_bool_mlock = tk.BooleanVar(value=False)                # mlock toggle state variable (boolean).

        iv_ctx_enabled = tk.BooleanVar(value=False)               # toggle to show ctx-size input.
        sv_host = tk.StringVar(value="0.0.0.0")                   # server host binding address.
        iv_port = tk.IntVar(value=8080)                           # HTTP port (1-65535).

        iv_threads_enabled = tk.BooleanVar(value=False)           # toggle to show threads input.
        iv_threads_val = tk.IntVar(value=os.cpu_count() or 4)     # thread count default from CPU cores.

        sv_sampling_enabled = tk.BooleanVar(value=False)          # collapsible sampling section visibility flag.
        sv_seed = tk.IntVar(value=-1)                             # seed (-1=auto, any int for fixed seed).
        sv_temp = tk.DoubleVar(value=0.8)                         # temperature (float 0.05–2.0).
        sv_topk = tk.IntVar(value=40)                              # top-k integer value >= 1.
        sv_topp = tk.DoubleVar(value=0.95)                        # top-p float between [0.05, 1.0].
        sv_rp = tk.DoubleVar(value=1.1)                           # repeat penalty (float > 1.0).

        self._vars = {
            "model_path": sv_model_path,
            "n_gpu_layers": iv_auto_gpu,
            "low_vram": lv_bool_low_vram,
            "mlock": ml_bool_mlock,
            "ctx_size_enabled": iv_ctx_enabled,
            "host": sv_host,
            "port": iv_port,
            "threads_enabled": iv_threads_enabled,
            "num_threads": iv_threads_val,
            "sampling_params_enabled": sv_sampling_enabled,
            "seed": sv_seed,
            "temperature": sv_temp,
            "top_k": sv_topk,
            "top_p": sv_topp,
            "repeat_penalty": sv_rp,
        }

        # Store all Tk variables on self for cross-method access.
        self._tk = {
            "model_path": sv_model_path,
            "n_gpu_layers": iv_auto_gpu,
            "low_vram": lv_bool_low_vram,
            "mlock": ml_bool_mlock,
            "ctx_size_enabled": iv_ctx_enabled,
            "host": sv_host,
            "port": iv_port,
            "threads_enabled": iv_threads_enabled,
            "num_threads": iv_threads_val,
            "sampling_params_enabled": sv_sampling_enabled,
            "seed": sv_seed,
            "temperature": sv_temp,
            "top_k": sv_topk,
            "top_p": sv_topp,
            "repeat_penalty": sv_rp,
        }

        # -----------------------------------------------------------------------
        # Change handlers — update config state and trigger command rebuild.
        # -----------------------------------------------------------------------
        def _on_low_vram_change(*_):
            try:
                if lv_bool_low_vram.get():
                    ml_bool_mlock.set(False)
                    self.config.mlock = False
                self.config.low_vram = bool(lv_bool_low_vram.get())
            except Exception:
                pass

        def _on_mlock_change(*_):
            try:
                if ml_bool_mlock.get():
                    lv_bool_low_vram.set(False)
                    self.config.low_vram = False
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

        def _on_host_change(*_):
            self.config.host = sv_host.get() or "0.0.0.0"

        def _on_port_change(*_):
            try:
                val = int(iv_port.get())
                if not (1 <= val <= 65535): return
                self.config.port = max(1, min(val, 65535))
            except (ValueError, TypeError):
                pass

        def _on_threads_enabled_change(*_):
            self.config.threads_enabled = bool(iv_threads_enabled.get())

        def _on_num_threads_change(*_):
            try:
                val = int(iv_threads_val.get())
                if not (1 <= val <= 256): return
                self.config.num_threads = max(1, min(val, 256))
            except (ValueError, TypeError):
                pass





        # Register all traces on the Tk variables so every change triggers live command update.
        lv_bool_low_vram.trace_add("write", lambda *_: (_on_low_vram_change(), self._update_command()))
        ml_bool_mlock.trace_add("write", lambda *_: (_on_mlock_change(), self._update_command()))
        iv_auto_gpu.trace_add("write", lambda *_: (_on_gpu_layers_change(), self._update_command()))

        # Context size toggle and input.
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

        # Threads toggle and input.
        def _threads_toggle_wrapper(*_):
            try:
                if not self.config.threads_enabled:
                    iv_threads_enabled.set(True)
                _on_threads_enabled_change()
                self._update_command()
            except Exception:
                pass
        iv_threads_enabled.trace_add("write", lambda *_: (_threads_toggle_wrapper(),))




        # Build the entire UI.
        self._build_ui()


# ---------------------------------------------------------------------------
# Section builders — each returns a ttk.Frame packed into *parent*.
# Each section method creates a LabelFrame with its widgets and packs it into *parent*.
# Live command generation is triggered by Tk variable traces on every input field.
# ---------------------------------------------------------------------------

    def _build_ui(self):
        """Construct all sections and pack them into a scrollable canvas."""
        root = self.root

        # Use grid so everything expands properly with the window.
        root.grid_rowconfigure(0, weight=1)
        root.grid_columnconfigure(0, weight=1)

        outer_frame = ttk.Frame(root, padding=(6, 4))
        outer_frame.grid(row=0, column=0, sticky="ns")

        # Title label at top of window (bold, larger font).
        title_label = ttk.Label(
            outer_frame, text="llama-server CLI Generator",
            font=("Segoe UI", 16, "bold")
        )
        title_label.grid(row=0, column=0, sticky="ew", pady=(8, 4))

        # Scrollable canvas area for the section frames.
        inner_frame = ttk.Frame(outer_frame)
        inner_frame.grid(row=1, column=0, sticky="nswe")
        outer_frame.columnconfigure(0, weight=1)

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

        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True, pady=(0, 4))
        scrollbar.pack(side="right", fill="y")

        # Bind mousewheel scrolling.
        def _on_mousewheel(event):
            direction = -1 if event.delta > 0 else 1
            self.canvas.yview_scroll(int(1.5 * abs(direction)), "units" if direction < 0 else "pages")

        root.bind_all("<MouseWheel>", lambda e: _on_mousewheel(e))

        # Build each section into the scrollable frame (in order).
        self._section_model_loading(self.scrollable)
        self._section_context_gpu(self.scrollable)
        self._section_server_settings(self.scrollable)
        self._section_sampling_params(self.scrollable)

        # Generated command area at bottom of canvas.
        cmd_frame = ttk.LabelFrame(
            self.scrollable, text="Generated Command", padding=(10, 8)
        )
        cmd_frame.pack(fill="x", padx=6, pady=(4, 2))

        # Frame for the command text + scrollbar.
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

        # Copy button row below text.
        btn_frame = ttk.Frame(cmd_frame)
        btn_frame.pack(fill="x", pady=(4, 2))

        copy_btn = ttk.Button(
            btn_frame, text="\U0001F4CB Copy", command=self._copy_command
        )
        copy_btn.pack(side="right")

        # Initial command render on startup (before any user interaction).
        self._update_command()


    def _section_model_loading(self, parent):
        """Model loading section: browse button + Low VRAM / MLock toggles."""
        frame = ttk.LabelFrame(parent, text="Model Loading", padding=(8, 6))
        frame.pack(fill="x", padx=6, pady=4)

        # Browse row — file dialog to select a .gguf model.
        btn_row = ttk.Frame(frame)
        btn_row.pack(fill="x")

        browse_btn = ttk.Button(btn_row, text="Browse...", command=self._browse_model)
        browse_btn.pack(side="left", padx=(0, 6))

        # Read-only label showing selected path (updates on trace).
        self.model_path_label = tk.Label(
            btn_row, text="(no model selected)", anchor="w", justify="left"
        )
        self.model_path_label.pack(side="left", fill="x", expand=True)

        # Bind trace on model path var to update label.
        sv_model_path = self._tk["model_path"]
        def _update_model_label(*_):
            val = sv_model_path.get() or "(no model selected)"
            self.model_path_label.config(text=val)
            self.config.model_path = val

        sv_model_path.trace_add("write", lambda *_: (_update_model_label(),))

        # Low VRAM / MLock checkboxes side-by-side.
        check_row = ttk.Frame(frame)
        check_row.pack(fill="x", pady=(4, 0))

        lv_bool_low_vram = self._tk["low_vram"]
        ml_bool_mlock = self._tk["mlock"]

        tk.Checkbutton(check_row, text="Low VRAM", variable=lv_bool_low_vram).pack(side="left")
        tk.Checkbutton(check_row, text="MLock", variable=ml_bool_mlock).pack(side="left", padx=(12, 0))


    def _section_context_gpu(self, parent):
        """Context size & GPU layers section (side-by-side columns)."""
        frame = ttk.LabelFrame(parent, text="Performance", padding=(8, 6))
        frame.pack(fill="x", padx=6, pady=4)

        # Left column: Context Size.
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

        # Context toggle checkbox.
        tk.Checkbutton(ctx_frame, text="Enable context size", variable=iv_ctx_enabled).pack(side="left")

        # Entry + label for ctx-size value (shown only when enabled).
        self._ctx_input_widgets = []  # track widgets to pack/unpack.

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

        # Right column: GPU Layers.
        gpu_frame = ttk.Frame(frame)
        gpu_frame.grid(row=0, column=1, sticky="nsew", padx=(8, 0))

        iv_auto_gpu = self._tk["n_gpu_layers"]

        label = ttk.Label(gpu_frame, text="GPU Layers")
        label.pack(side="left")

        # Spinbox for GPU layers (-1 to 99).
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
                spinvar.set(max(-1, min(val, 99)))
            except (ValueError, TypeError):
                pass
        spinvar.trace_add("write", lambda *_: (_spinvar_safe(), self._update_command()))

        ttk.Spinbox(gpu_frame, from_=-1, to=99, textvariable=spinvar, width=5).pack(side="left")


    def _section_server_settings(self, parent):
        """Server settings section (always visible — sensible defaults)."""
        frame = ttk.LabelFrame(parent, text="Network & Server", padding=(8, 6))
        frame.pack(fill="x", padx=6, pady=4)

        # Host / Port row.
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

        port_frame = ttk.Frame(net_row)
        port_frame.pack(side="left", padx=(24, 0))
        ttk.Label(port_frame, text="Port:").pack(side="left", padx=(0, 4))
        entry_p = ttk.Spinbox(port_frame, from_=1, to=65535, textvariable=iv_port, width=8)
        entry_p.pack(side="left")


    def _section_sampling_params(self, parent):
        """Sampling parameters section (collapsible via checkbox)."""
        samp_toggle_var = self._tk["sampling_params_enabled"]

        # Collapsible toggle at top.
        ttk.Checkbutton(parent, text="Show Sampling Parameters", variable=samp_toggle_var).pack(fill="x")

        # Container frame for sampling inputs — shown/hidden by the toggle trace.
        self.samp_frame = samp_frame = ttk.Frame(parent)

        def _on_sampling_params_toggle():
            enabled = bool(samp_toggle_var.get())
            self.config.sampling_params_enabled = enabled
            if enabled:
                # Pack all child widgets back in.
                for w in [self._samp_seed_row, self._samp_temp_row, self._samp_topk_frame, self._samp_topp_frame, self._samp_rp_frame]:
                    w.pack(fill="x", pady=(2, 0))
            else:
                # Hide all child widgets.
                for w in [self._samp_seed_row, self._samp_temp_row, self._samp_topk_frame, self._samp_topp_frame, self._samp_rp_frame]:
                    w.pack_forget()

        def _sampling_trace():
            self.config.sampling_params_enabled = bool(samp_toggle_var.get())

        def _samp_wrapper(*_):
            try:
                enabled = bool(samp_toggle_var.get())
                self.config.sampling_params_enabled = enabled
                if enabled:
                    for w in [self._samp_seed_row, self._samp_temp_row, self._samp_topk_frame, self._samp_topp_frame, self._samp_rp_frame]:
                        w.pack(fill="x", pady=(2, 0))
                else:
                    for w in [self._samp_seed_row, self._samp_temp_row, self._samp_topk_frame, self._samp_topp_frame, self._samp_rp_frame]:
                        w.pack_forget()
                self._update_command()
            except Exception:
                pass
        def _on_sampling_trace(*_):
            try:
                enabled = bool(samp_toggle_var.get())
                self.config.sampling_params_enabled = enabled
                if enabled and hasattr(self, '_update_command'):
                    for w in [self._samp_seed_row, self._samp_temp_row, self._samp_topk_frame, self._samp_topp_frame, self._samp_rp_frame]:
                        try: w.pack(fill="x", pady=(2, 0))
                        except Exception: pass
                elif not enabled and hasattr(self, '_update_command'):
                    for w in [self._samp_seed_row, self._samp_temp_row, self._samp_topk_frame, self._samp_topp_frame, self._samp_rp_frame]:
                        try: w.pack_forget()
                        except Exception: pass
                if hasattr(self, '_update_command'):
                    self._update_command()
            except Exception:
                pass
        samp_toggle_var.trace_add("write", lambda *_: (_on_sampling_trace(),))

        # --- Seed row ---
        sv_seed = tk.IntVar(value=-1)  # -1=auto.
        seed_row = ttk.Frame(samp_frame)
        self._samp_seed_row = seed_row
        ttk.Label(seed_row, text="Seed").pack(side="left")
        entry_s = ttk.Spinbox(seed_row, from_=-99999, to=99999, width=8, textvariable=sv_seed)
        entry_s.pack(side="left", padx=(4, 0))

        def _seed_safe(*_):
            try:
                val = int(sv_seed.get())
                if not (-99999 <= val <= 99999): return
                self.config.seed = max(-1, min(val, 99999))
                sv_seed.set(max(-1, min(val, 99999)))
            except (ValueError, TypeError):
                pass
        def _seed_trace(*_):
            try: self._update_command()
            except Exception:
                pass
        sv_seed.trace_add("write", lambda *_: (_seed_safe(), _seed_trace()))

        # --- Temperature row ---
        def make_temp_frame():
            temp_row = ttk.Frame(samp_frame)
            self._samp_temp_row = temp_row
            tk.Label(temp_row, text="Temperature").pack(side="left")
            sv_temp = tk.DoubleVar(value=0.8)

            # Scale slider (visual).
            scale_var = tk.DoubleVar(value=0.8)

            def _temp_safe(*_):
                try:
                    val = float(scale_var.get())
                    if not 0.05 <= val <= 2.0: return
                    sv_temp.set(max(0.05, min(val, 2.0)))
                    self.config.temperature = max(0.05, min(val, 2.0))
                except Exception:
                    pass
            def _temp_entry_safe(*_):
                try:
                    val = float(sv_temp.get()) if sv_temp.get() else 0.8
                    scale_var.set(max(0.05, min(val, 2.0)))
                    self.config.temperature = max(0.05, min(val, 2.0))
                except Exception:
                    pass
            def _temp_cmd(*_):
                try: self._update_command()
                except Exception:
                    pass
            scale_var.trace_add("write", lambda *_: (_temp_safe(), _temp_entry_safe()))
            sv_temp.trace_add("write", lambda *_: (_temp_entry_safe(),))

        make_temp_frame()

        # --- Top-K row ---
        def make_topk_frame():
            topk_row = ttk.Frame(samp_frame)
            self._samp_topk_frame = topk_row
            tk.Label(topk_row, text="Top-K").pack(side="left")
            sv_topk = tk.IntVar(value=40)

            entry_k = ttk.Spinbox(topk_row, from_=1, to=9999, width=8, textvariable=sv_topk)
            entry_k.pack(side="left", padx=(4, 0))

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

        make_topk_frame()

        # --- Top-P row ---
        def make_topp_frame():
            topp_row = ttk.Frame(samp_frame)
            self._samp_topp_frame = topp_row
            tk.Label(topp_row, text="Top-P").pack(side="left")
            sv_topp = tk.DoubleVar(value=0.95)

            scale_var = tk.DoubleVar(value=0.95)

            def _topp_safe(*_):
                try:
                    val = float(scale_var.get()) if scale_var.get() else 0.95
                    self.config.top_p = min(max(val, 0.05), 1.0)
                    sv_topp.set(min(max(val, 0.05), 1.0))
                except Exception:
                    pass
            def _topp_entry_safe(*_):
                try:
                    val = float(sv_topp.get()) if sv_topp.get() else 0.95
                    scale_var.set(min(max(val, 0.05), 1.0))
                    self.config.top_p = min(max(val, 0.05), 1.0)
                except Exception:
                    pass
            def _topp_cmd(*_):
                try: self._update_command()
                except Exception:
                    pass
            scale_var.trace_add("write", lambda *_: (_topp_safe(),))
            sv_topp.trace_add("write", lambda *_: (_topp_entry_safe()))

        make_topp_frame()

        # --- Repeat Penalty row ---
        def make_rp_frame():
            rp_row = ttk.Frame(samp_frame)
            self._samp_rp_frame = rp_row
            tk.Label(rp_row, text="Repeat Pen.").pack(side="left")
            sv_rp = tk.DoubleVar(value=1.1)

            scale_var = tk.DoubleVar(value=1.1)

            def _rp_safe(*_):
                try:
                    val = float(scale_var.get()) if scale_var.get() else 1.1
                    self.config.repeat_penalty = min(max(val, 1.0), 3.0)
                    sv_rp.set(min(max(val, 1.0), 3.0))
                except Exception:
                    pass
            def _rp_entry_safe(*_):
                try:
                    val = float(sv_rp.get()) if sv_rp.get() else 1.1
                    scale_var.set(min(max(val, 1.0), 3.0))
                    self.config.repeat_penalty = min(max(val, 1.0), 3.0)
                except Exception:
                    pass
            def _rp_cmd(*_):
                try: self._update_command()
                except Exception:
                    pass
            scale_var.trace_add("write", lambda *_: (_rp_safe(),))
            sv_rp.trace_add("write", lambda *_: (_rp_entry_safe()))

        make_rp_frame()


# ---------------------------------------------------------------------------
# Event handlers & helpers — user interactions and command generation logic.
# Each section method creates a LabelFrame with its widgets and packs it into *parent*.
# Live command generation is triggered by Tk variable traces on every input field.
# ---------------------------------------------------------------------------

    def _browse_model(self):
        """Open file dialog to select a .gguf model file."""
        path = filedialog.askopenfilename(
            title="Select GGUF Model File",
            filetypes=[("GGUF files", "*.gguf"), ("All files", "*.*")],
            initialdir=os.path.expanduser("~"),  # start in user's home directory.
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
        """Copy current command to clipboard with confirmation dialog."""
        cmd = self.config.generate_command()
        self.root.clipboard_clear()
        self.root.clipboard_append(cmd)



