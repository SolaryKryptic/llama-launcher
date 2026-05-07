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
            ctx_val = max(2, min(int(str(self._safe_int("ctx", 512))), 8192))
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
            if lv_bool_low_vram.get():
                ml_bool_mlock.set(False)
                self.config.mlock = False
            self.config.low_vram = bool(lv_bool_low_vram.get())

        def _on_mlock_change(*_):
            if ml_bool_mlock.get():
                lv_bool_low_vram.set(False)
                self.config.low_vram = False
            self.config.mlock = bool(ml_bool_mlock.get())

        def _on_model_change(*_):
            self.config.model_path = sv_model_path.get()

        def _on_gpu_layers_change(*_):
            val = iv_auto_gpu.get()
            if -1 <= val <= 99:
                self.config.n_gpu_layers = val

        def _on_ctx_enabled_change(*_):
            self.config.ctx_size_enabled = bool(iv_ctx_enabled.get())

        def _on_host_change(*_):
            self.config.host = sv_host.get() or "0.0.0.0"

        def _on_port_change(*_):
            val = max(1, min(int(str(iv_port.get())), 65535)) if iv_port.get() else 8080
            self.config.port = val

        def _on_threads_enabled_change(*_):
            self.config.threads_enabled = bool(iv_threads_enabled.get())

        def _on_num_threads_change(*_):
            val = max(1, min(int(str(iv_threads_val.get())), 256)) if iv_threads_val.get() else os.cpu_count() or 4
            self.config.num_threads = val





        # Register all traces on the Tk variables so every change triggers live command update.
        lv_bool_low_vram.trace_add("write", lambda *_: (_on_low_vram_change(), _update_command()))
        ml_bool_mlock.trace_add("write", lambda *_: (_on_mlock_change(), _update_command()))
        iv_auto_gpu.trace_add("write", lambda *_: (_on_gpu_layers_change(), _update_command()))

        # Context size toggle and input.
        def _ctx_toggle_wrapper(*_):
            if not self.config.ctx_size_enabled:
                iv_ctx_enabled.set(True)
            _on_ctx_enabled_change()
            _update_command()
        iv_ctx_enabled.trace_add("write", lambda *_: (_ctx_toggle_wrapper(),))

        sv_host.trace_add("write", lambda *_: (_on_host_change(), _update_command()))
        iv_port.trace_add("write", lambda *_: (_on_port_change(), _update_command()))

        # Threads toggle and input.
        def _threads_toggle_wrapper(*_):
            if not self.config.threads_enabled:
                iv_threads_enabled.set(True)
            _on_threads_enabled_change()
            _update_command()
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

        # Title label at top of window (bold, larger font).
        title_label = ttk.Label(
            root, text="llama-server CLI Generator",
            font=("Segoe UI", 16, "bold"), padding=(20, 15, 20, 8)
        )
        title_label.pack(fill="x")

        # Scrollable canvas area for the section frames.
        main_frame = ttk.Frame(root)
        main_frame.pack(fill="both", expand=True, padx=6, pady=(0, 4))

        canvas = tk.Canvas(main_frame, highlightthickness=0)
        scrollbar = ttk.Scrollbar(
            main_frame, orient="vertical", command=canvas.yview
        )
        self.scrollable = ttk.Frame(canvas)
        scroll_content_id = canvas.create_window((0, 0), window=self.scrollable, anchor="nw")

        def _on_scroll_configure(event):
            """Update the scroll region whenever section frames change size."""
            canvas.configure(scrollregion=canvas.bbox("all"))

        self.scrollable.bind("<Configure>", lambda e: (_update_content_width(e), _on_scroll_configure(e)))

        # Helper to keep canvas width in sync with content area.
        def _update_content_width(event):
            """Resize the scrollable frame to match parent canvas width."""
            canvas.itemconfigure(scroll_content_id, width=event.width)

        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Bind mousewheel scrolling to the canvas only (not whole window).
        def _on_mousewheel(event):
            direction = -1 if event.delta > 0 else 1
            canvas.yview_scroll(int(1.5 * abs(direction)), "units" if direction < 0 else "pages")

        root.bind_all("<MouseWheel>", lambda e: (_update_content_width(e), _on_mousewheel(e)))

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
            val = max(2, min(int(iv_ctx_var.get()), 8192)) if iv_ctx_var.get() else 512
            iv_ctx_var.set(val)
            self.config.ctx_size_enabled = bool(iv_ctx_enabled.get())

        ctx_label = ttk.Label(ctx_frame, text="Ctx Size")
        entry_c = ttk.Entry(ctx_frame, textvariable=iv_ctx_var, width=8)
        for w in [ctx_label, entry_c]:
            self._ctx_input_widgets.append(w)

        def _ctx_trace_wrapper(*_):
            self.config.ctx_size_enabled = bool(iv_ctx_enabled.get())

        iv_ctx_enabled.trace_add("write", lambda *_: (_on_ctx_change(), _ctx_trace_wrapper(), _update_command()))
        def _iv_ctx_var_trace(*_):
            self.config.ctx_size_enabled = bool(iv_ctx_enabled.get())
            _on_ctx_val()
            _update_command()

        iv_ctx_var.trace_add("write", lambda *_: (_iv_ctx_var_trace(),))

        # Right column: GPU Layers.
        gpu_frame = ttk.Frame(frame)
        gpu_frame.grid(row=0, column=1, sticky="nsew", padx=(8, 0))

        iv_auto_gpu = self._tk["n_gpu_layers"]

        label = ttk.Label(gpu_frame, text="GPU Layers")
        label.pack(side="left")

        # Spinbox for GPU layers (-1 to 99).
        spinvar = tk.IntVar(value=-1)

        def _on_spinval(*_):
            val = max(-1, min(int(spinvar.get()), 99)) if spinvar.get() else -1
            spinvar.set(val)
            self.config.n_gpu_layers = val

        iv_auto_gpu.trace_add("write", lambda *_: (spinvar.set(iv_auto_gpu.get()), _on_spinval(), _update_command()))
        spinvar.trace_add("write", lambda *_: (_on_spinval(),))

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

        samp_toggle_var.trace_add(
            "write",
            lambda *_: (_on_sampling_params_toggle(), _sampling_trace(), _update_command()),
        )

        # --- Seed row ---
        sv_seed = tk.IntVar(value=-1)  # -1=auto.
        seed_row = ttk.Frame(samp_frame)
        self._samp_seed_row = seed_row
        ttk.Label(seed_row, text="Seed").pack(side="left")
        entry_s = ttk.Spinbox(seed_row, from_=-99999, to=99999, width=8, textvariable=sv_seed)
        entry_s.pack(side="left", padx=(4, 0))

        def _on_seed_change(*_):
            val = max(-1, int(sv_seed.get())) if sv_seed.get() else -1
            sv_seed.set(val)
            self.config.seed = val

        sv_seed.trace_add("write", lambda *_: (_on_seed_change(), _update_command()))

        # --- Temperature row ---
        def make_temp_frame():
            temp_row = ttk.Frame(samp_frame)
            self._samp_temp_row = temp_row
            tk.Label(temp_row, text="Temperature").pack(side="left")
            sv_temp = tk.DoubleVar(value=0.8)

            # Scale slider (visual).
            scale_var = tk.DoubleVar(value=0.8)

            def _on_scale(*_):
                val = max(0.05, min(float(scale_var.get()), 2.0))
                sv_temp.set(val)
                self.config.temperature = val

            def _on_entry(*_):
                val = max(0.05, min(float(sv_temp.get()), 2.0)) if sv_temp.get() else 0.8
                scale_var.set(val)
                self.config.temperature = val

            tk.Scale(temp_row, from_=0.05, to=2.0, orient="horizontal", length=140, resolution=0.05, variable=scale_var).pack(side="left")
            entry_t = ttk.Entry(temp_row, textvariable=sv_temp, width=6)
            entry_t.pack(side="left", padx=(4, 0))

            scale_var.trace_add("write", lambda *_: (_on_scale(), _update_command()))
            sv_temp.trace_add("write", lambda *_: (scale_var.set(sv_temp.get()), _on_entry(), _update_command()))

        make_temp_frame()

        # --- Top-K row ---
        def make_topk_frame():
            topk_row = ttk.Frame(samp_frame)
            self._samp_topk_frame = topk_row
            tk.Label(topk_row, text="Top-K").pack(side="left")
            sv_topk = tk.IntVar(value=40)

            entry_k = ttk.Spinbox(topk_row, from_=1, to=9999, width=8, textvariable=sv_topk)
            entry_k.pack(side="left", padx=(4, 0))

            def _on_topk_change(*_):
                val = max(1, int(sv_topk.get())) if sv_topk.get() else 40
                sv_topk.set(val)
                self.config.top_k = val

            sv_topk.trace_add("write", lambda *_: (_on_topk_change(), _update_command()))

        make_topk_frame()

        # --- Top-P row ---
        def make_topp_frame():
            topp_row = ttk.Frame(samp_frame)
            self._samp_topp_frame = topp_row
            tk.Label(topp_row, text="Top-P").pack(side="left")
            sv_topp = tk.DoubleVar(value=0.95)

            scale_var = tk.DoubleVar(value=0.95)

            def _on_scale(*_):
                val = min(max(float(scale_var.get()), 0.05), 1.0)
                sv_topp.set(val)
                self.config.top_p = val

            def _on_entry(*_):
                val = min(max(float(sv_topp.get()), 0.05), 1.0) if sv_topp.get() else 0.95
                scale_var.set(val)
                self.config.top_p = val

            tk.Scale(topp_row, from_=0.05, to=1.0, orient="horizontal", length=140, resolution=0.01, variable=scale_var).pack(side="left")
            entry_tp = ttk.Entry(topp_row, textvariable=sv_topp, width=6)
            entry_tp.pack(side="left", padx=(4, 0))

            scale_var.trace_add("write", lambda *_: (_on_scale(), _update_command()))
            sv_topp.trace_add("write", lambda *_: (scale_var.set(sv_topp.get()), _on_entry(), _update_command()))

        make_topp_frame()

        # --- Repeat Penalty row ---
        def make_rp_frame():
            rp_row = ttk.Frame(samp_frame)
            self._samp_rp_frame = rp_row
            tk.Label(rp_row, text="Repeat Pen.").pack(side="left")
            sv_rp = tk.DoubleVar(value=1.1)

            scale_var = tk.DoubleVar(value=1.1)

            def _on_scale(*_):
                val = min(max(float(scale_var.get()), 1.0), 3.0)
                sv_rp.set(val)
                self.config.repeat_penalty = val

            def _on_entry(*_):
                val = min(max(float(sv_rp.get()), 1.0), 3.0) if sv_rp.get() else 1.1
                scale_var.set(val)
                self.config.repeat_penalty = val

            tk.Scale(rp_row, from_=1.0, to=3.0, orient="horizontal", length=140, resolution=0.05, variable=scale_var).pack(side="left")
            entry_rp = ttk.Entry(rp_row, textvariable=sv_rp, width=6)
            entry_rp.pack(side="left", padx=(4, 0))

            scale_var.trace_add("write", lambda *_: (_on_scale(), _update_command()))
            sv_rp.trace_add("write", lambda *_: (scale_var.set(sv_rp.get()), _on_entry(), _update_command()))

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


if __name__ == "__main__":
    root = Tk()
    gui = LlamaServerGUI(root)
    root.title("llama-server CLI Generator")
    root.geometry("720x680")
    root.mainloop()
