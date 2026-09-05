"""
Theme constants for llama-launcher UI.
Centralizes fonts, colors, padding, and styling for easy theming.
"""

# Fonts
FONT_FAMILY = "Segoe UI"
FONT_MONO = "Consolas"
FONT_SIZE_NORMAL = 9
FONT_SIZE_SMALL = 8
FONT_SIZE_TITLE = 16
FONT_SIZE_SECTION = 10
FONT_WEIGHT_NORMAL = "normal"
FONT_WEIGHT_BOLD = "bold"

# Padding & spacing
PAD_X = 6
PAD_Y = 4
PAD_SMALL = 2
PAD_MEDIUM = 8
PAD_LARGE = 12
SECTION_PADDING = (8, 6)
BUTTON_PADX = 4
ENTRY_WIDTH_SMALL = 8
ENTRY_WIDTH_MEDIUM = 12
ENTRY_WIDTH_LARGE = 35

# Colors (Light theme)
LIGHT_BG = "#f5f5f5"
LIGHT_FG = "#1a1a1a"
LIGHT_SECONDARY_FG = "#666666"
LIGHT_ACCENT = "#0078d4"
LIGHT_ACCENT_HOVER = "#106ebe"
LIGHT_BORDER = "#d0d0d0"
LIGHT_TOOLTIP_BG = "#ffffe0"
LIGHT_TOOLTIP_FG = "#1a1a1a"
LIGHT_WARNING = "#cc3300"
LIGHT_SUCCESS = "#27ae60"
LIGHT_ERROR = "#cc0000"
LIGHT_DISABLED_BG = "#e0e0e0"
LIGHT_DISABLED_FG = "#999999"

# Colors (Dark theme)
DARK_BG = "#1e1e1e"
DARK_FG = "#e0e0e0"
DARK_SECONDARY_FG = "#aaaaaa"
DARK_ACCENT = "#3a96dd"
DARK_ACCENT_HOVER = "#5aa8e0"
DARK_BORDER = "#404040"
DARK_TOOLTIP_BG = "#333333"
DARK_TOOLTIP_FG = "#e0e0e0"
DARK_WARNING = "#ff6b4a"
DARK_SUCCESS = "#4ade80"
DARK_ERROR = "#f87171"
DARK_DISABLED_BG = "#3a3a3a"
DARK_DISABLED_FG = "#777777"

# Widget specific
SCROLLBAR_WIDTH = 12
PROGRESS_BAR_THICKNESS = 20
COMMAND_PREVIEW_HEIGHT = 5
COMMAND_PREVIEW_MIN_LINES = 5

# Tooltip
TOOLTIP_DELAY_MS = 500
TOOLTIP_WRAP_LENGTH = 300

# Window defaults
DEFAULT_WINDOW_WIDTH = 764
DEFAULT_WINDOW_HEIGHT = 693
MIN_WINDOW_WIDTH = 700
MIN_WINDOW_HEIGHT = 500

# Section titles
SECTION_TITLES = [
    "Hardware Info",
    "Model Loading",
    "Context & GPU",
    "Performance",
    "Speculative Decoding",
    "Network & Server",
    "Sampling Parameters",
]

# Cache type options
CACHE_TYPES = ["f16", "q8_0", "q5_0", "q4_0"]

# Speculative decoding types
SPEC_TYPES = ["ngram-mod", "draft-mtp"]

# Tooltip text map (widget key -> tooltip text)
TOOLTIPS = {
    "model_path": "Path to the .gguf model file to load",
    "browse_model": "Open file dialog to select a .gguf model",
    "find_model": "Open Hugging Face model hub in browser",
    "no_mmap": "Disable memory-mapped file I/O (useful for network drives)",
    "mlock": "Lock model pages in RAM to prevent swapping to disk",
    "no_warmup": "Skip the initial warmup inference run",
    "mmproj_enabled": "Enable multimodal projector (for vision models like LLaVA)",
    "mmproj_path": "Path to the mmproj file for multimodal models",
    "ctx_size_value": "Context window size in tokens (2 to 999,999,999)",
    "ctx_size_enabled": "Toggle to show/hide context size input",
    "n_gpu_layers": "Number of layers to offload to GPU (-1 = auto, 0 = CPU only)",
    "flash_attention": "Enable Flash Attention for faster attention computation",
    "fit_on": "Enable --fit to automatically fit model to GPU VRAM",
    "fitt": "Target size in MB for --fit (1-65536)",
    "batch_size": "Batch size for KV cache (1-8192)",
    "micro_batch_size": "Micro batch size for memory splitting (1-8192)",
    "threads": "Number of CPU threads (-1 = auto, up to 128)",
    "thread_batch": "Thread batch size for parallel processing (-1 = unset)",
    "cache_type_k": "KV cache quantization type for K (key) cache",
    "cache_type_v": "KV cache quantization type for V (value) cache",
    "cache_type_kd": "KV cache type for draft model K cache (draft-mtp)",
    "cache_type_vd": "KV cache type for draft model V cache (draft-mtp)",
    "spec_enabled": "Enable speculative decoding",
    "spec_type": "Speculative decoding strategy",
    "spec_draft_n_max": "Maximum draft tokens to generate (0 = unset)",
    "spec_draft_n_min": "Minimum draft tokens (0 = unset)",
    "spec_draft_p_min": "Minimum probability for draft acceptance (0 = unset)",
    "draft_model_path": "Path to draft model for draft-mtp speculation",
    "host": "Server bind address (0.0.0.0 = all interfaces)",
    "port": "HTTP server port (1-65535)",
    "cache_ram": "System RAM to reserve for cache in MB (0 = unlimited)",
    "parallel": "Number of parallel decoding streams (-np)",
    "threads_enabled": "Override automatic thread detection",
    "num_threads": "Manual thread count when override enabled",
    "temperature": "Sampling temperature (0.05-2.0, higher = more random)",
    "min_p": "Minimum probability threshold (-1.0 to 1.0)",
    "top_k": "Top-K sampling (1-9999, 0 = disabled)",
    "top_p": "Nucleus sampling threshold (0.05-1.0)",
    "repeat_penalty": "Repetition penalty (1.0-3.0, >1 reduces repeats)",
    "presence_penalty": "Presence penalty (-2.0 to 2.0, negative penalizes)",
    "copy_command": "Copy the generated command to clipboard",
    "run_command": "Launch command in a new CMD window",
    "save_bat": "Save command as a .bat batch file",
    "optimise": "Run Bayesian optimisation to find best flags",
    "perplexity_file": "Text file used for perplexity validation",
    "ppl_threshold": "Max PPL degradation allowed (%)",
    "metric_weight": "Weight for throughput vs latency in scoring",
    "dark_mode": "Switch to dark theme",
    "light_mode": "Switch to light theme",
}


def get_colors(dark_mode=False):
    """Return color dict for the current theme."""
    if dark_mode:
        return {
            "bg": DARK_BG,
            "fg": DARK_FG,
            "secondary_fg": DARK_SECONDARY_FG,
            "accent": DARK_ACCENT,
            "accent_hover": DARK_ACCENT_HOVER,
            "border": DARK_BORDER,
            "tooltip_bg": DARK_TOOLTIP_BG,
            "tooltip_fg": DARK_TOOLTIP_FG,
            "warning": DARK_WARNING,
            "success": DARK_SUCCESS,
            "error": DARK_ERROR,
            "disabled_bg": DARK_DISABLED_BG,
            "disabled_fg": DARK_DISABLED_FG,
        }
    return {
        "bg": LIGHT_BG,
        "fg": LIGHT_FG,
        "secondary_fg": LIGHT_SECONDARY_FG,
        "accent": LIGHT_ACCENT,
        "accent_hover": LIGHT_ACCENT_HOVER,
        "border": LIGHT_BORDER,
        "tooltip_bg": LIGHT_TOOLTIP_BG,
        "tooltip_fg": LIGHT_TOOLTIP_FG,
        "warning": LIGHT_WARNING,
        "success": LIGHT_SUCCESS,
        "error": LIGHT_ERROR,
        "disabled_bg": LIGHT_DISABLED_BG,
        "disabled_fg": LIGHT_DISABLED_FG,
    }


def get_font(size=FONT_SIZE_NORMAL, weight=FONT_WEIGHT_NORMAL, mono=False):
    """Return a font tuple."""
    family = FONT_MONO if mono else FONT_FAMILY
    return (family, size, weight)


# --- Theme application ---
def apply_theme(root, dark_mode=False):
    """Apply theme colors to all ttk styles and tk widgets in the widget tree."""
    colors = get_colors(dark_mode)
    style = ttk.Style(root)
    
    # Configure ttk styles
    style.configure(".", background=colors["bg"], foreground=colors["fg"])
    style.configure("TFrame", background=colors["bg"])
    style.configure("TLabel", background=colors["bg"], foreground=colors["fg"])
    style.configure("TButton", background=colors["bg"], foreground=colors["fg"])
    style.configure("TEntry", fieldbackground=colors["bg"], foreground=colors["fg"])
    style.configure("TSpinbox", fieldbackground=colors["bg"], foreground=colors["fg"])
    style.configure("TCombobox", fieldbackground=colors["bg"], foreground=colors["fg"])
    style.configure("TLabelframe", background=colors["bg"], foreground=colors["fg"])
    style.configure("TLabelframe.Label", background=colors["bg"], foreground=colors["fg"])
    style.configure("TScrollbar", background=colors["bg"], troughcolor=colors["border"])
    style.configure("TCheckbutton", background=colors["bg"], foreground=colors["fg"])
    style.configure("TRadiobutton", background=colors["bg"], foreground=colors["fg"])
    style.configure("TProgressbar", background=colors["accent"], troughcolor=colors["border"])
    
    # Map states for buttons
    style.map("TButton",
        background=[("active", colors["accent_hover"]), ("disabled", colors["disabled_bg"])],
        foreground=[("disabled", colors["disabled_fg"])]
    )
    style.map("TEntry",
        fieldbackground=[("disabled", colors["disabled_bg"])],
        foreground=[("disabled", colors["disabled_fg"])]
    )
    style.map("TSpinbox",
        fieldbackground=[("disabled", colors["disabled_bg"])],
        foreground=[("disabled", colors["disabled_fg"])]
    )
    style.map("TCombobox",
        fieldbackground=[("disabled", colors["disabled_bg"])],
        foreground=[("disabled", colors["disabled_fg"])]
    )
    style.map("TCheckbutton",
        background=[("active", colors["accent_hover"])],
        foreground=[("disabled", colors["disabled_fg"])]
    )
    
    # Recursively apply to all tk widgets
    def _apply_to_widget(widget):
        try:
            wclass = widget.winfo_class()
            if wclass in ("Frame", "Labelframe", "Toplevel"):
                widget.configure(background=colors["bg"])
            elif wclass in ("Label", "Button", "Checkbutton", "Radiobutton"):
                widget.configure(background=colors["bg"], foreground=colors["fg"])
            elif wclass in ("Entry", "Spinbox", "Text"):
                widget.configure(background=colors["bg"], foreground=colors["fg"],
                               insertbackground=colors["fg"],
                               selectbackground=colors["accent"],
                               selectforeground=colors["fg"])
            elif wclass == "Canvas":
                widget.configure(background=colors["bg"])
            elif wclass == "Scrollbar":
                widget.configure(background=colors["bg"], troughcolor=colors["border"])
        except Exception:
            pass
        for child in widget.winfo_children():
            _apply_to_widget(child)
    
    _apply_to_widget(root)
    
    # Force update
    root.update_idletasks()
    
    return colors