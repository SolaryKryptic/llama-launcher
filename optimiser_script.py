import subprocess
import re
import os
import socket
import time
import tempfile
import requests

_NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0

BENCH_PORT = 19876  # Fixed port for optimiser server instances

PP_PROMPT = (
    "The history of artificial intelligence is a long and complex story that spans decades of research, "
    "failure, and breakthrough. Early pioneers like Alan Turing and John McCarthy laid the theoretical "
    "foundations for machines that could think, reason, and learn. The field went through multiple cycles "
    "of hype and disappointment, known as AI winters, before the deep learning revolution of the 2010s "
    "transformed what was possible. Neural networks, once considered too slow and data-hungry to be "
    "practical, became the dominant paradigm as compute and data availability scaled dramatically. "
    "The introduction of the transformer architecture in 2017 was a pivotal moment, enabling models to "
    "process long sequences of text with unprecedented effectiveness. Large language models trained on "
    "vast corpora of internet text demonstrated emergent capabilities that surprised even their creators. "
    "Tasks like translation, summarisation, code generation, and open-ended reasoning became tractable "
    "for the first time. The release of GPT-3, followed by instruction-tuned variants, showed that "
    "a single model could generalise across an enormous range of tasks with minimal fine-tuning. "
    "This sparked a race among technology companies and research institutions to build ever-larger and "
    "more capable systems. Questions around alignment, safety, and the societal impact of powerful AI "
    "systems moved from the fringes of academic discourse to the centre of public debate. Governments "
    "began drafting legislation, researchers published alignment roadmaps, and the pace of capability "
    "gains showed no sign of slowing. The challenge of building systems that are both highly capable "
    "and reliably aligned with human values remains one of the most important open problems in the field."
)
TG_N_PREDICT = 128
PERPLEXITY_FILE = "Moby Dick.txt"
PPL_THRESHOLD = 0.03


# ==========================================
# PURE UTILITY FUNCTIONS (importable)
# ==========================================

def build_thread_list():
    """Build thread sweep list based on detected CPU count.
    If CPU count cannot be detected, threads list is empty and thread sweep is skipped
    
    Returns:
        threads: up to 75% of max threads (for -t flag)
        thread_batch: up to 100% of max threads (for -tb flag, thread batching)
        micro_batch_sizes: independent from batch sizes
    """
    max_threads = os.cpu_count()
    if max_threads is not None:
        cap_limit_75 = max(1, int(max_threads * 0.75))
        cap_limit_100 = max_threads
        step_size = max(1, int(max_threads * 0.25))
        threads_list = [t for t in range(step_size, cap_limit_75 + 1, step_size)]
        if cap_limit_75 not in threads_list:
            threads_list.append(cap_limit_75)
        
        # Thread batch can use up to 100% of threads
        thread_batch_list = [t for t in range(step_size, cap_limit_100 + 1, step_size)]
        if cap_limit_100 not in thread_batch_list:
            thread_batch_list.append(cap_limit_100)
    else:
        threads_list = []
        thread_batch_list = []
        cap_limit_75 = 0
        cap_limit_100 = 0
    return {
        "threads": threads_list,
        "thread_batch": thread_batch_list,
        "batch_sizes": [128, 256, 512, 1024, 2048],
        "micro_batch_sizes": [128, 256, 512, 1024, 2048],
        "fitt_targets": [50],
        "cache_k_types": ["f16", "q8_0", "q5_0", "q4_0"],
        "cache_v_types": ["f16", "q8_0", "q5_0", "q4_0"],
        "spec_draft_n": list(range(1, 5)),  # 1-4 for MTP speculative decoding
        "max_threads": max_threads,
        "cap_limit": cap_limit_75,
    }


def calculate_score(pp, tg, metric_weight=0.5):
    if pp is None or tg is None:
        return -1.0
    return (pp * metric_weight) + (tg * (1.0 - metric_weight))


def _project_file(path):
    if not path or os.path.isabs(path):
        return path
    from optimisation_service import get_exe_dir
    return os.path.join(get_exe_dir(), path)


def _load_perplexity_corpus(corpus_file):
    try:
        with open(_project_file(corpus_file or PERPLEXITY_FILE), "r", encoding="utf-8") as f:
            text = f.read().strip()
        return text if text else PP_PROMPT
    except Exception:
        return PP_PROMPT


def _write_temp_corpus(text):
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8")
    try:
        f.write(text)
        f.flush()
        return f.name
    finally:
        f.close()


def parse_perplexity(output):
    match = re.search(r"PPL\s*=\s*([0-9]+(?:\.[0-9]+)?)", output or "")
    return float(match.group(1)) if match else None


def _is_oom_error(stderr):
    text = (stderr or "").lower()
    return any(token in text for token in ("out of memory", "cudamalloc", "cuda malloc", "memory allocation", "failed to allocate", "oom"))


def build_perplexity_base_flags(context_size):
    context_size = int(context_size)
    return [
        "-fit", "on",
        "-fitc", str(context_size),
        "-c", str(context_size),
        "-fitt", "50",
        "-s", str(context_size // 8),
        "--chunks", "3",
    ]


def run_perplexity(model_path, perplexity_exe, context_size, flags=None, corpus_file=PERPLEXITY_FILE, timeout=None):
    corpus_path = _write_temp_corpus(_load_perplexity_corpus(corpus_file))
    try:
        cmd = (
            [perplexity_exe, "-m", model_path, "-f", corpus_path]
            + build_perplexity_base_flags(context_size)
            + list(flags or [])
        )
        result = subprocess.run(cmd, capture_output=True, text=True, creationflags=_NO_WINDOW, timeout=timeout)
        output = (result.stdout or "") + "\n" + (result.stderr or "")
        return parse_perplexity(output), result.returncode, output
    finally:
        try:
            os.remove(corpus_path)
        except Exception:
            pass


def run_perplexity_baseline(model_path, perplexity_exe, context_size, corpus_file=PERPLEXITY_FILE, timeout=None, spec_active=False):
    ppl, code, stderr = run_perplexity(model_path, perplexity_exe, context_size, corpus_file=corpus_file, timeout=timeout)
    if code == 0 and ppl is not None:
        print(f"[DEBUG] Baseline perplexity parsed: PPL={ppl:.4f} (f16 baseline).")
        return ppl, [], False
    if code == 0 and ppl is None:
        print("[WARN] Baseline perplexity completed but PPL could not be parsed.")
        return None, [], False
    if not _is_oom_error(stderr):
        preview = (stderr or "").strip().replace("\r", " ").replace("\n", " ")[:1200]
        print(f"[WARN] Baseline perplexity failed (exit {code}); cache quantisation quality gate will be skipped. {preview}")
        return None, [], False

    q8_flags = build_perplexity_cache_flags("q8_0", "q8_0", spec_active=False)
    ppl, code, stderr = run_perplexity(model_path, perplexity_exe, context_size, flags=q8_flags, corpus_file=corpus_file, timeout=timeout)
    if code == 0 and ppl is not None:
        print(f"[DEBUG] Baseline perplexity parsed: PPL={ppl:.4f} (q8_0 fallback baseline).")
        return ppl, q8_flags, True
    preview = (stderr or "").strip().replace("\r", " ").replace("\n", " ")[:1200]
    print(f"[WARN] Baseline q8_0 perplexity failed (exit {code}); cache quantisation quality gate will be skipped. {preview}")
    return None, q8_flags, True


def passes_perplexity_gate(ppl, baseline_ppl, threshold=PPL_THRESHOLD):
    if baseline_ppl is None:
        return True
    return ppl is not None and ppl <= baseline_ppl * (1.0 + threshold)


def cache_differs_from_baseline(config, baseline_cache):
    cache_k = config.get("cache_k")
    cache_v = config.get("cache_v")
    cache_kd = config.get("cache_kd")
    cache_vd = config.get("cache_vd")
    return (
        cache_k != baseline_cache.get("cache_k") or
        cache_v != baseline_cache.get("cache_v") or
        (cache_kd is not None and cache_kd != baseline_cache.get("cache_kd", "f16")) or
        (cache_vd is not None and cache_vd != baseline_cache.get("cache_vd", "f16"))
    )


def build_perplexity_cache_flags(cache_k, cache_v, cache_kd=None, cache_vd=None, spec_active=False):
    flags = ["-ctk", str(cache_k), "-ctv", str(cache_v)]
    if spec_active and cache_kd is not None and cache_vd is not None:
        flags += ["-ctkd", str(cache_kd), "-ctvd", str(cache_vd)]
    return flags


def get_neighbors(current, sweep_list):
    idx = sweep_list.index(current)
    neighbors = [current]
    if idx > 0:
        neighbors.append(sweep_list[idx - 1])
    if idx < len(sweep_list) - 1:
        neighbors.append(sweep_list[idx + 1])
    return list(set(neighbors))


def port_is_free(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def build_server_flags(context_size, t=None, tb=None, b=None, ub=None, fitt=None,
                       cache_k="f16", cache_v="f16", cache_kd="f16", cache_vd="f16",
                       no_mmap=False, is_base=False,
                       mtp=False, spec_draft_n=None, draft_model_path=None, spec_draft_p_min=None):
    flags = [
        "-c", str(context_size),
        "-fitc", str(context_size),
        "--port", str(BENCH_PORT),
        "-lv", "4",
        "--host", "127.0.0.1",
        "-np", "1",
        "--no-warmup",
        "--no-mmap"
    ]
    if draft_model_path:
        flags += ["--model-draft", draft_model_path]

    # If speculative decoding is active (either MTP or draft model), add spec-type for both baseline and tuned runs
    if mtp or draft_model_path:
        flags += ["--spec-type", "draft-mtp"]

    if not is_base:
        flags += [
            "-t", str(t),
            "-b", str(b),
            "-ub", str(ub if ub is not None else b),
            "-fa", "on",
            "--fit", "on",
            "-fitt", str(fitt),
            "-ctk", str(cache_k),
            "-ctv", str(cache_v),
        ]
        if mtp or draft_model_path:
            flags += ["-ctkd", str(cache_kd), "-ctvd", str(cache_vd)]
        if tb is not None:
            flags += ["-tb", str(tb)]
        if spec_draft_n is not None:
            flags += ["--spec-draft-n-max", str(spec_draft_n)]
            if spec_draft_p_min is not None:
                flags += ["--spec-draft-p-min", f"{spec_draft_p_min:.1f}"]
            else:
                flags += ["--spec-draft-p-min", "0.4"]
        if no_mmap:
            flags.append("--no-mmap")
    return flags


def start_server(model_path, server_exe, context_size, proc_holder=None, **config):
    flags = build_server_flags(context_size, **config)
    cmd = [server_exe, "-m", model_path] + flags
    print(f"[DEBUG] Starting: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        # creationflags=subprocess.CREATE_NEW_CONSOLE,
        stderr=subprocess.PIPE, # Pipe stderr so we can inspect errors when server fails to start
        creationflags=_NO_WINDOW,
    )
    if proc_holder is not None:
        proc_holder[0] = proc
    return proc


def wait_for_server(port=BENCH_PORT, timeout=120, proc=None):
    url = f"http://127.0.0.1:{port}/health"
    print(f"[DEBUG] Waiting for server on port {port}...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        # Check if the process has terminated prematurely
        if proc is not None and proc.poll() is not None:
            err = read_process_stderr(proc)
            print(f"[DEBUG] Server process terminated prematurely with exit code {proc.returncode}.")
            if err:
                print("[SERVER STDERR]\n" + err)
            return False
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                print(f"[DEBUG] Server ready.")
                return True
        except requests.RequestException:
            pass
        time.sleep(0.5)
    print(f"[DEBUG] Server timed out after {timeout}s.")
    if proc is not None:
        # If still running, terminate first so stderr read doesn't block
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except Exception:
                proc.kill()
        if proc.stderr:
            err = proc.stderr.read().decode(errors="ignore")
            if err:
                print("[SERVER STDERR]\n" + err)
    return False


def stop_server(proc):
    if proc is None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def read_process_stderr(proc, max_chars=12000):
    if proc is None or proc.stderr is None:
        return ""
    try:
        err = proc.stderr.read().decode(errors="ignore")
    except Exception as e:
        return f"[stderr read failed: {e}]"
    if not err:
        return ""
    if len(err) <= max_chars:
        return err
    return err[-max_chars:]


def run_completion(port=BENCH_PORT):
    url = f"http://127.0.0.1:{port}/completion"
    payload = {
        "prompt": PP_PROMPT,
        "n_predict": TG_N_PREDICT,
        "temperature": 0.0,
        "cache_prompt": False,
    }
    try:
        r = requests.post(url, json=payload, timeout=120)
        if r.status_code != 200:
            print(f"[DEBUG] Completion request failed with status code {r.status_code}: {r.text}")
        return r.json()
    except Exception as e:
        print(f"[DEBUG] Completion request failed: {e}")
        return None


def parse_completion_results(response):
    """Parse pp and tg tokens/sec from /completion response timings.
    TODO: swap in your own parsing logic here."""
    if response is None:
        return 0.0, 0.0
    timings = response.get("timings", {})
    pp = timings.get("prompt_per_second", 0.0)
    tg = timings.get("predicted_per_second", 0.0)
    return float(pp), float(tg)


def run_benchmark(model_path, server_exe, context_size, proc_holder=None,
                  t=None, tb=None, b=None, ub=None, fitt=None, cache_k="f16", cache_v="f16",
                  cache_kd="f16", cache_vd="f16",
                  no_mmap=False, is_base=False, avg_runs=1, mtp=False, spec_draft_n=None,
                  draft_model_path=None, spec_draft_p_min=None):
    """Start llama-server, run benchmark, stop server. Returns (pp_tps, tg_tps)."""
    pp_total, tg_total, valid = 0.0, 0.0, 0

    for run_i in range(avg_runs):
        # Wait until port is fully released before starting
        deadline = time.time() + 15
        while not port_is_free(BENCH_PORT) and time.time() < deadline:
            print(f"[DEBUG] Port {BENCH_PORT} still occupied, waiting...")
            time.sleep(1)
        if not port_is_free(BENCH_PORT):
            print(f"[DEBUG] Port {BENCH_PORT} still occupied after 15s, skipping run {run_i+1}.")
            continue

        proc = start_server(
            model_path, server_exe, context_size,
            proc_holder=proc_holder,
            t=t, tb=tb, b=b, ub=ub, fitt=fitt,
            cache_k=cache_k, cache_v=cache_v,
            cache_kd=cache_kd, cache_vd=cache_vd,
            no_mmap=no_mmap, is_base=is_base,
            mtp=mtp, spec_draft_n=spec_draft_n,
            draft_model_path=draft_model_path,
            spec_draft_p_min=spec_draft_p_min
        )

        try:
            if not wait_for_server(proc=proc):
                stop_server(proc)
                continue

            response = run_completion()
            pp, tg = parse_completion_results(response)
            print(f"[DEBUG] Result: pp={pp:.2f} tg={tg:.2f}")

            if pp == 0 and tg == 0:
                print(f"[DEBUG] Benchmark returned 0 speed. Inspecting server logs...")
                stderr_printed = False
                if proc.poll() is not None:
                    err = read_process_stderr(proc)
                    if err:
                        print("[SERVER STDERR AFTER COMPLETION FAILURE]\n" + err)
                        stderr_printed = True
                stop_server(proc)
                if not stderr_printed:
                    err = read_process_stderr(proc)
                    if err:
                        print("[SERVER STDERR AFTER TERMINATION]\n" + err)
                continue
            if pp > 0 and tg > 0:
                pp_total += pp
                tg_total += tg
                valid += 1
        finally:
            stop_server(proc)
            if proc_holder is not None:
                proc_holder[0] = None
            # Give OS time to fully release the port
            time.sleep(2)

    if valid == 0:
        return 0.0, 0.0
    return pp_total / valid, tg_total / valid
