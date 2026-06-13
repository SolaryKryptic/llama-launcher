import subprocess
import itertools
import re
import os
import socket
import time
import bayesian as bayes
import requests

_NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0

BENCH_PORT = 8080  # Fixed port for optimiser server instances

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


# ==========================================
# PURE UTILITY FUNCTIONS (importable)
# ==========================================

def build_thread_list():
    """Build thread sweep list based on detected CPU count.
    If CPU count cannot be detected, threads list is empty and thread sweep is skipped.
    
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
                       cache_k="f16", cache_v="f16", no_mmap=False, is_base=False,
                       mtp=False, spec_draft_n=None, draft_model_path=None, spec_draft_p_min=None):
    flags = [
        "-c", str(context_size),
        "--port", str(BENCH_PORT),
        "-lv", "3",
        "--host", "127.0.0.1",
        "-np", "1",
        "--no-warmup",
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
        stdout=subprocess.PIPE,
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


def run_full_optimisation(model_path, server_exe, context_size=16384, metric_weight=0.5,
                          progress_callback=None, cancel_flag=None, proc_holder=None,
                          draft_model_path=None):
    """Run full two-stage sequential greedy + neighbourhood optimisation using llama-server."""
    if cancel_flag is None:
        cancel_flag = [False]
    if proc_holder is None:
        proc_holder = [None]

    try:
        params = build_thread_list()
        threads_list  = params["threads"]
        batch_list    = params["batch_sizes"]
        fitt_list     = params["fitt_targets"]
        cache_k_types = params["cache_k_types"]
        cache_v_types = params["cache_v_types"]

        stage1_total = len(threads_list) + len(batch_list) + len(fitt_list) + len(cache_k_types) + len(cache_v_types)
        neighbourhood_estimate = 6
        total_runs = stage1_total + neighbourhood_estimate
        run_idx = [0]

        def _cb(step_name, last_score, best_score, baseline_score):
            run_idx[0] += 1
            if progress_callback:
                progress_callback(run_idx[0], total_runs, step_name, last_score, best_score, baseline_score)

        def bench(step_name, best_score, baseline_score, **config):
            pp, tg = run_benchmark(model_path, server_exe, context_size,
                                   proc_holder=proc_holder, draft_model_path=draft_model_path, **config)
            score = calculate_score(pp, tg, metric_weight)
            _cb(step_name, score, max(score, best_score), baseline_score)
            return pp, tg, score

        # ---- Baseline ----
        base_pp, base_tg = run_benchmark(model_path, server_exe, context_size,
                                         proc_holder=proc_holder, is_base=True,
                                         draft_model_path=draft_model_path)
        baseline_score = calculate_score(base_pp, base_tg, metric_weight)
        if baseline_score <= 0:
            return None

        # ---- Stage 1 init ----
        best_threads = threads_list[-1] if threads_list else None
        best_batch   = batch_list[0]
        best_fitt    = fitt_list[0]
        best_cache_k = cache_k_types[0]
        best_cache_v = cache_v_types[0]
        global_best  = baseline_score
        global_best_pp = base_pp
        global_best_tg = base_tg

        # Step 1.1 — Threads
        stage1_step_best = -1.0
        for t in threads_list:
            if cancel_flag[0]: return None
            pp, tg, score = bench(f"Threads={t}", global_best, baseline_score,
                                  t=t, b=best_batch, fitt=best_fitt,
                                  cache_k=best_cache_k, cache_v=best_cache_v)
            if score > stage1_step_best:
                stage1_step_best = score
                best_threads = t
            if score > global_best:
                global_best = score
                global_best_pp, global_best_tg = pp, tg

        # Step 1.2 — Batch size (with early stopping)
        stage1_step_best = -1.0
        drops = 0
        for b in batch_list:
            if cancel_flag[0]: return None
            pp, tg, score = bench(f"Batch={b}", global_best, baseline_score,
                                  t=best_threads, b=b, fitt=best_fitt,
                                  cache_k=best_cache_k, cache_v=best_cache_v)
            if score > stage1_step_best:
                stage1_step_best = score
                best_batch = b
                drops = 0
            else:
                drops += 1
                if drops >= 2:
                    break
            if score > global_best:
                global_best = score
                global_best_pp, global_best_tg = pp, tg

        # Step 1.3 — FITT target
        stage1_step_best = -1.0
        for fitt in fitt_list:
            if cancel_flag[0]: return None
            pp, tg, score = bench(f"FITT={fitt}", global_best, baseline_score,
                                  t=best_threads, b=best_batch, fitt=fitt,
                                  cache_k=best_cache_k, cache_v=best_cache_v)
            if score > stage1_step_best:
                stage1_step_best = score
                best_fitt = fitt
            if score > global_best:
                global_best = score
                global_best_pp, global_best_tg = pp, tg

        # Step 1.4 — K Cache type
        stage1_step_best = -1.0
        for ck in cache_k_types:
            if cancel_flag[0]: return None
            pp, tg, score = bench(f"CacheK={ck}", global_best, baseline_score,
                                  t=best_threads, b=best_batch, fitt=best_fitt,
                                  cache_k=ck, cache_v=best_cache_v)
            if score > stage1_step_best:
                stage1_step_best = score
                best_cache_k = ck
            if score > global_best:
                global_best = score
                global_best_pp, global_best_tg = pp, tg

        # Step 1.5 — V Cache type
        stage1_step_best = -1.0
        for cv in cache_v_types:
            if cancel_flag[0]: return None
            pp, tg, score = bench(f"CacheV={cv}", global_best, baseline_score,
                                  t=best_threads, b=best_batch, fitt=best_fitt,
                                  cache_k=best_cache_k, cache_v=cv)
            if score > stage1_step_best:
                stage1_step_best = score
                best_cache_v = cv
            if score > global_best:
                global_best = score
                global_best_pp, global_best_tg = pp, tg

        # ---- Stage 2 — Neighbourhood verification ----
        b_neighbors    = get_neighbors(best_batch, batch_list)
        fitt_neighbors = get_neighbors(best_fitt, fitt_list)
        grid = list(itertools.product(b_neighbors, fitt_neighbors))

        final_best_score = global_best
        final_config = {
            "threads": best_threads,
            "batch": best_batch,
            "fitt": best_fitt,
            "cache_k": best_cache_k,
            "cache_v": best_cache_v,
            "baseline_score": baseline_score,
            "best_score": global_best,
            "best_pp": global_best_pp,
            "best_tg": global_best_tg,
        }

        for b, fitt in grid:
            if b == best_batch and fitt == best_fitt:
                continue
            if cancel_flag[0]: return None
            pp, tg, score = bench(f"Verify B={b} FITT={fitt}", final_best_score, baseline_score,
                                  t=best_threads, b=b, fitt=fitt,
                                  cache_k=best_cache_k, cache_v=best_cache_v)
            if score > final_best_score:
                final_best_score = score
                final_config = {
                    "threads": best_threads,
                    "batch": b,
                    "fitt": fitt,
                    "cache_k": best_cache_k,
                    "cache_v": best_cache_v,
                    "baseline_score": baseline_score,
                    "best_score": score,
                    "best_pp": pp,
                    "best_tg": tg,
                }

        return final_config
    finally:
        # Ensure server process is stopped when function exits (success, error, or cancellation)
        if proc_holder is not None:
            try:
                proc = proc_holder[0]
                if proc is not None and proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except Exception:
                        proc.kill()
            except Exception:
                pass
            finally:
                proc_holder[0] = None


# ==========================================
# STANDALONE ENTRY POINT
# ==========================================
if __name__ == "__main__":
    MODEL_PATH   = r"H:\AI\unsloth\gemma-4-E4B-it-qat-GGUF\gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf"
    LLAMA_SERVER = "llama-server.exe"
    CONTEXT_SIZE = 16384
    METRIC_WEIGHT = 0.5

    params = build_thread_list()
    print("============================================================")
    print(f" Detected {params['max_threads']} Total Logical CPU Threads.")
    print(f" Capped Sweep Boundary (75%): {params['cap_limit']} threads maximum.")
    print(f" Thread configurations to sweep: {params['threads']}")
    print(f" Benchmark port: {BENCH_PORT}")
    print("============================================================")

    def _print_progress(run_idx, total, step_name, last_score, best_score, baseline_score):
        print(f"  [{run_idx}/{total}] {step_name} | Last: {last_score:.2f} | Best: {best_score:.2f} | Baseline: {baseline_score:.2f}")

    result = run_full_optimisation(
        model_path=MODEL_PATH,
        server_exe=LLAMA_SERVER,
        context_size=CONTEXT_SIZE,
        metric_weight=METRIC_WEIGHT,
        progress_callback=_print_progress,
    )

    if result:
        print("\n============================================================")
        print(" Optimization Complete!")
        print("============================================================")
        print(f"  Text Gen Speed:    {result['best_tg']:.2f} t/s")
        print(f"  Prompt Speed:      {result['best_pp']:.2f} t/s")
        print(f"  Best Score:        {result['best_score']:.2f}")
        print(f"  Baseline Score:    {result['baseline_score']:.2f}")
        pct = ((result['best_score'] - result['baseline_score']) / result['baseline_score']) * 100
        print(f"  Improvement:       {pct:.2f}%")
        print(f"\n  Flags: -t {result['threads']} -b {result['batch']} -ub {result['batch']} "
              f"-c {CONTEXT_SIZE} -fitt {result['fitt']} "
              f"-ctk {result['cache_k']} -ctv {result['cache_v']}")
    else:
        print("[FAIL] Optimisation did not complete.")
