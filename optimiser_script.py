import subprocess
import itertools
import re
import os

CACHE_V_TYPE = "f16"  # V cache always f16 (llama-bench limitation on Vulkan)
_NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0


# ==========================================
# PURE UTILITY FUNCTIONS (importable)
# ==========================================

def build_thread_list():
    """Build thread sweep list based on detected CPU count. Returns dict of sweep params.
    If CPU count cannot be detected, threads list is empty and thread sweep is skipped."""
    max_threads = os.cpu_count()
    if max_threads is not None:
        cap_limit = max(1, int(max_threads * 0.75))
        step_size = max(1, int(max_threads * 0.25))
        threads_list = [t for t in range(step_size, cap_limit + 1, step_size)]
        if cap_limit not in threads_list:
            threads_list.append(cap_limit)
    else:
        # Could not detect CPU count — skip thread sweep entirely
        threads_list = []
        cap_limit = 0
    return {
        "threads": threads_list,
        "batch_sizes": [512, 1024, 2048],
        "fitt_targets": [1024, 512, 256],
        "cache_k_types": ["f16", "q8_0", "q4_0"],
        "max_threads": max_threads,
        "cap_limit": cap_limit,
    }


def parse_bench_results(output):
    """Parse pp512 and tg128 tokens/sec from llama-bench markdown output."""
    results = {}
    for line in output.splitlines():
        if "pp512" in line:
            parts = [p.strip() for p in line.split("|")]
            tps = re.sub(r"[^\d.]", "", parts[-2].split("±")[0].strip())
            if tps:
                results["pp512"] = float(tps)
        elif "tg128" in line:
            parts = [p.strip() for p in line.split("|")]
            tps = re.sub(r"[^\d.]", "", parts[-2].split("±")[0].strip())
            if tps:
                results["tg128"] = float(tps)
    return results


def calculate_score(pp, tg, metric_weight=0.5):
    """Dual metric weighted score. metric_weight 0=TG only, 1=PP only, 0.5=balanced."""
    if pp is None or tg is None:
        return -1.0
    return (pp * metric_weight) + (tg * (1.0 - metric_weight))


def get_neighbors(current, sweep_list):
    """Return current value plus adjacent values in the sweep list."""
    idx = sweep_list.index(current)
    neighbors = [current]
    if idx > 0:
        neighbors.append(sweep_list[idx - 1])
    if idx < len(sweep_list) - 1:
        neighbors.append(sweep_list[idx + 1])
    return list(set(neighbors))


def run_benchmark(model_path, bench_exe, context_size, t=None, b=None,
                  fitt=None, cache_k=None, is_base=False, proc_holder=None):
    cmd = [bench_exe, "-m", model_path, "-o", "md", "--no-warmup",
           "-fitc", str(context_size), "-r", "1"]
    if not is_base:
        cmd.extend([
            "-t", str(t), "-b", str(b), "-ub", str(b),
            "-fa", "on", "-fitt", str(fitt),
            "-ctk", str(cache_k), "-ctv", CACHE_V_TYPE,
        ])
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, errors="ignore", creationflags=_NO_WINDOW
        )
        if proc_holder is not None:
            proc_holder[0] = proc
        stdout, _ = proc.communicate()
        if proc.returncode != 0:
            return 0.0, 0.0
        metrics = parse_bench_results(stdout)
        pp = metrics.get("pp512", 0.0)
        tg = metrics.get("tg128", 0.0)
        return (pp, tg) if pp or tg else (0.0, 0.0)
    except Exception:
        return 0.0, 0.0


def run_full_optimisation(model_path, bench_exe, context_size=16384, metric_weight=0.5,
                          progress_callback=None, cancel_flag=None):
    """
    Run full two-stage sequential greedy + neighbourhood optimisation.

    progress_callback(run_idx, total_runs, step_name, last_score, best_score, baseline_score)
    cancel_flag: mutable list [False] — set to [True] to abort mid-sweep.

    Returns final_config dict or None if cancelled/failed.
    """
    if cancel_flag is None:
        cancel_flag = [False]

    params = build_thread_list()
    threads_list  = params["threads"]
    batch_list    = params["batch_sizes"]
    fitt_list     = params["fitt_targets"]
    cache_k_types = params["cache_k_types"]

    # Estimate total runs: stage1 steps + ~neighbourhood grid
    stage1_total = len(threads_list) + len(batch_list) + len(fitt_list) + len(cache_k_types)
    neighbourhood_estimate = 6
    total_runs = stage1_total + neighbourhood_estimate
    run_idx = [0]

    def _cb(step_name, last_score, best_score, baseline_score):
        run_idx[0] += 1
        if progress_callback:
            progress_callback(run_idx[0], total_runs, step_name, last_score, best_score, baseline_score)

    proc_holder = [None]  # Holder for the current benchmark process, to allow cancellation

    def bench(t, b, fitt, ck, step_name, best_score, baseline_score):
        pp, tg = run_benchmark(model_path, bench_exe, context_size, t=t, b=b, fitt=fitt, cache_k=ck, proc_holder=proc_holder)
        score = calculate_score(pp, tg, metric_weight)
        _cb(step_name, score, max(score, best_score), baseline_score)
        return pp, tg, score

    # ---- Baseline ----
    base_pp, base_tg = run_benchmark(model_path, bench_exe, context_size, is_base=True)
    baseline_score = calculate_score(base_pp, base_tg, metric_weight)
    if baseline_score <= 0:
        return None

    # ---- Stage 1 init ----
    best_threads = threads_list[-1]
    best_batch   = batch_list[0]
    best_fitt    = fitt_list[0]
    best_cache_k = cache_k_types[0]
    global_best  = baseline_score
    global_best_pp = base_pp
    global_best_tg = base_tg

    # Step 1.1 — Threads
    stage1_step_best = -1.0
    for t in threads_list:
        if cancel_flag[0]: return None
        pp, tg, score = bench(t, best_batch, best_fitt, best_cache_k,
                              f"Threads={t}", global_best, baseline_score)
        if score > stage1_step_best:
            stage1_step_best = score
            best_threads = t
        if score > global_best:
            global_best = score
            global_best_pp, global_best_tg = pp, tg

    # Step 1.2 — Batch size
    stage1_step_best = -1.0
    prev_score = -1.0
    drops = 0
    for b in batch_list:
        if cancel_flag[0]: return None
        pp, tg, score = bench(best_threads, b, best_fitt, best_cache_k,
                              f"Batch={b}", global_best, baseline_score)
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
        prev_score = score

    # Step 1.3 — FITT target
    stage1_step_best = -1.0
    for fitt in fitt_list:
        if cancel_flag[0]: return None
        pp, tg, score = bench(best_threads, best_batch, fitt, best_cache_k,
                              f"FITT={fitt}", global_best, baseline_score)
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
        pp, tg, score = bench(best_threads, best_batch, best_fitt, ck,
                              f"CacheK={ck}", global_best, baseline_score)
        if score > stage1_step_best:
            stage1_step_best = score
            best_cache_k = ck
        if score > global_best:
            global_best = score
            global_best_pp, global_best_tg = pp, tg

    # ---- Stage 2 — Neighbourhood verification ----
    b_neighbors    = get_neighbors(best_batch, batch_list)
    fitt_neighbors = get_neighbors(best_fitt, fitt_list)
    grid = list(itertools.product(b_neighbors, fitt_neighbors))

    final_best_score = global_best
    final_config = {"threads": best_threads, "batch": best_batch,
                    "fitt": best_fitt, "cache_k": best_cache_k,
                    "baseline_score": baseline_score,
                    "best_score": global_best,
                    "best_pp": global_best_pp,
                    "best_tg": global_best_tg}

    for b, fitt in grid:
        if b == best_batch and fitt == best_fitt:
            continue
        if cancel_flag[0]: return None
        pp, tg, score = bench(best_threads, b, fitt, best_cache_k,
                              f"Verify B={b} FITT={fitt}", final_best_score, baseline_score)
        if score > final_best_score:
            final_best_score = score
            final_config = {"threads": best_threads, "batch": b,
                            "fitt": fitt, "cache_k": best_cache_k,
                            "baseline_score": baseline_score,
                            "best_score": score,
                            "best_pp": pp,
                            "best_tg": tg}

    return final_config


# ==========================================
# STANDALONE ENTRY POINT
# ==========================================
if __name__ == "__main__":
    MODEL_PATH     = r"H:\AI\unsloth\gemma-4-E4B-it-qat-GGUF\gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf"
    LLAMA_BENCH_PATH = "llama-bench.exe"
    CONTEXT_SIZE   = 16384
    METRIC_WEIGHT  = 0.5

    params = build_thread_list()
    print("============================================================")
    print(f" Detected {params['max_threads']} Total Logical CPU Threads.")
    print(f" Capped Sweep Boundary (75%): {params['cap_limit']} threads maximum.")
    print(f" Thread configurations to sweep: {params['threads']}")
    print(f" V Cache locked to: {CACHE_V_TYPE} (llama-bench limitation)")
    print("============================================================")

    def _print_progress(run_idx, total, step_name, last_score, best_score, baseline_score):
        print(f"  [{run_idx}/{total}] {step_name} | Last: {last_score:.2f} | Best: {best_score:.2f} | Baseline: {baseline_score:.2f}")

    result = run_full_optimisation(
        model_path=MODEL_PATH,
        bench_exe=LLAMA_BENCH_PATH,
        context_size=CONTEXT_SIZE,
        metric_weight=METRIC_WEIGHT,
        progress_callback=_print_progress
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
              f"-ctx {CONTEXT_SIZE} -fitt {result['fitt']} -ctk {result['cache_k']} -ctv {CACHE_V_TYPE}")
    else:
        print("[FAIL] Optimisation did not complete.")