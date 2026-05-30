import subprocess
import itertools
import re
import os

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_PATH = r"H:\AI\HauhauCS\Gemma-4-E2B-Uncensored-HauhauCS-Aggressive\Gemma-4-E2B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf"
LLAMA_BENCH_PATH = "llama-bench.exe" 

# Context Size
CONTEXT_SIZE = 16384
# Dynamic dual-metric focus score weight (0.0 to 1.0)
# 0.5 = Balanced equally | 0.2 = Focus on Chat Text Gen | 0.8 = Focus on Prompt Load
METRIC_WEIGHT = 0.5

# Automatically detect system logical threads (Hyperthreads/SMT included)
MAX_SYSTEM_THREADS = os.cpu_count() or 4

# Cap the maximum allowed threads at 75% of system capacity
THREAD_CAP_LIMIT = max(1, int(MAX_SYSTEM_THREADS * 0.75))

# Calculate step size as 25% of total system threads (minimum step size of 1)
STEP_SIZE = max(1, int(MAX_SYSTEM_THREADS * 0.25))

# Generate thread counts at 25% intervals up to the 75% cap limit
THREADS_LIST = [t for t in range(STEP_SIZE, THREAD_CAP_LIMIT + 1, STEP_SIZE)]

# Ensure the exact 75% cap limit boundary is explicitly included in the test pool
if THREAD_CAP_LIMIT not in THREADS_LIST:
    THREADS_LIST.append(THREAD_CAP_LIMIT)

# Other test parameters to sweep sequentially
BATCH_LIST = [512, 1024, 2048]
FITT_LIST = [1024, 512, 256]
CACHE_K_TYPES = ["f16", "q8_0", "q4_0"]  # K cache only — V cache locked to f16 (llama-bench limitation)
CACHE_V_TYPE = "f16"                       # V cache always f16

# Starting default placeholders for greedy pass sequential tracking
best_threads = THREADS_LIST[-1]
best_batch = 512
best_fitt = FITT_LIST[0]
best_cache_k = CACHE_K_TYPES[0]

def parse_bench_results(output):
    results = {}
    for line in output.splitlines():
        if "pp512" in line:
            parts = [p.strip() for p in line.split("|")]
            tps = parts[-2].split("±")[0].strip()
            tps = re.sub(r"[^\d.]", "", tps)
            results["pp512"] = float(tps)
        elif "tg128" in line:
            parts = [p.strip() for p in line.split("|")]
            tps = parts[-2].split("±")[0].strip()
            tps = re.sub(r"[^\d.]", "", tps)
            results["tg128"] = float(tps)
    return results

def run_benchmark(t=None, b=None, fitt=None, cache_k=None, is_base=False):
    # Base command structure
    cmd = [LLAMA_BENCH_PATH, "-m", MODEL_PATH, "-o", "md", "--no-warmup", "-fitc", str(CONTEXT_SIZE), "-r", "3"]
    
    # Only append custom overrides if it's not the naked baseline run
    if not is_base:
        cmd.extend([
            "-t", str(t),
            "-b", str(b),
            "-ub", str(b),
            "-fa", "on",
            "-fitt", str(fitt),
            "-fitc", str(CONTEXT_SIZE),
            "-ctk", str(cache_k),
            "-ctv", CACHE_V_TYPE,
        ])
        
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, errors="ignore")
        if result.returncode != 0:
            print(f"  --> [FAIL] llama-bench exited with code {result.returncode}.")
            return 0.0, 0.0
        
        metrics = parse_bench_results(result.stdout)
        pp_tps = metrics.get("pp512", 0.0)
        tg_tps = metrics.get("tg128", 0.0)
        
        if pp_tps == 0.0 and tg_tps == 0.0:
            print("  --> [WARNING] No metrics extracted for this configuration.")
            return 0.0, 0.0
            
        return pp_tps, tg_tps
    except Exception as e:
        print(f"  --> [ERROR] Unexpected execution failure: {e}")
        return 0.0, 0.0

def calculate_score(pp, tg):
    return (pp * METRIC_WEIGHT) + (tg * (1.0 - METRIC_WEIGHT))

print("============================================================")
print(f" Detected {MAX_SYSTEM_THREADS} Total Logical CPU Threads.")
print(f" Capped Sweep Boundary (75%): {THREAD_CAP_LIMIT} threads maximum.")
print(f" Thread configurations to sweep: {THREADS_LIST}")
print(f" V Cache locked to: {CACHE_V_TYPE} (llama-bench limitation)")
print("============================================================")
print(" INITIAL STAGE: Running Unoptimized System Baseline")
print("============================================================")

base_pp, base_tg = run_benchmark(is_base=True)
base_score = calculate_score(base_pp, base_tg)

if base_score > 0:
    print(f"[BASE RUN] Prompt: {base_pp:.2f} t/s | Text Gen: {base_tg:.2f} t/s | Score: {base_score:.2f}")
else:
    print("[BASE RUN FAIL] Could not establish baseline metrics. Exiting script.")
    exit(1)

print("\n============================================================")
print(f" STAGE 1: Fast Sequential Greedy Screening (Weight: {METRIC_WEIGHT})")
print("============================================================")

# Global best tracks the single highest score seen across all Stage 1 steps
global_best_score = -1.0
global_best_pp = 0.0
global_best_tg = 0.0

# 1. Optimize Threads
print(f"\n--- Step 1.1: Optimizing Threads (Testing: {THREADS_LIST}) ---")
stage1_score = -1.0
for t in THREADS_LIST:
    print(f"Testing: Threads={t} (Batch={best_batch}, FITT={best_fitt}, CacheK={best_cache_k})")
    pp, tg = run_benchmark(t, best_batch, best_fitt, best_cache_k)
    if pp == 0.0 and tg == 0.0: continue
    score = calculate_score(pp, tg)
    print(f"  --> [SUCCESS] Prompt: {pp:.2f} t/s | Text: {tg:.2f} t/s | Score: {score:.2f}")
    if score > stage1_score:
        stage1_score = score
        best_threads = t
    if score > global_best_score:
        global_best_score = score
        global_best_pp = pp
        global_best_tg = tg

# 2. Optimize Batch Size
print(f"\n--- Step 1.2: Optimizing Batch Size (Testing: {BATCH_LIST}) ---")
stage1_score = -1.0
for b in BATCH_LIST:
    print(f"Testing: Batch={b} (Threads={best_threads}, FITT={best_fitt}, CacheK={best_cache_k})")
    pp, tg = run_benchmark(best_threads, b, best_fitt, best_cache_k)
    if pp == 0.0 and tg == 0.0: continue
    score = calculate_score(pp, tg)
    print(f"  --> [SUCCESS] Prompt: {pp:.2f} t/s | Text: {tg:.2f} t/s | Score: {score:.2f}")
    if score > stage1_score:
        stage1_score = score
        best_batch = b
    if score > global_best_score:
        global_best_score = score
        global_best_pp = pp
        global_best_tg = tg

# 3. Optimize VRAM Fit Target (FITT)
print(f"\n--- Step 1.3: Optimizing VRAM Fit Target (Testing: {FITT_LIST}) ---")
stage1_score = -1.0
for fitt in FITT_LIST:
    print(f"Testing: FITT={fitt} (Threads={best_threads}, Batch={best_batch}, CacheK={best_cache_k})")
    pp, tg = run_benchmark(best_threads, best_batch, fitt, best_cache_k)
    if pp == 0.0 and tg == 0.0: continue
    score = calculate_score(pp, tg)
    print(f"  --> [SUCCESS] Prompt: {pp:.2f} t/s | Text: {tg:.2f} t/s | Score: {score:.2f}")
    if score > stage1_score:
        stage1_score = score
        best_fitt = fitt
    if score > global_best_score:
        global_best_score = score
        global_best_pp = pp
        global_best_tg = tg

# 4. Optimize K Cache Type only
print(f"\n--- Step 1.4: Optimizing K Cache / -ctk (Testing: {CACHE_K_TYPES}, -ctv locked to {CACHE_V_TYPE}) ---")
stage1_score = -1.0
for cache_k in CACHE_K_TYPES:
    print(f"Testing: CacheK={cache_k} (Threads={best_threads}, Batch={best_batch}, FITT={best_fitt})")
    pp, tg = run_benchmark(best_threads, best_batch, best_fitt, cache_k)
    if pp == 0.0 and tg == 0.0: continue
    score = calculate_score(pp, tg)
    print(f"  --> [SUCCESS] Prompt: {pp:.2f} t/s | Text: {tg:.2f} t/s | Score: {score:.2f}")
    if score > stage1_score:
        stage1_score = score
        best_cache_k = cache_k
    if score > global_best_score:
        global_best_score = score
        global_best_pp = pp
        global_best_tg = tg

print(f"\n>> Stage 1 Complete. Best Params: -t {best_threads} -b {best_batch} -fitt {best_fitt} -ctk {best_cache_k} | Global Best Score: {global_best_score:.2f}")

print("\n============================================================")
print(" STAGE 2: Targeted Neighborhood Verification")
print("============================================================")
print("Testing parameter grids adjacent to Stage 1 winner to bypass parameter interactions...")

def get_neighbors(current, sweep_list):
    idx = sweep_list.index(current)
    neighbors = [current]
    if idx > 0:
        neighbors.append(sweep_list[idx - 1])
    if idx < len(sweep_list) - 1:
        neighbors.append(sweep_list[idx + 1])
    return list(set(neighbors))

# Threads and cache locked to Stage 1 best — only batch and fitt are neighbour-verified
b_neighbors = get_neighbors(best_batch, BATCH_LIST)
fitt_neighbors = get_neighbors(best_fitt, FITT_LIST)

neighborhood_grid = list(itertools.product(b_neighbors, fitt_neighbors))

final_best_score = global_best_score
final_best_pp = global_best_pp
final_best_tg = global_best_tg
final_config = {"threads": best_threads, "batch": best_batch, "fitt": best_fitt, "cache_k": best_cache_k}

for b, fitt in neighborhood_grid:
    if b == best_batch and fitt == best_fitt:
        continue
        
    print(f"Verifying Interaction: Batch={b}, FITT={fitt} (Threads={best_threads}, CacheK={best_cache_k})")
    pp, tg = run_benchmark(best_threads, b, fitt, best_cache_k)
    if pp == 0.0 and tg == 0.0: continue
    score = calculate_score(pp, tg)
    print(f"  --> [SUCCESS] Prompt: {pp:.2f} t/s | Text: {tg:.2f} t/s | Score: {score:.2f}")
    
    if score > final_best_score:
        final_best_score = score
        final_best_pp = pp
        final_best_tg = tg
        final_config = {"threads": best_threads, "batch": b, "fitt": fitt, "cache_k": best_cache_k}

print("\n============================================================")
print(" Optimization Complete!")
print("============================================================")
if final_best_score > 0.0:
    print("Absolute Best Balanced Configuration Found:")
    print(f"  Text Generation Speed: {final_best_tg:.2f} tokens/sec")
    print(f"  Prompt Processing Speed: {final_best_pp:.2f} tokens/sec")
    print("  Recommended Optimization Flags for llama-server:")
    print(f"    -t {final_config['threads']} -b {final_config['batch']} -ub {final_config['batch']} -ctx {CONTEXT_SIZE} -fitt {final_config['fitt']} -ctk {final_config['cache_k']} -ctv {CACHE_V_TYPE}")
    
    percentage_gain = ((final_best_score - base_score) / base_score) * 100.0
    print(f"\nPerformance Improvement Over Baseline: {percentage_gain:.2f}% (Results may vary by nature of variance)")