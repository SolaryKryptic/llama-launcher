import subprocess
import itertools
import re
import os

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_PATH = r"H:\AI\HauhauCS\Gemma-4-E2B-Uncensored-HauhauCS-Aggressive\Gemma-4-E2B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf"
LLAMA_BENCH_PATH = "llama-bench.exe" 

# Automatically detect system logical threads (Hyperthreads/SMT included)
MAX_SYSTEM_THREADS = os.cpu_count() or 4

# Calculate step size as 25% of total threads (minimum step size of 1)
STEP_SIZE = max(1, int(MAX_SYSTEM_THREADS * 0.25))

# Generate thread counts at 25% intervals (e.g., for 16 threads: 4, 8, 12, 16)
THREADS_LIST = [t for t in range(STEP_SIZE, MAX_SYSTEM_THREADS + 1, STEP_SIZE)]

# Ensure the exact maximum is included if rounding leaves it out
if MAX_SYSTEM_THREADS not in THREADS_LIST:
    THREADS_LIST.append(MAX_SYSTEM_THREADS)

# other test parameters
BATCH_LIST = [128, 256, 512, 1024, 2048, 4096]
NGL_LIST = [10, 20, 32, 99]      

def parse_bench_results(output):
    results = {}
    for line in output.splitlines():
        if "pp512" in line:
            parts = [p.strip() for p in line.split("|")]
            tps = parts[-2].split("±")[0].strip()
            # Clean out any broken encoding characters like  leaving only numbers
            tps = re.sub(r"[^\d.]", "", tps)
            results["pp512"] = float(tps)
        elif "tg128" in line:
            parts = [p.strip() for p in line.split("|")]
            tps = parts[-2].split("±")[0].strip()
            tps = re.sub(r"[^\d.]", "", tps)
            results["tg128"] = float(tps)
    return results

print("============================================================")
print(" Starting llama-bench Parameter Optimization Sweep")
print("============================================================")

best_tg_tps = 0.0
best_pp_tps = 0.0
best_config = {}

# Generate all unique combinations
parameter_combinations = list(itertools.product(THREADS_LIST, BATCH_LIST, NGL_LIST))

for threads, batch, ngl in parameter_combinations:
    print(f"Testing: Threads={threads}, Batch={batch}, NGL={ngl}")
    
    cmd = [
        LLAMA_BENCH_PATH,
        "-m", MODEL_PATH,
        "-t", str(threads),
        "-b", str(batch),
        "-ub", str(batch),  # Using batch size for unbatching as well
        "-ngl", str(ngl),
        "-o", "md"
    ]
    
    try:
        # Capture raw output and ignore bad byte encodings safely
        result = subprocess.run(cmd, capture_output=True, text=True, errors="ignore")
        
        if result.returncode != 0:
            print(f"  --> [FAIL] llama-bench exited with code {result.returncode}.")
            continue

        # Use your custom parser logic
        metrics = parse_bench_results(result.stdout)
        
        pp_tps = metrics.get("pp512", 0.0)
        tg_tps = metrics.get("tg128", 0.0)
        
        if pp_tps == 0.0 and tg_tps == 0.0:
            print("  --> [WARNING] No metrics extracted for this configuration.")
            continue
            
        print(f"  --> [SUCCESS] Prompt processing: {pp_tps:.2f} t/s | Text generation: {tg_tps:.2f} t/s")
        
        if tg_tps > best_tg_tps:
            best_tg_tps = tg_tps
            best_pp_tps = pp_tps
            best_config = {
                "threads": threads,
                "batch": batch,
                "ngl": ngl
            }
            
    except Exception as e:
        print(f"  --> [ERROR] Unexpected execution failure: {e}")

print("============================================================")
print(" Optimization Complete!")
print("============================================================")
if best_config:
    print("Best Configuration Found:")
    print(f"  Text Generation Speed: {best_tg_tps:.2f} tokens/sec")
    print(f"  Prompt Processing Speed: {best_pp_tps:.2f} tokens/sec")
    print("  Recommended Optimization Flags for llama-server:")
    print(f"    -t {best_config['threads']} -b {best_config['batch']} -ngl {best_config['ngl']}")
else:
    print("No valid configurations completed successfully.")
print("============================================================")
