"""
Bayesian optimisation prototype harness for llama-launcher.
Uses Optuna (TPE) to search mixed parameter space and wraps
`optimiser_script.run_benchmark` as the objective.

Run as a standalone script for quick tests. Example:

    python bayesian.py --model "path/to/model.gguf" --server llama-server.exe --trials 6

If `optuna` is not installed, the script prints instructions to install it.
"""

import argparse
import time

try:
    import optuna
except Exception:
    optuna = None

import optimiser_script as opt


def run_bayesian_optimisation(model_path, server_exe, context_size=16384,
                              metric_weight=0.1, n_trials=30, avg_runs=1,
                              progress_callback=None, cancel_flag=None, proc_holder=None,
                              mtp=False, draft_model_path=None):
    """Run an Optuna (TPE) search over the same parameter families used by
    `optimiser_script.run_full_optimisation`. Returns a final_config dict
    matching the existing optimiser's returned structure, or None on failure.
    
    If mtp=True or draft_model_path is set, includes spec_draft_n parameter for MTP/Speculative decoding.
    """
    if optuna is None:
        raise RuntimeError("optuna is not available — install with `pip install optuna`")

    if cancel_flag is None:
        cancel_flag = [False]

    is_speculative = mtp or bool(draft_model_path)

    # Baseline
    base_pp, base_tg = opt.run_benchmark(
        model_path, server_exe, context_size,
        proc_holder=proc_holder, is_base=True, avg_runs=avg_runs,
        draft_model_path=draft_model_path, mtp=is_speculative
    )
    baseline_score = opt.calculate_score(base_pp, base_tg, metric_weight)
    if baseline_score <= 0:
        print("[ERROR] Baseline measurement failed or produced non-positive score.")
        return None

    params = opt.build_thread_list()
    threads_choices = params.get("threads") or [max(1, params.get("cap_limit", 1))]
    thread_batch_choices = params.get("thread_batch") or threads_choices  # -tb, can use 100% threads
    batch_choices = params.get("batch_sizes", [128, 256, 512, 1024, 2048])
    micro_batch_choices = params.get("micro_batch_sizes", [128, 256, 512, 1024, 2048])  # -ub, now independent
    fitt_choices = params.get("fitt_targets", [50])
    cache_k_choices = params.get("cache_k_types", ["f16", "q8_0", "q5_0", "q4_0"])
    cache_v_choices = params.get("cache_v_types", ["f16", "q8_0", "q5_0", "q4_0"])
    spec_draft_n_choices = params.get("spec_draft_n", list(range(1, 8))) if is_speculative else None
    
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(n_startup_trials=9)
        )
    
    # Early stopping: track consecutive trials without improvement
    early_stop_state = {"best_value": None, "no_improve_count": 0}

    def callback(study, trial):
        """Called after each trial completes. Stop if 10 consecutive trials show no improvement."""
        try:
            current_best = study.best_value
        except ValueError:
            return

        if early_stop_state["best_value"] is None:
            early_stop_state["best_value"] = current_best
            early_stop_state["no_improve_count"] = 0
        elif current_best > early_stop_state["best_value"]:
            # Improvement found
            early_stop_state["best_value"] = current_best
            early_stop_state["no_improve_count"] = 0
        else:
            # No improvement
            early_stop_state["no_improve_count"] += 1
            if early_stop_state["no_improve_count"] >= 15:
                print(f"[INFO] Early stopping: 15 consecutive trials without improvement.")
                study.stop()

    def objective(trial):
        if cancel_flag and cancel_flag[0]:
            raise optuna.TrialPruned()

        t = trial.suggest_categorical("threads", threads_choices)
        tb = trial.suggest_categorical("thread_batch", thread_batch_choices)
        b = trial.suggest_categorical("batch", batch_choices)
        ub = trial.suggest_categorical("micro_batch", micro_batch_choices)
        fitt = trial.suggest_categorical("fitt", fitt_choices)
        ck = trial.suggest_categorical("cache_k", cache_k_choices)
        cv = trial.suggest_categorical("cache_v", cache_v_choices)
        sdn = trial.suggest_int("spec_draft_n", min(spec_draft_n_choices), max(spec_draft_n_choices)) if is_speculative else None
        sdp = trial.suggest_float("spec_draft_p_min", 0.1, 0.9, step=0.1) if is_speculative else None

        pp, tg = opt.run_benchmark(
            model_path, server_exe, context_size,
            proc_holder=proc_holder,
            t=t, tb=tb, b=b, ub=ub, fitt=fitt,
            cache_k=ck, cache_v=cv,
            mtp=is_speculative, spec_draft_n=sdn,
            avg_runs=avg_runs, draft_model_path=draft_model_path,
            spec_draft_p_min=sdp
        )
        score = opt.calculate_score(pp, tg, metric_weight)

        # Save prompt processing and text generation speeds to avoid a final verification run
        trial.set_user_attr("pp", pp)
        trial.set_user_attr("tg", tg)

        # Progress callback with best-so-far
        try:
            best = study.best_value if study.best_value is not None else baseline_score
        except Exception:
            best = baseline_score
        if progress_callback:
            progress_callback(trial.number + 1, n_trials, f"Trial-{trial.number+1}", score, best, baseline_score)

        # If measurement failed, return a low score
        if pp == 0 and tg == 0:
            return -1.0
        return score

    # Warm-start with a sensible baseline configuration
    default_threads = threads_choices[-1] if threads_choices else max(1, params.get("cap_limit", 1))
    default_tb = thread_batch_choices[-1] if thread_batch_choices else default_threads
    default_b = 512 if 512 in batch_choices else batch_choices[0]
    default_ub = 512 if 512 in micro_batch_choices else micro_batch_choices[0]
    default_fitt = fitt_choices[0]
    default_ck = "f16" if "f16" in cache_k_choices else cache_k_choices[0]
    default_cv = "f16" if "f16" in cache_v_choices else cache_v_choices[0]

    baseline_trial = {
        "threads": default_threads,
        "thread_batch": default_tb,
        "batch": default_b,
        "micro_batch": default_ub,
        "fitt": default_fitt,
        "cache_k": default_ck,
        "cache_v": default_cv,
    }
    if is_speculative and spec_draft_n_choices:
        baseline_trial["spec_draft_n"] = 4 if 4 in spec_draft_n_choices else spec_draft_n_choices[0]
        baseline_trial["spec_draft_p_min"] = 0.4

    try:
        study.enqueue_trial(baseline_trial)
    except Exception as e:
        print(f"[DEBUG] Could not enqueue baseline trial: {e}")

    try:
        study.optimize(objective, n_trials=n_trials, callbacks=[callback])
    except KeyboardInterrupt:
        print("[INFO] Study interrupted by user.")
    except optuna.TrialPruned:
        print("[INFO] Study pruned/cancelled.")

    if study.best_trial is None:
        print("[INFO] No successful trials completed.")
        return None

    best_params = study.best_trial.params
    best_score = study.best_value
    best_pp = study.best_trial.user_attrs.get("pp", 0.0)
    best_tg = study.best_trial.user_attrs.get("tg", 0.0)

    final_config = {
        "threads": best_params["threads"],
        "thread_batch": best_params.get("thread_batch", best_params["threads"]),
        "batch": best_params["batch"],
        "micro_batch": best_params.get("micro_batch", best_params["batch"]),
        "fitt": best_params["fitt"],
        "cache_k": best_params["cache_k"],
        "cache_v": best_params["cache_v"],
        "mtp": is_speculative,
        "spec_draft_n": best_params.get("spec_draft_n") if is_speculative else None,
        "spec_draft_p_min": best_params.get("spec_draft_p_min.2") if is_speculative else None,
        "draft_model_path": draft_model_path,
        "baseline_score": f"{baseline_score:.2f}",
        "best_score": f"{best_score:.2f}",
        "best_pp": f"{best_pp:.2f}",
        "best_tg": f"{best_tg:.2f}",
    }
    return final_config


def _print_progress(run_idx, total, step_name, last_score, best_score, baseline_score):
    print(f"[{run_idx}/{total}] {step_name} | Last: {last_score:.2f} | Best: {best_score:.2f} | Baseline: {baseline_score:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Optuna Bayesian optimisation test harness for llama-launcher")
    parser.add_argument("--model", required=True, help="Path to model (gguf) to benchmark")
    parser.add_argument("--server", default="llama-server.exe", help="llama-server executable")
    parser.add_argument("--context", type=int, default=16384, help="Context size")
    parser.add_argument("--trials", type=int, default=6, help="Number of Optuna trials")
    parser.add_argument("--avg", type=int, default=1, help="Average runs per trial to reduce noise")
    parser.add_argument("--mtp", action="store_true", help="Enable multi-token prediction (MTP) optimization; requires MTP-capable model")
    parser.add_argument("--draft", default=None, help="Path to separate draft model GGUF for speculative decoding")
    args = parser.parse_args()

    if optuna is None:
        print("optuna is not installed. Install with: pip install optuna")
        return

    print("Starting Bayesian (Optuna) prototype harness")
    if args.mtp:
        print("  [MTP enabled] optimizing spec_draft_n parameter (1-7)")
    if args.draft:
        print(f"  [Speculative decoding enabled] draft model: {args.draft}")
    start = time.time()
    final = run_bayesian_optimisation(
        args.model, args.server, context_size=args.context,
        metric_weight=0.5, n_trials=args.trials, avg_runs=args.avg,
        progress_callback=_print_progress, mtp=args.mtp, draft_model_path=args.draft
    )
    elapsed = time.time() - start
    if final:
        print("\n=== Final Config ===")
        for k, v in final.items():
            print(f"{k}: {v}")
        print(f"Elapsed: {elapsed:.1f}s")
    else:
        print("No result produced.")


if __name__ == "__main__":
    main()
