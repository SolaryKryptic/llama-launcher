"""
Bayesian optimisation prototype harness for llama-launcher.
Uses Optuna (TPE) to search mixed parameter space and wraps
`optimiser_script.run_benchmark` as the objective.

Run as a standalone script for quick tests. Example:

    python bayesian.py --model "path/to/model.gguf"

If `optuna` is not installed, the script prints instructions to install it
"""

import argparse
import csv
import os
import time

try:
    import optuna
except Exception:
    optuna = None

import optimiser_script as opt


def run_bayesian_optimisation(model_path, server_exe, context_size=16384,
                              metric_weight=0.1, n_trials=40, avg_runs=1,
                              progress_callback=None, cancel_flag=None, proc_holder=None,
                              mtp=False, draft_model_path=None, seed=42,
                              time_budget=None, trial_csv_path=None,
                              perplexity_exe=None, perplexity_file=opt.PERPLEXITY_FILE,
                              ppl_threshold=opt.PPL_THRESHOLD):
    """Run an Optuna (TPE) search over the same parameter families used by
    `optimiser_script.run_benchmark`. Returns a final_config dict
    matching the existing optimiser's returned structure, or None on failure.
    
    If mtp=True or draft_model_path is set, includes spec_draft_n parameter for MTP
    """
    if optuna is None:
        raise RuntimeError("optuna is not available —> install with `pip install optuna`")

    if cancel_flag is None:
        cancel_flag = [False]

    is_speculative = mtp or bool(draft_model_path)

    if n_trials <= 0:
        print("[ERROR] n_trials must be positive.")
        return None

    start_time = time.time()

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
    print(f"[INFO] Base command baseline score: {baseline_score:.2f} (no tuned -t/-tb/-b/-ub/-fa/-fit/-ct flags).")

    baseline_ppl = None
    baseline_ppl_flags = []
    baseline_ppl_f16_oom = False
    if perplexity_exe:
        baseline_ppl, baseline_ppl_flags, baseline_ppl_f16_oom = opt.run_perplexity_baseline(
            model_path, perplexity_exe, context_size, corpus_file=perplexity_file, spec_active=is_speculative
        )
        if baseline_ppl_f16_oom:
            print("[INFO] Baseline f16 perplexity OOM; using q8_0 baseline for quality gate.")
        elif baseline_ppl is None:
            print("[WARN] Baseline perplexity unavailable; cache quantisation quality gate will be skipped.")

    params = opt.build_thread_list()
    threads_choices = params.get("threads") or [max(1, params.get("cap_limit", 1))]
    cap_limit = params.get("cap_limit", 1) or max(1, threads_choices[-1] if threads_choices else 1)
    threads_choices = [t for t in threads_choices if 1 <= t <= cap_limit] or [max(1, cap_limit)]
    thread_batch_choices = params.get("thread_batch") or threads_choices
    thread_batch_choices = [t for t in thread_batch_choices if 1 <= t <= max(1, cap_limit)] or threads_choices
    batch_choices = params.get("batch_sizes", [128, 256, 512, 1024, 2048])
    micro_batch_choices = params.get("micro_batch_sizes", [128, 256, 512, 1024, 2048])
    fitt_choices = params.get("fitt_targets", [50])
    cache_k_choices = params.get("cache_k_types", ["f16", "q8_0", "q5_0", "q4_0"])
    cache_v_choices = params.get("cache_v_types", ["f16", "q8_0", "q5_0", "q4_0"])
    if baseline_ppl_f16_oom:
        cache_k_choices = [t for t in cache_k_choices if t != "f16"] or ["q8_0"]
        cache_v_choices = [t for t in cache_v_choices if t != "f16"] or ["q8_0"]
    cache_kd_choices = cache_k_choices if is_speculative else None
    cache_vd_choices = cache_v_choices if is_speculative else None
    spec_draft_n_choices = params.get("spec_draft_n", list(range(1, 5))) if is_speculative else None
    baseline_cache = {
        "cache_k": "q8_0" if baseline_ppl_f16_oom else "f16",
        "cache_v": "q8_0" if baseline_ppl_f16_oom else "f16",
        "cache_kd": "q8_0" if is_speculative and baseline_ppl_f16_oom else "f16",
        "cache_vd": "q8_0" if is_speculative and baseline_ppl_f16_oom else "f16",
    }
    best_quality_score = baseline_score
    best_quality_trial = None

    def next_higher_thread_batch(current_t):
        for choice in sorted(set(thread_batch_choices)):
            if choice > current_t:
                return choice
        return max(thread_batch_choices)

    def choose_thread_batch(raw_tb, current_t):
        valid_choices = [choice for choice in sorted(set(thread_batch_choices)) if choice >= current_t]
        if not valid_choices:
            return max(thread_batch_choices)
        return min(valid_choices, key=lambda choice: abs(choice - raw_tb))
    
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            seed=seed,
            n_startup_trials=min(25, max(15, n_trials // 4)),
        ),
    )
    
    trial_log = None
    if trial_csv_path:
        os.makedirs(os.path.dirname(os.path.abspath(trial_csv_path)), exist_ok=True)
        trial_log = open(trial_csv_path, "w", newline="", encoding="utf-8")

    csv_fieldnames = [
        "number", "state", "value", "pp", "tg", "best_quality_score", "error",
        "discarded_by", "perplexity", "ppl_validated", "ppl_skipped_reason",
        "ppl_cache_k", "ppl_cache_v", "ppl_cache_kd", "ppl_cache_vd",
        "trial_role", "effective_thread_batch", "raw_thread_batch",
        "param_threads", "param_thread_batch", "param_batch", "param_micro_batch",
        "param_fitt", "param_cache_k", "param_cache_v",
    ]
    if is_speculative:
        csv_fieldnames += ["param_spec_draft_n", "param_spec_draft_p_min", "param_cache_kd", "param_cache_vd"]

    def callback(study, trial):
        """Called after each trial completes. Stop on time budget only."""
        try:
            current_best = best_quality_score
            display_best = current_best

            if time_budget and (time.time() - start_time) >= time_budget:
                print(f"[INFO] Early stopping: time budget reached ({time_budget:.0f}s).")
                study.stop()

            if trial_log:
                trial_role = trial.user_attrs.get("trial_role", "trial")
                step_name = "DefaultConfig" if trial_role == "default_config" else f"Trial-{trial.number+1}"
                row = {
                    "number": trial.number,
                    "state": trial.state.name,
                    "value": trial.value,
                    "pp": trial.user_attrs.get("pp"),
                    "tg": trial.user_attrs.get("tg"),
                    "best_quality_score": best_quality_score,
                    "error": trial.user_attrs.get("error"),
                    "discarded_by": trial.user_attrs.get("discarded_by"),
                    "perplexity": trial.user_attrs.get("perplexity"),
                    "ppl_validated": trial.user_attrs.get("ppl_validated", False),
                    "ppl_skipped_reason": trial.user_attrs.get("ppl_skipped_reason"),
                    "ppl_cache_k": trial.user_attrs.get("ppl_cache_k"),
                    "ppl_cache_v": trial.user_attrs.get("ppl_cache_v"),
                    "ppl_cache_kd": trial.user_attrs.get("ppl_cache_kd"),
                    "ppl_cache_vd": trial.user_attrs.get("ppl_cache_vd"),
                    "trial_role": trial_role,
                    "effective_thread_batch": trial.user_attrs.get("thread_batch_effective"),
                    "raw_thread_batch": trial.user_attrs.get("thread_batch_raw"),
                }
                row.update({f"param_{k}": v for k, v in trial.params.items()})
                writer = csv.DictWriter(trial_log, fieldnames=csv_fieldnames, extrasaction="ignore")
                if trial_log.tell() == 0:
                    writer.writeheader()
                writer.writerow(row)
                trial_log.flush()
        except Exception as e:
            print(f"[DEBUG] Bayesian callback failed: {e}")

    def objective(trial):
        nonlocal best_quality_score, best_quality_trial
        if cancel_flag and cancel_flag[0]:
            raise optuna.TrialPruned()

        t = trial.suggest_categorical("threads", threads_choices)
        raw_tb = trial.suggest_int("thread_batch", min(thread_batch_choices), max(thread_batch_choices))
        tb = choose_thread_batch(raw_tb, t)
        trial.set_user_attr("thread_batch_raw", raw_tb)
        trial.set_user_attr("thread_batch_effective", tb)
        b = trial.suggest_categorical("batch", batch_choices)
        ub_candidate = trial.suggest_categorical("micro_batch", micro_batch_choices)
        ub = min(ub_candidate, b)
        fitt = trial.suggest_categorical("fitt", fitt_choices)
        ck = trial.suggest_categorical("cache_k", cache_k_choices)
        cv = trial.suggest_categorical("cache_v", cache_v_choices)
        ckd = trial.suggest_categorical("cache_kd", cache_kd_choices) if is_speculative else None
        cvd = trial.suggest_categorical("cache_vd", cache_vd_choices) if is_speculative else None
        sdn = trial.suggest_categorical("spec_draft_n", spec_draft_n_choices) if is_speculative else None
        sdp = trial.suggest_float("spec_draft_p_min", 0.0, 0.7, step=0.1) if is_speculative else None

        # trial_role: compare effective params (using tb, not tb_candidate) against baseline
        effective_params = {
            "threads": t, "thread_batch": tb, "batch": b,
            "micro_batch": ub, "fitt": fitt, "cache_k": ck, "cache_v": cv,
        }
        if is_speculative:
            effective_params["cache_kd"] = ckd
            effective_params["cache_vd"] = cvd
            effective_params["spec_draft_n"] = sdn
            effective_params["spec_draft_p_min"] = round(sdp, 1) if sdp is not None else None
        baseline_effective = dict(baseline_trial)
        if is_speculative:
            baseline_effective["cache_kd"] = baseline_trial["cache_kd"]
            baseline_effective["cache_vd"] = baseline_trial["cache_vd"]
        trial_role = "default_config" if all(effective_params.get(k) == v for k, v in baseline_effective.items()) else "trial"
        trial.set_user_attr("trial_role", trial_role)

        cache_quant_changed = opt.cache_differs_from_baseline(
            {"cache_k": ck, "cache_v": cv, "cache_kd": ckd, "cache_vd": cvd},
            baseline_cache
        )
        candidate_cache = {
            "cache_k": ck,
            "cache_v": cv,
            "cache_kd": ckd,
            "cache_vd": cvd,
        }

        def penalized_failure_score(raw_score=None):
            if raw_score is None:
                return -1_000_000.0
            return -1_000_000.0 + raw_score

        def report_progress(last_score):
            if progress_callback:
                step_name = "DefaultConfig" if trial_role == "default_config" else f"Trial-{trial.number+1}"
                progress_callback(trial.number + 1, n_trials, step_name, last_score, best_quality_score, baseline_score)

        def benchmark_with_retry():
            last_error = None
            for attempt in range(1, 4):
                if cancel_flag and cancel_flag[0]:
                    raise optuna.TrialPruned()

                try:
                    trial_avg_runs = avg_runs
                    pp, tg = opt.run_benchmark(
                        model_path, server_exe, context_size,
                        proc_holder=proc_holder,
                        t=t, tb=tb, b=b, ub=ub, fitt=fitt,
                        cache_k=ck, cache_v=cv,
                        cache_kd=ckd, cache_vd=cvd,
                        mtp=is_speculative, spec_draft_n=sdn,
                        avg_runs=trial_avg_runs, draft_model_path=draft_model_path,
                        spec_draft_p_min=sdp
                    )
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    last_error = f"attempt {attempt}/3 failed with exception: {e}"
                    print(f"[DEBUG] {last_error}")
                    if attempt < 3:
                        time.sleep(3)
                    continue

                if pp == 0 and tg == 0:
                    last_error = f"attempt {attempt}/3 returned zero speed"
                    print(f"[DEBUG] {last_error}")
                    if attempt < 3:
                        time.sleep(3)
                    continue

                return pp, tg, None

            return 0.0, 0.0, f"all 3 attempts failed; last: {last_error}"

        pp, tg, bench_error = benchmark_with_retry()
        if bench_error:
            failure_score = penalized_failure_score()
            trial.set_user_attr("error", bench_error)
            trial.set_user_attr("discarded_by", "benchmark_failure")
            report_progress(failure_score)
            return failure_score

        score = opt.calculate_score(pp, tg, metric_weight)
        step_name = "DefaultConfig" if trial_role == "default_config" else f"Trial-{trial.number+1}"
        print(f"[DEBUG] Speed score for {step_name}: pp={pp:.2f}, tg={tg:.2f}, score={score:.2f}, baseline={baseline_score:.2f}, metric_weight={metric_weight:.2f}.")

        # Save prompt processing and text generation speeds to avoid a final verification run
        trial.set_user_attr("pp", pp)
        trial.set_user_attr("tg", tg)

        def update_best_quality(last_score):
            nonlocal best_quality_score, best_quality_trial
            if last_score > best_quality_score:
                best_quality_score = last_score
                best_quality_trial = trial

        step_name = "DefaultConfig" if trial_role == "default_config" else f"Trial-{trial.number+1}"
        if not cache_quant_changed:
            trial.set_user_attr("ppl_skipped_reason", "cache_matches_baseline")
            trial.set_user_attr("ppl_validated", True)
            print(f"[INFO] {step_name} matches baseline cache quantisation; skipping PPL benchmark and allowing score to update best_quality_score.")
            update_best_quality(score)
            report_progress(score)
            return score

        if score <= best_quality_score:
            trial.set_user_attr("ppl_skipped_reason", "not_faster_than_current_best")
            print(f"[INFO] {step_name} did not beat current PPL-validated best score; skipping PPL benchmark: score {score:.2f}, best_quality {best_quality_score:.2f}.")
            report_progress(score)
            return score

        if baseline_ppl is None or not perplexity_exe:
            failure_score = penalized_failure_score(score)
            trial.set_user_attr("ppl_skipped_reason", "baseline_ppl_unavailable")
            trial.set_user_attr("discarded_by", "baseline_ppl_unavailable")
            print(f"[INFO] {step_name} beat current PPL-validated best score but baseline PPL is unavailable; returning penalty score.")
            report_progress(failure_score)
            return failure_score

        ppl_flags = list(baseline_ppl_flags)
        ppl_flags += opt.build_perplexity_cache_flags(
            candidate_cache["cache_k"],
            candidate_cache["cache_v"],
            spec_active=False,
        )
        print(f"[DEBUG] PPL cache flags: {ppl_flags}")
        trial.set_user_attr("ppl_cache_k", ck)
        trial.set_user_attr("ppl_cache_v", cv)
        trial.set_user_attr("ppl_cache_kd", None)
        trial.set_user_attr("ppl_cache_vd", None)
        ppl, ppl_code, ppl_stderr = opt.run_perplexity(
            model_path, perplexity_exe, context_size,
            flags=ppl_flags, corpus_file=perplexity_file
        )
        step_name = "DefaultConfig" if trial_role == "default_config" else f"Trial-{trial.number+1}"
        print(f"[DEBUG] Perplexity parsed for {step_name}: PPL={ppl if ppl is not None else 'unparsed'}, baseline={baseline_ppl:.4f}, threshold={ppl_threshold * 100.0:.1f}%.")
        accepted = opt.passes_perplexity_gate(ppl, baseline_ppl, ppl_threshold)
        trial.set_user_attr("perplexity", ppl)
        trial.set_user_attr("ppl_validated", accepted)
        if not accepted:
            step_name = "DefaultConfig" if trial_role == "default_config" else f"Trial-{trial.number+1}"
            required = baseline_ppl * (1.0 + ppl_threshold)
            failure_score = penalized_failure_score(score)
            print(f"[INFO] Discarded {step_name} by perplexity_gate: ppl {ppl}, baseline {baseline_ppl:.4f}, required <= {required:.4f}.")
            trial.set_user_attr("discarded_by", "perplexity_gate")
            report_progress(failure_score)
            return failure_score

        update_best_quality(score)
        report_progress(score)
        return score

    # Warm-start with a sensible baseline configuration
    default_threads = threads_choices[-1] if threads_choices else max(1, params.get("cap_limit", 1))
    default_tb = next_higher_thread_batch(default_threads)
    default_b = 512 if 512 in batch_choices else batch_choices[0]
    default_ub = min(default_b, 512 if 512 in micro_batch_choices else micro_batch_choices[0])
    default_fitt = fitt_choices[0]
    default_ck = "f16" if "f16" in cache_k_choices else cache_k_choices[0]
    default_cv = "f16" if "f16" in cache_v_choices else cache_v_choices[0]
    default_cache_kd = ("f16" if "f16" in cache_kd_choices else cache_kd_choices[0]) if is_speculative else None
    default_cache_vd = ("f16" if "f16" in cache_vd_choices else cache_vd_choices[0]) if is_speculative else None

    baseline_trial = {
        "threads": default_threads,
        "thread_batch": default_tb,
        "batch": default_b,
        "micro_batch": default_ub,
        "fitt": default_fitt,
        "cache_k": default_ck,
        "cache_v": default_cv,
    }
    if is_speculative:
        baseline_trial["cache_kd"] = default_cache_kd
        baseline_trial["cache_vd"] = default_cache_vd
    if is_speculative and spec_draft_n_choices:
        baseline_trial["spec_draft_n"] = 4 if 4 in spec_draft_n_choices else spec_draft_n_choices[0]
        baseline_trial["spec_draft_p_min"] = 0.4

    try:
        study.enqueue_trial(baseline_trial)
        print(
            "[INFO] Enqueued default config trial, separate from base baseline: "
            f"-t {default_threads} -tb {default_tb} -b {default_b} -ub {default_ub}"
        )
    except Exception as e:
        print(f"[DEBUG] Could not enqueue baseline trial: {e}")

    try:
        study.optimize(objective, n_trials=n_trials, callbacks=[callback])
    except KeyboardInterrupt:
        print("[INFO] Study interrupted by user.")
    except optuna.TrialPruned:
        print("[INFO] Study pruned/cancelled.")
    finally:
        if trial_log:
            trial_log.close()

    successful_trials = [
        trial for trial in study.trials
        if trial.value is not None and trial.value > float("-inf")
    ]
    if not successful_trials:
        print("[INFO] No successful trials completed.")
        return None

    default_trial = next(
        (trial for trial in study.trials if trial.user_attrs.get("trial_role") == "default_config"),
        None,
    )
    best_trial = best_quality_trial
    if best_trial is None:
        print("[INFO] No PPL-validated trial completed; returning baseline command as result.")
        return {
            "use_baseline_command": True,
            "baseline_is_base_command": True,
            "threads": -1,
            "thread_batch": -1,
            "batch": default_b,
            "micro_batch": default_ub,
            "fitt": default_fitt,
            "cache_k": "f16",
            "cache_v": "f16",
            "cache_type_kd": "f16",
            "cache_type_vd": "f16",
            "mtp": is_speculative,
            "spec_enabled": is_speculative,
            "spec_type": "draft-mtp" if is_speculative else "",
            "spec_draft_n": baseline_trial.get("spec_draft_n") if is_speculative else None,
            "spec_draft_p_min": baseline_trial.get("spec_draft_p_min") if is_speculative else None,
            "draft_model_path": draft_model_path,
            "flash_attention": False,
            "fit_on": False,
            "baseline_pp": f"{base_pp:.2f}",
            "baseline_tg": f"{base_tg:.2f}",
            "baseline_score": f"{baseline_score:.2f}",
            "best_score": f"{baseline_score:.2f}",
            "best_quality_score": f"{best_quality_score:.2f}",
            "best_pp": f"{base_pp:.2f}",
            "best_tg": f"{base_tg:.2f}",
            "best_ppl": None,
            "baseline_ppl": baseline_ppl,
            "ppl_threshold": ppl_threshold,
        }

    best_score = best_quality_score

    best_params = best_trial.params

    final_config = {
        "threads": best_params["threads"],
        "thread_batch": best_trial.user_attrs.get(
            "thread_batch_effective",
            next_higher_thread_batch(best_params.get("thread_batch", best_params["threads"])),
        ),
        "batch": best_params["batch"],
        "micro_batch": best_params.get("micro_batch", best_params["batch"]),
        "fitt": best_params["fitt"],
        "cache_k": best_params["cache_k"],
        "cache_v": best_params["cache_v"],
        "mtp": is_speculative,
        "spec_enabled": is_speculative,
        "spec_type": "draft-mtp" if is_speculative else "",
        "spec_draft_n": best_params.get("spec_draft_n") if is_speculative else None,
        "spec_draft_p_min": best_params.get("spec_draft_p_min") if is_speculative else None,
        "draft_model_path": draft_model_path,
        "flash_attention": True,
        "fit_on": True,
        "baseline_is_base_command": True,
        "baseline_pp": f"{base_pp:.2f}",
        "baseline_tg": f"{base_tg:.2f}",
        "baseline_score": f"{baseline_score:.2f}",
        "baseline_ppl": baseline_ppl,
        "ppl_threshold": ppl_threshold,
        "best_score": f"{best_score:.2f}",
        "best_quality_score": f"{best_quality_score:.2f}",
        "best_pp": f"{best_trial.user_attrs.get('pp', 0.0):.2f}",
        "best_tg": f"{best_trial.user_attrs.get('tg', 0.0):.2f}",
        "best_ppl": best_quality_trial.user_attrs.get("perplexity") if best_quality_trial is not None else None,
    }
    if is_speculative:
        final_config.update({
            "cache_type_kd": best_params.get("cache_kd", "f16"),
            "cache_type_vd": best_params.get("cache_vd", "f16"),
        })
    if default_trial is not None:
        final_config["default_trial_score"] = f"{default_trial.value:.2f}"
        final_config["default_trial_pp"] = f"{default_trial.user_attrs.get('pp', 0.0):.2f}"
        final_config["default_trial_tg"] = f"{default_trial.user_attrs.get('tg', 0.0):.2f}"
    return final_config


def _print_progress(run_idx, total, step_name, last_score, best_score, baseline_score):
    print(f"[{run_idx}/{total}] {step_name} | Last: {last_score:.2f} | Best PPL: {best_score:.2f} | Baseline: {baseline_score:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Optuna Bayesian optimisation test harness for llama-launcher")
    parser.add_argument("--model", required=True, help="Path to model (gguf) to benchmark")
    parser.add_argument("--server", default="llama-server.exe", help="llama-server executable")
    parser.add_argument("--context", type=int, default=16384, help="Context size")
    parser.add_argument("--trials", type=int, default=6, help="Number of Optuna trials")
    parser.add_argument("--avg", type=int, default=1, help="Average runs per trial to reduce noise")
    parser.add_argument("--mtp", action="store_true", help="Enable multi-token prediction (MTP) optimization; requires MTP-capable model")
    parser.add_argument("--draft", default=None, help="Path to separate draft model GGUF for speculative decoding")
    parser.add_argument("--seed", type=int, default=42, help="Optuna sampler seed")
    parser.add_argument("--time-budget", type=float, default=None, help="Stop after N seconds; current trial may finish first")
    parser.add_argument("--trial-csv", default=None, help="Write completed trial params/results to CSV")
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
        progress_callback=_print_progress, mtp=args.mtp, draft_model_path=args.draft,
        seed=args.seed, time_budget=args.time_budget, trial_csv_path=args.trial_csv
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