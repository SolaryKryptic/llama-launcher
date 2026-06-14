"""Modular optimisation entry points for llama-launcher

GUI code should call OptimisationService only, optimisation algorithms are in
their own scripts and are wired here
"""

from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, Optional


DEFAULT_PERPLEXITY_FILE = "perplexity_corpus.txt"


METHOD_BAYESIAN = "bayesian"
AVAILABLE_METHODS = (METHOD_BAYESIAN,)


@dataclass
class OptimisationRequest:
    model_path: str
    server_exe: str
    perplexity_exe: Optional[str] = None
    perplexity_file: str = DEFAULT_PERPLEXITY_FILE
    context_size: int = 16384
    metric_weight: float = 0.5
    ppl_threshold_percent: float = 3.0
    method: str = METHOD_BAYESIAN
    draft_model_path: Optional[str] = None
    mtp: bool = False
    trials: int = 40
    avg_runs: int = 1
    seed: int = 42
    time_budget: Optional[float] = None
    trial_csv_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class OptimisationService:
    """dispatcher"""

    def run(
        self,
        request: OptimisationRequest,
        progress_callback: Optional[Callable[..., None]] = None,
        cancel_flag: Optional[list] = None,
        proc_holder: Optional[list] = None,
    ):
        method = (request.method or METHOD_BAYESIAN).lower()
        if method == METHOD_BAYESIAN:
            return self._run_bayesian(request, progress_callback, cancel_flag, proc_holder)
        raise ValueError(f"Unknown optimisation method: {method}")

    def _run_bayesian(self, request, progress_callback, cancel_flag, proc_holder):
        import bayesian as opt

        return opt.run_bayesian_optimisation(
            model_path=request.model_path,
            server_exe=request.server_exe,
            perplexity_exe=request.perplexity_exe,
            perplexity_file=request.perplexity_file,
            context_size=request.context_size,
            metric_weight=request.metric_weight,
            ppl_threshold=request.ppl_threshold_percent / 100.0,
            n_trials=request.trials,
            avg_runs=request.avg_runs,
            progress_callback=progress_callback,
            cancel_flag=cancel_flag,
            proc_holder=proc_holder,
            draft_model_path=request.draft_model_path,
            mtp=request.mtp,
            seed=request.seed,
            time_budget=request.time_budget,
            trial_csv_path=request.trial_csv_path,
        )
