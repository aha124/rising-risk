"""Command line entry point.

    python -m riskmodel.run all        every stage, in order
    python -m riskmodel.run train      fit the models, write artifacts/model/
    python -m riskmodel.run evaluate   metrics.json and sanity.json
    python -m riskmodel.run explain    explain.json
    python -m riskmodel.run fairness   fairness.json
    python -m riskmodel.run capacity   capacity.json
    python -m riskmodel.run drift      drift.json
    python -m riskmodel.run export     docs/data/*.json

Set RISKMODEL_FAST=1 for a quick run (few trees, no Optuna, 50 bootstrap resamples).
Every stage after `train` reads the saved model, so stages can be rerun on their own.
"""

from __future__ import annotations

import argparse
import sys
import time

from . import config, dataset


def _stages() -> dict:
    from . import (
        capacity,
        drift,
        evaluate,
        explain,
        export_browser_model,
        export_site,
        fairness,
        train,
    )

    def export():
        export_browser_model.run()
        export_site.run()

    return {
        "train": train.run,
        "evaluate": evaluate.run,
        "explain": explain.run,
        "fairness": fairness.run,
        "capacity": capacity.run,
        "drift": drift.run,
        "export": export,
    }


def main(argv: list[str] | None = None) -> int:
    stages = _stages()
    parser = argparse.ArgumentParser(
        prog="python -m riskmodel.run",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("stage", choices=["all", *stages])
    args = parser.parse_args(argv)

    paths, s = config.paths(), config.settings()
    print(
        f"mart {paths.mart}\nmodel {paths.model_dir}\nreports {paths.reports_dir}\n"
        f"site data {paths.site_data_dir}\nfast mode {s.fast}, {s.n_bootstrap} bootstrap "
        f"resamples, {s.n_trials} Optuna trials, at most {s.max_trees} trees"
    )
    todo = list(stages) if args.stage == "all" else [args.stage]
    timings = {}
    started = time.time()
    for name in todo:
        t0 = time.time()
        stages[name]()
        timings[name] = round(time.time() - t0, 1)
        print(f"  [{name}] {timings[name]} s")
    total = round(time.time() - started, 1)
    print(f"done in {total} s")
    if args.stage == "all":
        dataset.write_json(
            paths.reports_dir / "run_summary.json",
            {
                "stage_seconds": timings,
                "total_seconds": total,
                "fast_mode": s.fast,
                "n_bootstrap": s.n_bootstrap,
                "n_trials": s.n_trials,
                "git_sha": dataset.git_sha(),
                "data_sha256": dataset.file_sha256(paths.mart),
            },
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
