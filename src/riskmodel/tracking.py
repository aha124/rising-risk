"""MLflow logging to a local file store (./mlruns, not committed)."""

from __future__ import annotations

import logging
import math
import os
from contextlib import contextmanager
from pathlib import Path

from . import config, dataset

EXPERIMENT = "rising-risk"


def _mlflow():
    # MLflow 3 asks for an explicit opt-in before it writes to a plain folder.
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    logging.disable(logging.INFO)  # MLflow prints setup notes at import time
    try:
        import mlflow
    finally:
        logging.disable(logging.NOTSET)
    logging.getLogger("mlflow").setLevel(logging.WARNING)
    return mlflow


@contextmanager
def run(name: str, params: dict | None = None, tags: dict | None = None):
    """Start a run tagged with the commit and the hash of the mart it read."""
    mlflow = _mlflow()
    paths = config.paths()
    paths.mlruns_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(paths.mlruns_dir.as_uri())
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=name) as active:
        mlflow.set_tags(
            {
                "git_sha": dataset.git_sha(),
                "data_sha256": dataset.file_sha256(paths.mart),
                "model_version": config.MODEL_VERSION,
                "fast_mode": str(config.settings().fast),
                **(tags or {}),
            }
        )
        if params:
            mlflow.log_params({k: _plain(v) for k, v in params.items()})
        yield Logger(mlflow, active)


class Logger:
    def __init__(self, mlflow, active):
        self._mlflow = mlflow
        self.run_id = active.info.run_id

    def metrics(self, values: dict, step: int | None = None) -> None:
        clean = {
            k: float(v) for k, v in values.items() if v is not None and not math.isnan(float(v))
        }
        if clean:
            self._mlflow.log_metrics(clean, step=step)

    def params(self, values: dict) -> None:
        self._mlflow.log_params({k: _plain(v) for k, v in values.items()})

    def artifact(self, path: Path) -> None:
        if Path(path).exists():
            self._mlflow.log_artifact(str(path))


def _plain(v):
    if isinstance(v, (list, tuple)):
        return ",".join(str(x) for x in v)
    return v
