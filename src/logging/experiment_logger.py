"""BaseExperimentLogger and optional W&B / Comet / DB backends."""

from __future__ import annotations

import json
import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional


class BaseExperimentLogger(ABC):
    """Unified interface for experiment run logging."""

    def __init__(self, run_id: str, experiment_name: str) -> None:
        self.run_id = run_id
        self.experiment_name = experiment_name

    @abstractmethod
    def log_params(self, params: Dict[str, Any]) -> None:
        raise NotImplementedError

    @abstractmethod
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def log_artifact(self, path: str, name: Optional[str] = None) -> None:
        raise NotImplementedError

    def finish(self) -> None:
        """Optional cleanup hook."""
        return None


class LocalExperimentLogger(BaseExperimentLogger):
    """Write params/metrics/artifact references under a run directory."""

    def __init__(self, run_dir: str | Path, experiment_name: str = "local") -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        super().__init__(run_id=self.run_dir.name, experiment_name=experiment_name)
        self._metrics_path = self.run_dir / "logged_metrics.jsonl"
        self._params_path = self.run_dir / "logged_params.json"

    def log_params(self, params: Dict[str, Any]) -> None:
        existing: Dict[str, Any] = {}
        if self._params_path.exists():
            existing = json.loads(self._params_path.read_text(encoding="utf-8"))
        existing.update(params)
        self._params_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        record = {"step": step, "metrics": metrics}
        with open(self._metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")

    def log_artifact(self, path: str, name: Optional[str] = None) -> None:
        manifest_path = self.run_dir / "artifacts.json"
        manifest: list[Dict[str, str]] = []
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.append({"path": path, "name": name or Path(path).name})
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


class WandbExperimentLogger(BaseExperimentLogger):
    """Weights & Biases backend (optional dependency)."""

    def __init__(self, project: str, run_name: Optional[str] = None) -> None:
        try:
            import wandb  # type: ignore
        except ImportError as exc:
            raise ImportError("WandbExperimentLogger requires `pip install wandb`") from exc

        self._wandb = wandb
        self._run = wandb.init(project=project, name=run_name, reinit=True)
        super().__init__(run_id=self._run.id, experiment_name=project)

    def log_params(self, params: Dict[str, Any]) -> None:
        self._wandb.config.update(params)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        self._wandb.log(metrics, step=step)

    def log_artifact(self, path: str, name: Optional[str] = None) -> None:
        artifact = self._wandb.Artifact(name or Path(path).stem, type="dataset")
        artifact.add_file(path)
        self._run.log_artifact(artifact)

    def finish(self) -> None:
        self._run.finish()


class CometExperimentLogger(BaseExperimentLogger):
    """Comet ML backend (optional dependency)."""

    def __init__(self, project_name: str, workspace: Optional[str] = None) -> None:
        try:
            from comet_ml import Experiment  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "CometExperimentLogger requires `pip install comet_ml`"
            ) from exc

        self._experiment = Experiment(project_name=project_name, workspace=workspace)
        super().__init__(
            run_id=str(self._experiment.id),
            experiment_name=project_name,
        )

    def log_params(self, params: Dict[str, Any]) -> None:
        self._experiment.log_parameters(params)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        self._experiment.log_metrics(metrics, step=step)

    def log_artifact(self, path: str, name: Optional[str] = None) -> None:
        self._experiment.log_asset(path, file_name=name)

    def finish(self) -> None:
        self._experiment.end()


class DatabaseExperimentLogger(BaseExperimentLogger):
    """Minimal SQLite experiment log."""

    def __init__(self, db_path: str | Path, experiment_name: str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._init_schema()
        super().__init__(run_id=experiment_name, experiment_name=experiment_name)

    def _init_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS params (
                experiment TEXT, key TEXT, value TEXT,
                PRIMARY KEY (experiment, key)
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS metrics (
                experiment TEXT, step INTEGER, key TEXT, value REAL
            )
            """
        )
        self._conn.commit()

    def log_params(self, params: Dict[str, Any]) -> None:
        for key, value in params.items():
            self._conn.execute(
                "INSERT OR REPLACE INTO params VALUES (?, ?, ?)",
                (self.experiment_name, key, json.dumps(value)),
            )
        self._conn.commit()

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        step_val = -1 if step is None else step
        for key, value in metrics.items():
            self._conn.execute(
                "INSERT INTO metrics VALUES (?, ?, ?, ?)",
                (self.experiment_name, step_val, key, float(value)),
            )
        self._conn.commit()

    def log_artifact(self, path: str, name: Optional[str] = None) -> None:
        self.log_params({f"artifact:{name or Path(path).name}": path})

    def finish(self) -> None:
        self._conn.close()


def create_experiment_logger(backend: str, **kwargs: Any) -> BaseExperimentLogger:
    """Factory for experiment loggers."""
    normalized = backend.lower()
    if normalized in {"local", "filesystem"}:
        return LocalExperimentLogger(**kwargs)
    if normalized == "wandb":
        return WandbExperimentLogger(**kwargs)
    if normalized == "comet":
        return CometExperimentLogger(**kwargs)
    if normalized in {"db", "sqlite", "database"}:
        return DatabaseExperimentLogger(**kwargs)
    raise ValueError(f"Unknown experiment logger backend: {backend}")
