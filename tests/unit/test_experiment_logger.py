"""Unit tests for BaseExperimentLogger backends."""

import json

from src.logging.experiment_logger import (
    DatabaseExperimentLogger,
    LocalExperimentLogger,
    create_experiment_logger,
)


def test_local_logger_writes_params_and_metrics(tmp_path):
    logger = LocalExperimentLogger(run_dir=tmp_path, experiment_name="unit")
    logger.log_params({"lr": 0.01})
    logger.log_metrics({"loss": 1.5}, step=1)

    params = json.loads((tmp_path / "logged_params.json").read_text(encoding="utf-8"))
    assert params["lr"] == 0.01

    lines = (tmp_path / "logged_metrics.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1


def test_create_local_factory(tmp_path):
    logger = create_experiment_logger("local", run_dir=tmp_path)
    assert isinstance(logger, LocalExperimentLogger)


def test_database_logger_sqlite(tmp_path):
    db_path = tmp_path / "experiments.db"
    logger = DatabaseExperimentLogger(db_path=db_path, experiment_name="exp1")
    logger.log_params({"seed": 42})
    logger.log_metrics({"accuracy": 0.9}, step=0)
    logger.finish()
    assert db_path.exists()
