"""Unit tests for guidance errors."""

import pytest

from src.unlabeled_guidance.errors import GuidanceConfigError, MissingLabelError


def test_missing_label_error_message():
    err = MissingLabelError(["r1", "r2", "r3"], hint="label.unlabeled_guidance.enabled")
    assert "3 row(s) missing 'label'" in str(err)
    assert "label.unlabeled_guidance.enabled" in str(err)
    assert err.record_ids == ["r1", "r2", "r3"]


def test_missing_label_error_truncates_ids():
    ids = [f"r{i}" for i in range(10)]
    err = MissingLabelError(ids)
    assert "..." in str(err)


def test_guidance_config_error():
    with pytest.raises(GuidanceConfigError):
        raise GuidanceConfigError("invalid config")
