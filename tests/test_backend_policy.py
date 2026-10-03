"""The backend is always explicit: the model or a loud failure, never a silent heuristic."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest
from fastapi.testclient import TestClient

from groundcheck import GroundCheck, ModelUnavailableError, api, cli
from groundcheck.config import DEFAULT_MODEL_ID, DEFAULT_MODEL_REVISION, DetectorSettings, get_detector_settings


@pytest.fixture()
def clean_env(monkeypatch):
    for var in ("GROUNDCHECK_BACKEND", "GROUNDCHECK_MODEL_PATH", "GROUNDCHECK_MODEL_REVISION"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture()
def no_torch(monkeypatch):
    # Simulates an install without the [model] extra: `import torch` raises ImportError.
    monkeypatch.setitem(sys.modules, "torch", None)


def test_default_settings_select_the_pinned_model(clean_env):
    s = get_detector_settings()
    assert s.backend == "model"
    assert s.model_path == DEFAULT_MODEL_ID
    assert s.revision == DEFAULT_MODEL_REVISION


def test_empty_model_path_env_still_means_the_published_model(clean_env, monkeypatch):
    # .env files often carry `GROUNDCHECK_MODEL_PATH=`; that used to switch to the heuristic.
    monkeypatch.setenv("GROUNDCHECK_MODEL_PATH", "")
    assert get_detector_settings().model_path == DEFAULT_MODEL_ID


def test_custom_model_path_is_not_given_the_published_revision(clean_env, monkeypatch):
    monkeypatch.setenv("GROUNDCHECK_MODEL_PATH", "./my-model")
    assert get_detector_settings().revision is None


def test_overriding_model_path_drops_the_published_revision(clean_env):
    assert GroundCheck(model_path="./my-model", backend="heuristic").settings.revision is None
    assert GroundCheck(model_path="./my-model", revision="abc", backend="heuristic").settings.revision == "abc"
    assert GroundCheck(model_path=DEFAULT_MODEL_ID, backend="heuristic").settings.revision == DEFAULT_MODEL_REVISION


def test_settings_built_directly_pin_only_the_published_model():
    assert DetectorSettings().revision == DEFAULT_MODEL_REVISION
    assert DetectorSettings(model_path="./my-model").revision is None
    assert DetectorSettings(model_path="./my-model", revision="abc").revision == "abc"
    gc = GroundCheck(DetectorSettings(model_path="./my-model", backend="heuristic"))
    assert gc.settings.revision is None


def test_auto_backend_is_rejected():
    with pytest.raises(ValueError, match="removed in 0.2.0"):
        DetectorSettings(backend="auto")


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError, match="Unknown backend"):
        DetectorSettings(backend="llm")


def test_heuristic_only_when_requested(no_torch):
    gc = GroundCheck(backend="heuristic")
    assert gc.backend == "heuristic"
    assert gc.check("a b c", "a b c").backend == "heuristic"
    assert gc.model_info() == {"backend": "heuristic"}


def test_model_unavailable_raises_instead_of_falling_back(clean_env, no_torch):
    with pytest.raises(ModelUnavailableError, match=r"groundcheck-rag\[model\]"):
        GroundCheck()


def test_env_heuristic_is_honoured(clean_env, no_torch, monkeypatch):
    monkeypatch.setenv("GROUNDCHECK_BACKEND", "heuristic")
    assert GroundCheck().backend == "heuristic"


def test_cli_exits_2_when_model_unavailable(clean_env, no_torch, capsys):
    assert cli.main(["--source", "a", "--answer", "a"]) == 2
    assert "groundcheck-rag[model]" in capsys.readouterr().err


def test_cli_heuristic_flag(no_torch, capsys):
    assert cli.main(["--source", "a b", "--answer", "a b", "--backend", "heuristic", "--json"]) == 0
    assert '"backend": "heuristic"' in capsys.readouterr().out


def test_api_refuses_to_start_without_model(clean_env, no_torch):
    with pytest.raises(ModelUnavailableError):
        with TestClient(api.app):
            pass


def test_invalid_model_reference_raises(clean_env):
    pytest.importorskip("transformers")
    with pytest.raises(ModelUnavailableError, match="Could not load model"):
        GroundCheck(model_path="/nonexistent/groundcheck-model", revision=None)


def test_offline_without_cached_weights_raises(tmp_path):
    pytest.importorskip("transformers")
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "HF_HOME": str(tmp_path), "GROUNDCHECK_BACKEND": "model"}
    env.pop("GROUNDCHECK_MODEL_PATH", None)
    proc = subprocess.run(
        [sys.executable, "-m", "groundcheck.cli", "--source", "a", "--answer", "a"],
        env=env, capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 2, proc.stderr
    assert "Could not load model" in proc.stderr
