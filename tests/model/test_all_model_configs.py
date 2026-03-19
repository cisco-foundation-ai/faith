# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Ensure every model config YAML under __models__ loads successfully."""

from pathlib import Path

import pytest

from faith._internal.io.resources import models_root
from faith.model.config import load_model_config

# Collect all per-model YAML configs (exclude shared defaults like default_engines).
_MODELS_DIR = models_root()
_MODEL_CONFIGS = sorted(
    p
    for p in _MODELS_DIR.rglob("*.yaml")
    if p.parent != _MODELS_DIR  # skip top-level shared files
)


@pytest.mark.parametrize(
    "config_path",
    _MODEL_CONFIGS,
    ids=[str(p.relative_to(_MODELS_DIR)) for p in _MODEL_CONFIGS],
)
def test_model_config_loads(config_path: Path) -> None:
    """Each model config YAML should produce a valid ModelSpec."""
    spec = load_model_config(config_path)
    assert spec.path, f"ModelSpec loaded from {config_path} has empty path"
    assert spec.engine is not None, f"ModelSpec loaded from {config_path} has no engine"
    assert spec.name, f"ModelSpec loaded from {config_path} has empty name"
