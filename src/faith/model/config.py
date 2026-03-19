# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Loader for model configuration files."""

from pathlib import Path

from faith._internal.io.yaml import read_extended_yaml_file
from faith._types.model.spec import ModelSpec


def load_model_config(config_path: Path) -> ModelSpec:
    """Load a ModelSpec from a YAML configuration file."""
    model_spec_dict = read_extended_yaml_file(config_path).get("model") or {}
    assert isinstance(
        model_spec_dict, dict
    ), f"Model config '{config_path}' must be a YAML mapping."

    return ModelSpec.from_dict(model_spec_dict)
