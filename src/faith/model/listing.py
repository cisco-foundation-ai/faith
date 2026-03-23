# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Functions that list model configs from the packaged resources."""

from collections.abc import Sequence
from pathlib import Path

from faith._internal.io.resources import models_root


def _get_model_paths(root_dir: Path) -> Sequence[str]:
    """Get a list of all sub-paths of `root_dir` that are valid model config paths."""
    return sorted(
        str(f.relative_to(root_dir).with_suffix("")) for f in root_dir.glob("*/*.yaml")
    )


def model_choices() -> Sequence[str]:
    """Get a list of available packaged model config names."""
    return list(_get_model_paths(models_root()))


def choice_to_model(choice: str) -> Path:
    """Convert a model choice to a model config path.

    If ``choice`` is a known packaged model name (e.g. ``"instruct/phi-4"``),
    returns its path under ``models_root()``.  Otherwise treats it as a local
    file path.
    """
    root = models_root()
    if choice in set(_get_model_paths(root)):
        return root / f"{choice}.yaml"
    return Path(choice)


def find_models(root_dir: Path) -> Sequence[Path]:
    """Find all valid model configs in the `root_dir` directory."""
    return [root_dir / f"{sub_path}.yaml" for sub_path in _get_model_paths(root_dir)]
