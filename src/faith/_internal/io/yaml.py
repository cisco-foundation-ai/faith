# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""A custom YAML loader that supports the !from directive for including external YAML files."""

import re
from copy import deepcopy
from functools import cache
from pathlib import Path
from typing import IO, Any

import yaml

from faith._internal.io.benchmarks import benchmarks_root

_INDEXED_PATH_RE = re.compile(r'([^\[]+)((?:\[(?:\d+|(?:\'[^\']+\')|(?:"[^"]+"))\])*)')
_INDEX_RE = re.compile(r'\[(\d+|\'[^\']+\'|"[^"]+")\]')


def _resolve_path(path: str) -> Path:
    if path.startswith("$BENCHMARKS_ROOT/"):
        return benchmarks_root() / path.replace("$BENCHMARKS_ROOT/", "")
    return Path(path)


def _parse_config_path(path: str) -> tuple[Path, list[int | str]]:
    """Parse the config path and return the base path and indices."""
    match = _INDEXED_PATH_RE.fullmatch(path)
    if not match:
        raise yaml.YAMLError(f"Invalid include path: {path}")
    file_path, selection = match.groups()
    indices = []
    if len(selection) > 0:
        indices = [
            int(idx) if idx.isdigit() else idx.strip("'\"")
            for idx in _INDEX_RE.findall(selection)
        ]
    return _resolve_path(file_path), indices


def _deep_merge_config(
    source: dict[Any, Any], overrides: dict[Any, Any]
) -> dict[Any, Any]:
    """Recursively merge two dictionaries, with `overrides` taking precedence."""
    for key, value in overrides.items():
        if isinstance(value, dict) and key in source and isinstance(source[key], dict):
            source[key] = _deep_merge_config(source[key], value)
        else:
            source[key] = value
    return source


def _import_config(base_path: Path, config_path: str) -> dict:
    """Import a YAML configuration file, handling !from directives recursively."""
    file_path, indices = _parse_config_path(config_path)
    if not file_path.is_absolute():
        file_path = base_path / file_path
    # Use _YamlIncludeLoader to load the included YAML file to support !from
    # directives within it.
    imported_config = deepcopy(read_extended_yaml_file(file_path))
    for index in indices:
        if isinstance(imported_config, list) and isinstance(index, int):
            imported_config = imported_config[index]
        elif isinstance(imported_config, dict) and isinstance(index, str):
            imported_config = imported_config.get(index, None)
        else:
            raise yaml.YAMLError(
                f"Invalid index {index} for included config: {str(file_path)}"
            )
    return imported_config


def _include_handler(loader: "_YamlIncludeLoader", node: yaml.Node) -> dict:
    """YAML !from handler to include and parse external YAML files.

    This function handles the !from directive in YAML files, allowing for
    the inclusion of external YAML configurations. It supports both scalar
    and mapping nodes. If the node is a scalar, it treats it as a file path
    and imports the YAML file from that path. If the node is a mapping, it
    merges the included configuration with the current node's mapping acting
    as overrides.
    """
    if isinstance(node, yaml.ScalarNode):
        config_path = loader.construct_scalar(node)
        return _import_config(loader.base_path, config_path)
    if isinstance(node, yaml.MappingNode):
        merged_map = {}
        overrides = loader.construct_mapping(node, deep=True)
        for key, value in overrides.items():
            imported_config = _import_config(loader.base_path, str(key))
            merged_map.update(_deep_merge_config(imported_config, value))
        return merged_map

    raise yaml.YAMLError(f"Unsupported node type for !from: {type(node)}")


def _string_from_handler(loader: "_YamlIncludeLoader", node: yaml.Node) -> str:
    """YAML !string_from handler to include external files as raw strings."""
    if isinstance(node, yaml.ScalarNode):
        file_path = _resolve_path(loader.construct_scalar(node))
        if not file_path.is_absolute():
            file_path = loader.base_path / file_path
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    raise yaml.YAMLError(f"Unsupported node type for !string_from: {type(node)}")


class _YamlIncludeLoader(yaml.SafeLoader):
    """Custom YAML loader that supports the !from directive with overrides."""

    def __init__(self, stream: IO[str]):
        """Initialize the custom YAML loader."""
        super().__init__(stream)
        self._base_path = (
            Path(stream.name).parent if hasattr(stream, "name") else Path.cwd()
        )

    @property
    def base_path(self) -> Path:
        """Get the base path for file includes."""
        return self._base_path


# Register custom YAML tags.
_YamlIncludeLoader.add_constructor("!from", _include_handler)
_YamlIncludeLoader.add_constructor("!string_from", _string_from_handler)


@cache
def read_extended_yaml_file(file_path: Path) -> Any:
    """Read a YAML file with support for !from directives."""
    if not file_path.exists():
        raise FileNotFoundError(f"YAML file '{file_path}' does not exist.")
    with open(file_path, "r", encoding="utf-8") as file:
        # Use the custom loader to handle !from directives
        return yaml.load(stream=file, Loader=_YamlIncludeLoader)
