# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
import yaml

from faith._internal.io.yaml import read_extended_yaml_file


def test_read_extended_yaml_file() -> None:
    """Test the read_extended_yaml_file function with a simple YAML file."""
    cfg = read_extended_yaml_file(
        Path(__file__).parent / "testdata" / "configs" / "d2" / "a.yaml",
    )
    assert cfg == {
        "key1": "value1",
        "key2": {
            "other": 123,
            "lst": [1, 2, 3],
            "d": {
                "a": 456,
                "b": 789,
                "d": {
                    "x": 100,
                    "y": 200,
                },
                "eyes": """Eyes that last I saw in tears
Through division
Here in death's dream kingdom
The golden vision reappears
I see the eyes but not the tears
This is my affliction""",
            },
            "import": {
                "x": "foo",
                "y": "baz",
            },
        },
        "key3": {
            "a": {
                "a": 123,
                "b": 789,
                "c": 456,
                "d": {
                    "x": 101,
                    "y": 200,
                },
                "eyes": """Eyes that last I saw in tears
Through division
Here in death's dream kingdom
The golden vision reappears
I see the eyes but not the tears
This is my affliction""",
            },
            "x": "foo",
            "y": "bar",
            "z": "foo",
        },
        "key4": 1,
    }

    cfg = read_extended_yaml_file(
        Path(__file__).parent / "testdata" / "configs" / "d1" / "b.yaml",
    )
    assert cfg == {
        "other": 123,
        "lst": [1, 2, 3],
        "d": {
            "a": 456,
            "b": 789,
            "d": {
                "x": 100,
                "y": 200,
            },
            "eyes": """Eyes that last I saw in tears
Through division
Here in death's dream kingdom
The golden vision reappears
I see the eyes but not the tears
This is my affliction""",
        },
        "import": {
            "x": "foo",
            "y": "baz",
        },
    }

    with pytest.raises(FileNotFoundError):
        read_extended_yaml_file(
            Path(__file__).parent / "testdata" / "configs" / "d2" / "nonexistent.yaml",
        )
    with pytest.raises(yaml.YAMLError, match="Invalid include path"):
        read_extended_yaml_file(
            Path(__file__).parent / "testdata" / "configs" / "bad" / "bad_import.yaml",
        )
    with pytest.raises(yaml.YAMLError, match="Invalid index"):
        read_extended_yaml_file(
            Path(__file__).parent / "testdata" / "configs" / "bad" / "bad_index.yaml",
        )
