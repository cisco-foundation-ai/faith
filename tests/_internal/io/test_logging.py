# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

import tempfile
from pathlib import Path
from typing import TypeAlias

from faith._internal.io.json import read_json_file
from faith._internal.io.logging import LogCollector, LoggingTransform

_TestLog: TypeAlias = dict[str, int]


def test_log_collector() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpfile = Path(tmpdir) / "log.json"
        with LogCollector[_TestLog](tmpfile) as logger:
            logger.log({"foo": 1})
            logger.log({"bar": 2})

        assert tmpfile.exists()
        logs = read_json_file(tmpfile)
        assert logs == [{"foo": 1}, {"bar": 2}]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpfile = Path(tmpdir) / "no_log.json"
        with LogCollector[_TestLog](tmpfile) as logger:
            pass
        assert not tmpfile.exists()


def test_log_collector_on_exception() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpfile = Path(tmpdir) / "log_on_exception.json"
        try:
            with LogCollector[_TestLog](tmpfile, log_on_exception=True) as logger:
                logger.log({"foo": 1})
                raise ValueError("Test exception")
        except ValueError:
            pass

        assert tmpfile.exists()
        logs = read_json_file(tmpfile)
        assert logs == [{"foo": 1}]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpfile = Path(tmpdir) / "no_log_on_exception.json"
        try:
            with LogCollector[_TestLog](tmpfile, log_on_exception=False) as logger:
                logger.log({"foo": 1})
                raise ValueError("Test exception")
        except ValueError:
            pass

        assert not tmpfile.exists()


def test_logging_transform() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpfile = Path(tmpdir) / "log.json"
        assert list(
            [{"foo": 1}, {"bar": 3}] >> LoggingTransform[_TestLog](tmpfile)
        ) == [{"foo": 1}, {"bar": 3}]
        assert tmpfile.exists()
        assert read_json_file(tmpfile) == [{"foo": 1}, {"bar": 3}]

        tmpfile2 = Path(tmpdir) / "log2.json"
        assert not list([] >> LoggingTransform[_TestLog](tmpfile2))
        assert not tmpfile2.exists()
