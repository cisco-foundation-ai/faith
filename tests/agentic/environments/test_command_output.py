# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""The standalone runtime must drain output without retaining it all."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

_SOURCE = (
    Path(__file__).parents[3]
    / "src/faith/__benchmarks__/vulnerability-localization/runtime/command_output.py"
)
_SPEC = importlib.util.spec_from_file_location("faith_runtime_command_output", _SOURCE)
assert _SPEC is not None and _SPEC.loader is not None
capture = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(capture)


@pytest.mark.parametrize(
    "data",
    [b"", b"ascii\n", "é世界🐍".encode(), b"\xff\xc3x\xe2\x82", b"a\r\nb\rc\n\r"],
)
@pytest.mark.parametrize("chunk_size", [1, 2, 3, 65536])
def test_incremental_decoding_matches_subprocess(data, chunk_size, monkeypatch):
    """Read boundaries must not corrupt UTF-8 or change universal newlines."""
    monkeypatch.setattr(capture, "READ_CHUNK_BYTES", chunk_size)
    argv = [
        sys.executable,
        "-c",
        f"import os; os.write(1, {data!r}); os.write(2, {data!r})",
    ]
    expected = subprocess.run(
        argv,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    result = capture.run_captured(argv)
    assert (result.stdout, result.stderr, result.returncode) == (
        expected.stdout,
        expected.stderr,
        expected.returncode,
    )


@pytest.mark.parametrize("size", [127, 128, 129])
def test_cap_counts_characters_not_utf8_bytes(size, monkeypatch):
    monkeypatch.setattr(capture, "MAX_OUTPUT_CHARS", 128)
    result = capture.run_captured(
        [sys.executable, "-c", f"import os; os.write(1, ('🐍' * {size}).encode())"]
    )
    assert result.returncode == 0
    if size <= 128:
        assert result.stdout == "🐍" * size
    else:
        notice, prefix = result.stdout.split("\n", 1)
        assert "129 total chars (stdout=129, stderr=0)" in notice
        assert prefix == "🐍" * 128
    assert result.stderr == ""


@pytest.mark.parametrize("stdout_size,stderr_size", [(64, 200), (200, 64), (0, 200)])
def test_retains_combined_stdout_then_stderr_prefix(
    stdout_size, stderr_size, monkeypatch
):
    monkeypatch.setattr(capture, "MAX_OUTPUT_CHARS", 128)
    result = capture.run_captured(
        [
            sys.executable,
            "-c",
            f"import os; os.write(2, b'e'*{stderr_size}); os.write(1, b'o'*{stdout_size}); raise SystemExit(23)",
        ]
    )
    notice, output = (result.stdout + result.stderr).split("\n", 1)
    assert f"{stdout_size + stderr_size} total chars" in notice
    assert output == ("o" * stdout_size + "e" * stderr_size)[:128]
    assert result.returncode == 23


def test_drains_large_simultaneous_streams_without_replay():
    """Each pipe exceeds capacity many times; both writers must finish normally."""
    size = 32 * 1024 * 1024
    result = capture.run_captured(
        [
            sys.executable,
            "-c",
            "import os, threading; "
            "write = lambda fd: [os.write(fd, b'x'*65536) for _ in range(512)]; "
            "worker = threading.Thread(target=write, args=(2,)); "
            "worker.start(); write(1); worker.join(); raise SystemExit(17)",
        ]
    )
    notice, prefix = result.stdout.split("\n", 1)
    assert f"{size * 2} total chars (stdout={size}, stderr={size})" in notice
    assert prefix == "x" * capture.MAX_OUTPUT_CHARS
    assert result.stderr == ""
    assert result.returncode == 17


def test_decoder_counts_discarded_unicode_and_newlines(monkeypatch):
    monkeypatch.setattr(capture, "MAX_OUTPUT_CHARS", 3)
    monkeypatch.setattr(capture, "READ_CHUNK_BYTES", 1)
    result = capture.run_captured(
        [
            sys.executable,
            "-c",
            "import os; os.write(1, b'abc\\r\\n\\xff\\xf0\\x9f\\x90\\x8d\\r')",
        ]
    )
    notice, prefix = result.stdout.split("\n", 1)
    assert "7 total chars" in notice
    assert prefix == "abc"
