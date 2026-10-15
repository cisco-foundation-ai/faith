# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for cached, content-verified fetching of GitHub source archives."""

import hashlib
import http.client
import io
import logging
import urllib.error
import zipfile
from concurrent.futures import ThreadPoolExecutor
from itertools import permutations
from pathlib import Path
from threading import Barrier
from unittest.mock import MagicMock, patch

import pytest

from faith.agentic.loaders._fetch import _content_md5 as _digest
from faith.agentic.loaders._fetch import cache_root, fetch_file, fetch_repo

_SHA = "a" * 40
_PREFIX = f"widget-{_SHA}/"
_GIT_VERSION = 'gitVersion = "v0.0.0-master+{sha}"'

# A digest pinned as a literal, anchoring the wire format the manifest records:
# md5 over each path's bytes then its content, in lexicographic path order.
# Declared out of path order, so a digest that skipped the sort would differ.
_PINNED_TREE = {"b.txt": "beta", "a.txt": "alpha"}
_PINNED_MD5 = "5750c54849c74e8cde3756c556f2cc72"


def _archive(files: dict[str, str], symlinks: dict[str, str] | None = None) -> bytes:
    """Build a GitHub-shaped archive: every path under a `{repo}-{sha}/` wrapper."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as out:
        out.writestr(_PREFIX, "")
        for path, content in files.items():
            out.writestr(f"{_PREFIX}{path}", content)
        for path, target in (symlinks or {}).items():
            info = zipfile.ZipInfo(f"{_PREFIX}{path}")
            info.external_attr = 0o120777 << 16
            out.writestr(info, target)
    return buffer.getvalue()


def _content_md5(entries: dict[str, str]) -> str:
    """The digest the manifest would record for an extracted, resolved tree."""
    return _digest({path: content.encode() for path, content in entries.items()})


def _reader(payload: bytes, read_side_effect: BaseException | None = None) -> object:
    """A stand-in for urlopen's context-managed response over `payload`."""
    read = MagicMock(
        side_effect=read_side_effect if read_side_effect is not None else None,
        return_value=payload,
    )
    return MagicMock(
        __enter__=MagicMock(return_value=MagicMock(read=read)),
        __exit__=MagicMock(return_value=False),
    )


def _fetch(cache: Path, payload: bytes, md5: str, prefix: str = "repo") -> Path:
    """Fetch against a stubbed download, returning the cached zip."""
    with patch("faith.agentic.loaders._fetch._download", return_value=payload):
        return fetch_repo(cache, "acme/widget", _SHA, md5, prefix)


def _members(zip_path: Path) -> dict[str, bytes]:
    """The (path, content) pairs a cached zip holds."""
    with zipfile.ZipFile(zip_path) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def test_the_github_wrapper_directory_is_stripped(tmp_path: Path) -> None:
    """Ground truth names paths from the repo root, so the wrapper must go.

    An unstripped tree would score every task 0.0 with nothing visibly broken.
    """
    files = {"pkg/read.go": "package main", "README.md": "# widget"}
    cached = _fetch(tmp_path, _archive(files), _content_md5(files))
    assert _members(cached) == {
        "repo/README.md": b"# widget",
        "repo/pkg/read.go": b"package main",
    }


@pytest.mark.parametrize(
    ("prefix", "expected"),
    [
        ("repo", "repo/pkg/read.go"),
        ("src/repo", "src/repo/pkg/read.go"),
        ("", "pkg/read.go"),
    ],
    ids=["one_level", "nested", "at_the_workspace_root"],
)
def test_the_caller_names_the_directory_the_tree_unpacks_into(
    tmp_path: Path, prefix: str, expected: str
) -> None:
    """The prefix comes from the caller, which is what sends it where the agent is.

    A tree that unpacked somewhere else would leave the prompt naming an empty
    directory, so the benchmark's working subdir is the only thing that sets it.
    """
    files = {"pkg/read.go": "package main"}
    cached = _fetch(tmp_path, _archive(files), _content_md5(files), prefix)
    assert set(_members(cached)) == {expected}


def test_a_symlink_is_replaced_by_its_targets_content(tmp_path: Path) -> None:
    """The manifest hashes resolved trees, so a link must carry real bytes."""
    files = {"pkg/real.go": "package main"}
    payload = _archive(files, symlinks={"pkg/link.go": "real.go"})
    expected = _content_md5(files | {"pkg/link.go": "package main"})
    cached = _fetch(tmp_path, payload, expected)
    assert _members(cached)["repo/pkg/link.go"] == b"package main"


def test_a_symlink_pointing_outside_the_tree_is_dropped(tmp_path: Path) -> None:
    """An unresolvable target contributes nothing, never the host path it names."""
    files = {"pkg/real.go": "package main"}
    payload = _archive(files, symlinks={"link": "../../etc/passwd"})
    cached = _fetch(tmp_path, payload, _content_md5(files))
    assert set(_members(cached)) == {"repo/pkg/real.go"}


def test_a_symlink_through_a_directory_symlink_resolves(tmp_path: Path) -> None:
    """Multi-hop targets are real in the corpus, and stop at the final file."""
    files = {"real/provider.go": "package main"}
    symlinks = {"hidden": "real", "provider.go": "hidden/provider.go"}
    payload = _archive(files, symlinks=symlinks)
    expected = files | {"provider.go": "package main"}
    cached = _fetch(tmp_path, payload, _content_md5(expected))
    assert _members(cached) == {
        f"repo/{path}": content.encode() for path, content in expected.items()
    }


def test_a_symlink_to_a_directory_is_omitted(tmp_path: Path) -> None:
    """A directory link resolves to no file, and the manifest excludes it too."""
    files = {"real/provider.go": "package main"}
    payload = _archive(files, symlinks={"hidden": "real"})
    cached = _fetch(tmp_path, payload, _content_md5(files))
    assert set(_members(cached)) == {"repo/real/provider.go"}


@pytest.mark.parametrize(
    "path, archived, extracted",
    [
        ("pkg/version/base.go", "0123456789abc", "0123456789a"),
        ("pkg/version/base.go", "0" * 40, "0" * 40),
        ("pkg/version/other.go", "0123456789abc", "0123456789abc"),
    ],
    ids=[
        "an-abbreviated-hash-is-trimmed-by-two",
        "a-full-sha1-is-left-alone",
        "another-path-is-left-alone",
    ],
)
def test_githubs_export_subst_expansion_is_normalized(
    tmp_path: Path, path: str, archived: str, extracted: str
) -> None:
    """GitHub expands `$Format:%h$` two hex characters longer than git-clone."""
    payload = _archive({path: _GIT_VERSION.format(sha=archived)})
    expected = {path: _GIT_VERSION.format(sha=extracted)}
    cached = _fetch(tmp_path, payload, _content_md5(expected))
    assert _members(cached) == {f"repo/{path}": expected[path].encode()}


def test_a_symlink_carries_its_targets_normalized_bytes(tmp_path: Path) -> None:
    """Normalization precedes inlining, so a link cannot smuggle raw bytes in."""
    archived = {"version/base.go": _GIT_VERSION.format(sha="0123456789abc")}
    payload = _archive(archived, symlinks={"version/link.go": "base.go"})
    trimmed = _GIT_VERSION.format(sha="0123456789a")
    expected = {"version/base.go": trimmed, "version/link.go": trimmed}
    cached = _fetch(tmp_path, payload, _content_md5(expected))
    assert _members(cached) == {
        f"repo/{path}": content.encode() for path, content in expected.items()
    }


def test_the_content_digest_matches_the_manifests_wire_format() -> None:
    """A pinned digest anchors path order and encoding to the manifest's own."""
    entries = {path: content.encode() for path, content in _PINNED_TREE.items()}
    assert _digest(entries) == _PINNED_MD5


def test_a_matching_content_hash_is_silent(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A verified fetch says nothing, so a warning always means something."""
    files = {"pkg/read.go": "package main"}
    with caplog.at_level(logging.WARNING):
        _fetch(tmp_path, _archive(files), _content_md5(files))
    assert caplog.text == ""


def test_a_content_hash_mismatch_raises_and_caches_nothing(tmp_path: Path) -> None:
    """Unverified content must not reach the cache, where nothing re-checks it.

    Raising fails the one task whose setup fetches it, not the run: the caller
    records a setup failure, which the aggregator excludes from the metrics
    rather than scoring as a wrong answer.
    """
    with pytest.raises(ValueError) as excinfo:
        _fetch(tmp_path, _archive({"a.go": "x"}), "0" * 32)
    assert str(excinfo.value) == (
        f"Content hash mismatch for acme/widget@{_SHA}: "
        f"got 870dd390db14292e7c9bd38b00e9e4e5, expected {'0' * 32}."
    )
    assert not list(tmp_path.glob("*"))


def test_a_second_fetch_reuses_the_cached_zip(tmp_path: Path) -> None:
    """A repeated commit is not re-downloaded, across phases or across runs."""
    files = {"pkg/read.go": "package main"}
    md5 = _content_md5(files)
    first = _fetch(tmp_path, _archive(files), md5)
    with patch("faith.agentic.loaders._fetch._download") as download:
        second = fetch_repo(tmp_path, "acme/widget", _SHA, md5, "repo")
    download.assert_not_called()
    assert second == first


@pytest.mark.parametrize(
    "first_prefix,second_prefix", permutations(["repo", "src/repo", ""], 2)
)
def test_a_warm_cache_keeps_each_prefix_separate(
    tmp_path: Path, first_prefix: str, second_prefix: str
) -> None:
    """The same commit can be unpacked into multiple layouts in one cache."""
    files = {"read.go": "package main"}
    payload, md5 = _archive(files), _content_md5(files)
    first = _fetch(tmp_path, payload, md5, first_prefix)
    first_bytes = first.read_bytes()
    second = _fetch(tmp_path, payload, md5, second_prefix)
    assert first != second
    assert first.read_bytes() == first_bytes
    for prefix, path in ((first_prefix, first), (second_prefix, second)):
        expected = f"{prefix}/read.go" if prefix else "read.go"
        assert _members(path) == {expected: b"package main"}
        with patch("faith.agentic.loaders._fetch._download") as download:
            assert fetch_repo(tmp_path, "acme/widget", _SHA, md5, prefix) == path
        download.assert_not_called()
    assert not list(tmp_path.glob("*.partial"))


@pytest.mark.parametrize("prefix", ["repo", "src/repo", ""])
def test_legacy_sha_only_cache_is_not_trusted_or_modified(
    tmp_path: Path, prefix: str
) -> None:
    """A legacy cache hit might have been produced for any working directory."""
    legacy = tmp_path / f"{_SHA}.zip"
    with zipfile.ZipFile(legacy, "w") as archive:
        archive.writestr("old-layout/read.go", "package main")
    original = legacy.read_bytes()
    files = {"read.go": "package main"}
    with patch(
        "faith.agentic.loaders._fetch._download", return_value=_archive(files)
    ) as download:
        result = fetch_repo(tmp_path, "acme/widget", _SHA, _content_md5(files), prefix)
    download.assert_called_once()
    assert result != legacy
    assert _members(result) == {
        (f"{prefix}/read.go" if prefix else "read.go"): b"package main"
    }
    assert legacy.read_bytes() == original


def test_concurrent_prefixes_have_separate_partial_files(tmp_path: Path) -> None:
    """Two layouts built in the same process must not share a staging filename."""
    payload = _archive({"read.go": "package main"})
    md5 = _content_md5({"read.go": "package main"})
    ready = Barrier(2, timeout=10)

    def download(_url: str) -> bytes:
        ready.wait()
        return payload

    def fetch(prefix: str) -> Path:
        return fetch_repo(tmp_path, "acme/widget", _SHA, md5, prefix)

    with patch("faith.agentic.loaders._fetch._download", side_effect=download):
        with ThreadPoolExecutor(max_workers=2) as workers:
            first, second = workers.map(fetch, ("repo", "src/repo"))
    assert first != second
    assert _members(first) == {"repo/read.go": b"package main"}
    assert _members(second) == {"src/repo/read.go": b"package main"}
    assert not list(tmp_path.glob("*.partial"))


def test_a_failed_download_leaves_no_cache_entry(tmp_path: Path) -> None:
    """A partial fetch must not be mistaken for a complete one on the next run."""
    error = urllib.error.URLError("connection reset")
    with patch("faith.agentic.loaders._fetch._download", side_effect=error):
        with pytest.raises(urllib.error.URLError):
            fetch_repo(tmp_path, "acme/widget", _SHA, "0" * 32, "repo")
    assert not list(tmp_path.glob("*"))


def test_a_transient_failure_is_retried(tmp_path: Path) -> None:
    """A reset mid-corpus self-heals rather than erroring one task out."""
    responses = [urllib.error.URLError("reset"), _reader(_archive({"a.go": "x"}))]
    with patch("faith.agentic.loaders._fetch.time.sleep"):
        with patch("urllib.request.urlopen", side_effect=responses) as urlopen:
            fetch_repo(
                tmp_path, "acme/widget", _SHA, _content_md5({"a.go": "x"}), "repo"
            )
    assert urlopen.call_count == 2


@pytest.mark.parametrize(
    "error",
    [http.client.IncompleteRead(b"partial"), ConnectionResetError("reset")],
    ids=["incomplete_read", "connection_reset"],
)
def test_a_mid_download_failure_is_retried(
    tmp_path: Path, error: BaseException
) -> None:
    """Failures while reading a response are transient network failures too."""
    payload = _archive({"a.go": "x"})
    responses = [_reader(b"", read_side_effect=error), _reader(payload)]
    with patch("faith.agentic.loaders._fetch.time.sleep"):
        with patch("urllib.request.urlopen", side_effect=responses) as urlopen:
            fetch_repo(
                tmp_path, "acme/widget", _SHA, _content_md5({"a.go": "x"}), "repo"
            )
    assert urlopen.call_count == 2


def test_retries_are_bounded(tmp_path: Path) -> None:
    """A repo that is down stays a failure instead of retrying indefinitely."""
    with patch("faith.agentic.loaders._fetch.time.sleep"):
        with patch(
            "urllib.request.urlopen", side_effect=urllib.error.URLError("down")
        ) as urlopen:
            with pytest.raises(urllib.error.URLError):
                fetch_repo(tmp_path, "acme/widget", _SHA, "0" * 32, "repo")
    assert urlopen.call_count == 3


def test_a_missing_archive_is_not_retried(tmp_path: Path) -> None:
    """A 404 is permanent, so retrying it only delays the same failure."""
    missing = urllib.error.HTTPError("u", 404, "Not Found", {}, None)  # type: ignore[arg-type]
    with patch("faith.agentic.loaders._fetch.time.sleep"):
        with patch("urllib.request.urlopen", side_effect=missing) as urlopen:
            with pytest.raises(urllib.error.HTTPError):
                fetch_repo(tmp_path, "acme/widget", _SHA, "0" * 32, "repo")
    assert urlopen.call_count == 1


def test_a_single_file_is_fetched_and_cached(tmp_path: Path) -> None:
    """The manifest is fetched once, then read from the cache."""
    payload = b"alpha_id\naaa111\n"
    md5 = hashlib.md5(payload, usedforsecurity=False).hexdigest()
    with patch("faith.agentic.loaders._fetch._download", return_value=payload) as first:
        path = fetch_file(tmp_path, "acme/widget", _SHA, "data/m.csv", md5)
    assert (path.read_bytes(), first.call_count) == (payload, 1)
    with patch("faith.agentic.loaders._fetch._download") as download:
        again = fetch_file(tmp_path, "acme/widget", _SHA, "data/m.csv", md5)
    download.assert_not_called()
    assert again == path


def test_a_single_files_hash_mismatch_raises(tmp_path: Path) -> None:
    """Unlike an archive, a bad manifest has no partial value: fail, don't warn."""
    with patch("faith.agentic.loaders._fetch._download", return_value=b"tampered"):
        with pytest.raises(ValueError, match="Content hash mismatch"):
            fetch_file(tmp_path, "acme/widget", _SHA, "data/m.csv", "0" * 32)
    assert not list(tmp_path.iterdir())


def test_the_cache_root_follows_the_configured_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A deployment can place the cache on a volume with room for it."""
    monkeypatch.setenv("FAITH_CACHE_DIR", "/data/cache")
    assert cache_root("vulnerability_localization") == Path(
        "/data/cache/faith/vulnerability_localization"
    )


@pytest.mark.parametrize(
    "environment, expected",
    [
        (
            {"FAITH_CACHE_DIR": "/a", "XDG_CACHE_HOME": "/b"},
            "/a/faith/vulnerability_localization",
        ),
        ({"XDG_CACHE_HOME": "/b"}, "/b/faith/vulnerability_localization"),
        ({}, None),
    ],
    ids=[
        "faith-cache-dir-wins",
        "xdg-cache-home-is-the-fallback",
        "neither-set-falls-back-to-home",
    ],
)
def test_the_cache_root_precedence(
    monkeypatch: pytest.MonkeyPatch, environment: dict[str, str], expected: str | None
) -> None:
    """FAITH's own variable takes precedence, then XDG's, then the home default."""
    monkeypatch.delenv("FAITH_CACHE_DIR", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    assert cache_root("vulnerability_localization") == (
        Path(expected)
        if expected
        else Path.home() / ".cache" / "faith" / "vulnerability_localization"
    )
