# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Cached, content-verified fetching of GitHub source archives."""

import hashlib
import http.client
import io
import logging
import os
import posixpath
import re
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)

_ARCHIVE_URL = "https://github.com/{repo}/archive/{sha}.zip"
_RAW_URL = "https://raw.githubusercontent.com/{repo}/{sha}/{path}"
_MAX_ATTEMPTS = 3
_TIMEOUT_SECONDS = 300
_USER_AGENT = "faith-benchmark-downloader/1.0"
_SYMLINK_MODE = 0o120000
_MAX_SYMLINK_DEPTH = 10
_EXPORT_SUBST_RE = re.compile(rb'("v0\.0\.0-master\+)([0-9a-f]{9,})(")')


def cache_root(name: str) -> Path:
    """The cache directory for a benchmark's fetched payloads."""
    base = os.environ.get("FAITH_CACHE_DIR") or os.environ.get("XDG_CACHE_HOME")
    root = Path(base) if base else Path.home() / ".cache"
    return root / "faith" / name


def repo_cache_path(cache: Path, sha: str, prefix: str) -> Path:
    """Name a snapshot by commit and archive layout, not commit alone."""
    layout = hashlib.sha256(prefix.encode("utf-8")).hexdigest()
    # Legacy SHA-only zips may contain any prefix; do not reuse them unchecked.
    return cache / f"{sha}-{layout}.zip"


def fetch_repo(
    cache: Path, repo: str, sha: str, expected_md5: str, prefix: str
) -> Path:
    """Fetch a repository snapshot as a flat zip, cached and content-verified.

    GitHub's ``{repo}-{sha}/`` wrapper is removed and `prefix` put in its place,
    so unzipping the result in a workspace lands the source at
    ``<workspace>/<prefix>/``. The caller passes the subdirectory the agent works
    from, which is what makes the two the same directory.

    Raises:
        ValueError: If the fetched tree does not match `expected_md5`. Ground
            truth was recorded against the bytes the digest names, so scoring
            against different ones measures nothing; nothing is cached, and the
            caller fails this task rather than the run.
    """
    dest = repo_cache_path(cache, sha, prefix)
    if dest.exists():
        return dest
    payload = _download(_ARCHIVE_URL.format(repo=repo, sha=sha))
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        entries = _read_entries(archive)
    digest = _content_md5(entries)
    if digest != expected_md5:
        raise ValueError(
            f"Content hash mismatch for {repo}@{sha}: "
            f"got {digest}, expected {expected_md5}."
        )
    cache.mkdir(parents=True, exist_ok=True)
    # Pid-suffixed so concurrent workers on the same repo cannot clobber a
    # half-written file; `replace` then publishes it atomically.
    partial = cache / f"{dest.name}.{os.getpid()}.partial"
    with zipfile.ZipFile(partial, "w", zipfile.ZIP_DEFLATED) as out:
        for path, content in sorted(entries.items()):
            out.writestr(posixpath.join(prefix, path), content)
    partial.replace(dest)
    return dest


def fetch_file(cache: Path, repo: str, sha: str, path: str, expected_md5: str) -> Path:
    """Fetch one file from a pinned commit, cached and content-verified.

    Raises:
        ValueError: If the fetched bytes do not match `expected_md5`.
    """
    dest = cache / f"{sha}-{posixpath.basename(path)}"
    if dest.exists():
        return dest
    payload = _download(_RAW_URL.format(repo=repo, sha=sha, path=path))
    digest = hashlib.md5(payload, usedforsecurity=False).hexdigest()
    if digest != expected_md5:
        raise ValueError(
            f"Content hash mismatch for {path}@{sha}: "
            f"got {digest}, expected {expected_md5}."
        )
    cache.mkdir(parents=True, exist_ok=True)
    partial = cache / f"{dest.name}.{os.getpid()}.partial"
    partial.write_bytes(payload)
    partial.replace(dest)
    return dest


def _download(url: str) -> bytes:
    """Download a URL, retrying transient failures with exponential backoff.

    A 404 is permanent (deleted repo, garbage-collected commit), so it is raised
    without spending the remaining attempts on it.
    """
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    for attempt in range(_MAX_ATTEMPTS):
        try:
            with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as response:
                return bytes(response.read())
        except urllib.error.HTTPError as e:
            if e.code == 404 or attempt == _MAX_ATTEMPTS - 1:
                raise
        except (urllib.error.URLError, http.client.HTTPException, OSError):
            if attempt == _MAX_ATTEMPTS - 1:
                raise
        logger.warning("Retrying %s (attempt %d).", url, attempt + 2)
        time.sleep(2**attempt)
    raise RuntimeError(f"Unreachable: exhausted retries for {url}")


def _archive_prefix(archive: zipfile.ZipFile) -> str:
    """The `{repo}-{sha}/` directory GitHub wraps an archive in."""
    first = archive.namelist()[0]
    return first if first.endswith("/") else first.split("/")[0] + "/"


def _normalize_export_subst(path: str, content: bytes) -> bytes:
    """Trim a `$Format:%h$` expansion to the length git-clone produces.

    GitHub's archive service abbreviates two hex characters longer than
    git-clone does, except for a full 40-character SHA1, where the two agree.
    """
    if not path.endswith("version/base.go"):
        return content

    def trim(match: re.Match[bytes]) -> bytes:
        prefix, sha, suffix = match.groups()
        return match[0] if len(sha) == 40 else prefix + sha[:-2] + suffix

    return _EXPORT_SUBST_RE.sub(trim, content)


def _resolve_symlink(path: str, symlinks: dict[str, str]) -> str:
    """Follow a path through symlinks naming it or any of its parent directories."""
    for _ in range(_MAX_SYMLINK_DEPTH):
        if path in symlinks:
            path = posixpath.normpath(
                posixpath.join(posixpath.dirname(path), symlinks[path])
            )
            continue
        parts = path.split("/")
        for index in range(1, len(parts)):
            prefix = "/".join(parts[:index])
            if prefix in symlinks:
                target = symlinks[prefix].rstrip("/")
                parent = posixpath.normpath(
                    posixpath.join(posixpath.dirname(prefix), target)
                )
                path = posixpath.join(parent, "/".join(parts[index:]))
                break
        else:
            break
    return path


def _read_entries(archive: zipfile.ZipFile) -> dict[str, bytes]:
    """Read an archive's files, unwrapped, normalized, and with symlinks inlined.

    A symlink carries its target's path as content, so it is replaced by the
    target's bytes; the manifest's content hash is computed over resolved trees.
    Directory symlinks resolve to no file and are omitted.
    """
    prefix = _archive_prefix(archive)
    entries: dict[str, bytes] = {}
    symlinks: dict[str, str] = {}
    for info in archive.infolist():
        if info.is_dir():
            continue
        path = info.filename.removeprefix(prefix)
        content = archive.read(info.filename)
        if (info.external_attr >> 16) & 0o170000 == _SYMLINK_MODE:
            symlinks[path] = content.decode("utf-8", "replace").strip()
        else:
            entries[path] = _normalize_export_subst(path, content)
    for path, target in symlinks.items():
        direct = posixpath.normpath(posixpath.join(posixpath.dirname(path), target))
        resolved = _resolve_symlink(direct, symlinks)
        if resolved in entries:
            entries[path] = entries[resolved]
        elif direct in entries:
            entries[path] = entries[direct]
    return entries


def _content_md5(entries: dict[str, bytes]) -> str:
    """Hash a file tree by path and content, independent of zip metadata.

    MD5 to match the digests the benchmark manifest records; an integrity check
    against a published manifest, not a security boundary.
    """
    digest = hashlib.md5(usedforsecurity=False)
    for path, content in sorted(entries.items()):
        digest.update(path.encode("utf-8"))
        digest.update(content)
    return digest.hexdigest()
