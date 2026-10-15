# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Content-derived tags for the images FAITH builds itself.

A tag a config author types is a wish, not an identity: it comes out of the same
flat namespace Docker Hub occupies, so nothing makes it theirs and nothing ties it
to the recipe that produced it. Deriving the tag from the build context instead
settles both — the namespace is FAITH's by construction, and an edited recipe
names an image that has never existed, so it builds.
"""

import hashlib
import os
import re
from pathlib import Path

# Images FAITH builds live under one namespace it owns by construction, so a build
# cannot overwrite a tag someone else published.
BUILT_IMAGE_NAMESPACE = "faith-build"

# Enough to make a collision between two recipes not worth designing around, while
# staying readable in a log line and a `docker images` listing.
_DIGEST_LENGTH = 12


def derived_image_tag(dockerfile: Path) -> str:
    """The tag to build `dockerfile` under, fingerprinting its build context.

    Named for the context directory, since a benchmark's Dockerfile sits in the
    benchmark's own directory, so `$BENCHMARKS_ROOT/foo-bench/Dockerfile` gives
    `faith-build/foo-bench:<digest>`. The environment is handed only its `config`
    sub-dict and never learns the benchmark's name, so the directory is what is
    available — and it is the thing the digest covers anyway.
    """
    return f"{BUILT_IMAGE_NAMESPACE}/{_context_name(dockerfile)}:{_digest(dockerfile)}"


def _context_name(dockerfile: Path) -> str:
    """The context directory's name, reduced to what a Docker repository allows."""
    cleaned = re.sub(r"[^a-z0-9._-]+", "-", dockerfile.parent.name.lower())
    return cleaned.strip("-._") or "agent"


def _digest(dockerfile: Path) -> str:
    """Hash context files by path, mode and content, and symlinks by target text.

    Covers the whole context rather than the Dockerfile alone: a `COPY`'d script
    changes the image without the recipe's text changing at all. The executable bit
    is hashed because it survives into the image and nothing else here records it.

    Do not approximate Docker's ignore matcher: excluding a file Docker includes
    can reuse a stale image. Hash ignored files and .git too. This may rebuild for
    irrelevant edits, but Docker still applies .dockerignore to the actual build.
    """
    context = dockerfile.parent
    digest = hashlib.sha256()
    # Two recipes can coexist in an unchanged context. Hash which one Docker
    # will execute as well as their contents, without tying the tag to a host path.
    digest.update(b"dockerfile\x00")
    digest.update(dockerfile.relative_to(context).as_posix().encode("utf-8"))
    digest.update(b"\x00")
    for path in sorted(p for p in context.rglob("*") if p.is_symlink() or p.is_file()):
        relative = path.relative_to(context).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\x00")
        if path.is_symlink():
            # Docker archives the link itself, including dangling/directory links.
            digest.update(b"l")
            digest.update(os.fsencode(os.readlink(path)))
        else:
            digest.update(b"x" if path.stat().st_mode & 0o111 else b"-")
            digest.update(path.read_bytes())
        digest.update(b"\x00")
    return digest.hexdigest()[:_DIGEST_LENGTH]
