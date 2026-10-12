# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the content-derived tags FAITH builds its own images under."""

import shutil
from pathlib import Path

import pytest

from faith.agentic.environments._build import derived_image_tag


def _context(
    root: Path,
    files: dict[str, str] | None = None,
    name: str = "vulnerability-localization",
) -> Path:
    """Write a build context directory, returning its Dockerfile."""
    context = root / name
    context.mkdir(parents=True)
    for filename, content in (files or {}).items():
        (context / filename).parent.mkdir(parents=True, exist_ok=True)
        (context / filename).write_text(content, encoding="utf-8")
    dockerfile = context / "Dockerfile"
    if not dockerfile.exists():
        dockerfile.write_text("FROM ubuntu:24.04\n", encoding="utf-8")
    return dockerfile


def test_the_tag_names_the_faith_namespace_the_context_and_a_digest(
    tmp_path: Path,
) -> None:
    """The whole shape, since each part is load-bearing.

    The namespace is FAITH's by construction, so a build cannot overwrite a tag
    someone else published; the context's name keeps a `docker images` listing
    readable; the digest is what makes the tag identify a recipe.
    """
    tag = derived_image_tag(_context(tmp_path))
    namespace, _, digest = tag.partition(":")

    assert namespace == "faith-build/vulnerability-localization"
    assert len(digest) == 12 and digest.isalnum() and digest.islower()


def test_the_same_context_derives_the_same_tag(tmp_path: Path) -> None:
    """Reuse depends on this: an unchanged benchmark must not rebuild."""
    first = derived_image_tag(_context(tmp_path / "a"))
    second = derived_image_tag(_context(tmp_path / "b"))

    assert first == second


@pytest.mark.parametrize("recipe", ["FROM ubuntu:24.04\n", "FROM ubuntu:22.04\n"])
def test_selecting_another_dockerfile_changes_the_tag(
    tmp_path: Path, recipe: str
) -> None:
    """The selected recipe is part of the identity, even in an unchanged context."""
    first = _context(tmp_path, {"Dockerfile.other": recipe})
    second = first.with_name("Dockerfile.other")

    assert derived_image_tag(first) != derived_image_tag(second)


def test_an_alternate_recipe_is_stable_across_checkout_locations(
    tmp_path: Path,
) -> None:
    """Selection must be context-relative, never an absolute host path."""
    files = {"Dockerfile.other": "FROM ubuntu:22.04\n"}
    first = _context(tmp_path / "a", files).with_name("Dockerfile.other")
    second = _context(tmp_path / "b", files).with_name("Dockerfile.other")

    assert derived_image_tag(first) == derived_image_tag(second)


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ({"Dockerfile": "FROM ubuntu:24.04\n"}, {"Dockerfile": "FROM ubuntu:22.04\n"}),
        ({"setup.sh": "echo one\n"}, {"setup.sh": "echo two\n"}),
        ({"setup.sh": "echo one\n"}, {"other.sh": "echo one\n"}),
        ({}, {"extra.txt": "new file\n"}),
    ],
    ids=["dockerfile_edited", "context_file_edited", "file_renamed", "file_added"],
)
def test_any_change_the_build_would_see_changes_the_tag(
    tmp_path: Path, before: dict[str, str], after: dict[str, str]
) -> None:
    """Covers the context, not just the Dockerfile: a COPY'd script is part of the image.

    Hashing only the recipe's text would reuse a stale image whenever a file it
    copies in changed — the staleness this whole scheme exists to prevent, one
    level down.
    """
    assert derived_image_tag(_context(tmp_path / "a", before)) != derived_image_tag(
        _context(tmp_path / "b", after)
    )


def test_a_reverted_edit_derives_the_original_tag_again(tmp_path: Path) -> None:
    """The cache gain: going back to a recipe already built needs no rebuild.

    A single mutable tag cannot do this — the revert would rebuild, because the
    image that tag names was overwritten by the edit.
    """
    dockerfile = _context(tmp_path)
    original = derived_image_tag(dockerfile)
    dockerfile.write_text("FROM ubuntu:22.04\n", encoding="utf-8")
    edited = derived_image_tag(dockerfile)
    dockerfile.write_text("FROM ubuntu:24.04\n", encoding="utf-8")

    assert (derived_image_tag(dockerfile), edited != original) == (original, True)


def test_the_executable_bit_is_part_of_the_digest(tmp_path: Path) -> None:
    """It survives into the image, and no file's content records it."""
    dockerfile = _context(tmp_path, {"setup.sh": "echo hi\n"})
    before = derived_image_tag(dockerfile)
    (dockerfile.parent / "setup.sh").chmod(0o755)

    assert derived_image_tag(dockerfile) != before


def test_even_a_dockerignored_file_is_hashed_conservatively(tmp_path: Path) -> None:
    """Extra rebuilds are preferable to omitting a file Docker actually includes."""
    dockerfile = _context(
        tmp_path, {".dockerignore": "*.yaml\n", "benchmark.yaml": "limits: 1\n"}
    )
    before = derived_image_tag(dockerfile)
    (dockerfile.parent / "benchmark.yaml").write_text("limits: 2\n", encoding="utf-8")

    assert derived_image_tag(dockerfile) != before


@pytest.mark.parametrize("pattern", ["src/*.txt", "./src/*.txt", "/src/*.txt"])
def test_nested_files_are_not_omitted_by_an_approximate_ignore_matcher(
    tmp_path: Path, pattern: str
) -> None:
    """Python fnmatch's star crosses slashes; Docker's single star does not."""
    dockerfile = _context(
        tmp_path,
        {".dockerignore": pattern + "\n", "src/nested/input.txt": "first\n"},
    )
    before = derived_image_tag(dockerfile)
    (dockerfile.parent / "src/nested/input.txt").write_text(
        "second\n", encoding="utf-8"
    )

    assert derived_image_tag(dockerfile) != before


def test_an_ignored_dockerfile_is_still_hashed(tmp_path: Path) -> None:
    """The daemon sends the Dockerfile whether or not `.dockerignore` names it."""
    dockerfile = _context(tmp_path, {".dockerignore": "Dockerfile\n"})
    before = derived_image_tag(dockerfile)
    dockerfile.write_text("FROM alpine:3.20\n", encoding="utf-8")

    assert derived_image_tag(dockerfile) != before


def test_a_negated_dockerignore_pattern_puts_a_file_back(tmp_path: Path) -> None:
    """Later patterns win, as Docker's own matcher has them."""
    dockerfile = _context(
        tmp_path, {".dockerignore": "*.yaml\n!keep.yaml\n", "keep.yaml": "a\n"}
    )
    before = derived_image_tag(dockerfile)
    (dockerfile.parent / "keep.yaml").write_text("b\n", encoding="utf-8")

    assert derived_image_tag(dockerfile) != before


def test_git_files_are_hashed_too(tmp_path: Path) -> None:
    """Docker can COPY .git; silently excluding it would leave another stale hit."""
    dockerfile = _context(tmp_path)
    before = derived_image_tag(dockerfile)
    git_dir = dockerfile.parent / ".git"
    git_dir.mkdir()
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")

    assert derived_image_tag(dockerfile) != before


@pytest.mark.parametrize("kind", ["file", "directory", "dangling"])
def test_symlink_identity_is_hashed_without_resolving_its_target(
    tmp_path: Path, kind: str
) -> None:
    """Even equal-content targets produce different links in Docker's context."""
    dockerfile = _context(tmp_path)
    for name in ("one", "two"):
        target = dockerfile.parent / name
        if kind == "directory":
            target.mkdir()
        elif kind == "file":
            target.write_text("same content", encoding="utf-8")
    link = dockerfile.parent / "selected"
    absent = derived_image_tag(dockerfile)
    link.symlink_to("one", target_is_directory=kind == "directory")
    first = derived_image_tag(dockerfile)
    assert first != absent
    link.unlink()
    link.symlink_to("two", target_is_directory=kind == "directory")
    assert derived_image_tag(dockerfile) != first
    link.unlink()
    link.symlink_to("one", target_is_directory=kind == "directory")
    assert derived_image_tag(dockerfile) == first
    link.unlink()
    assert derived_image_tag(dockerfile) == absent


def test_replacing_a_regular_file_with_a_symlink_changes_the_tag(
    tmp_path: Path,
) -> None:
    dockerfile = _context(tmp_path, {"target": "same", "selected": "same"})
    before = derived_image_tag(dockerfile)
    selected = dockerfile.parent / "selected"
    selected.unlink()
    selected.symlink_to("target")
    assert derived_image_tag(dockerfile) != before


def test_symlink_targets_are_not_normalized_or_followed(tmp_path: Path) -> None:
    dockerfile = _context(tmp_path, {"target": "same"})
    selected = dockerfile.parent / "selected"
    selected.symlink_to("target")
    before = derived_image_tag(dockerfile)
    selected.unlink()
    selected.symlink_to("./target")
    assert derived_image_tag(dockerfile) != before

    external = tmp_path / "external"
    external.write_text("outside the context", encoding="utf-8")
    (dockerfile.parent / "external-link").symlink_to(external)
    (dockerfile.parent / "loop").symlink_to("loop")
    before = derived_image_tag(dockerfile)
    external.write_text("not sent to Docker", encoding="utf-8")
    assert derived_image_tag(dockerfile) == before


def test_context_with_symlinks_is_stable_across_checkout_locations(
    tmp_path: Path,
) -> None:
    dockerfile = _context(tmp_path / "original", {"file": "data", "dir/child": "data"})
    for name, target in (
        ("file-link", "file"),
        ("dir-link", "dir"),
        ("dangling-link", "missing"),
    ):
        (dockerfile.parent / name).symlink_to(
            target, target_is_directory=target == "dir"
        )
    before = derived_image_tag(dockerfile)
    moved = tmp_path / "relocated" / dockerfile.parent.name
    shutil.copytree(dockerfile.parent, moved, symlinks=True)
    assert derived_image_tag(moved / dockerfile.name) == before


@pytest.mark.parametrize(
    ("directory", "expected"),
    [
        ("Vulnerability-Localization", "faith-build/vulnerability-localization"),
        ("my bench!", "faith-build/my-bench"),
        ("...", "faith-build/agent"),
    ],
    ids=["uppercased", "illegal_characters", "nothing_usable_left"],
)
def test_a_context_directory_name_is_reduced_to_a_legal_repository(
    tmp_path: Path, directory: str, expected: str
) -> None:
    """Docker rejects an uppercase or punctuated repository, so the name is not passed through."""
    tag = derived_image_tag(_context(tmp_path, name=directory))

    assert tag.rpartition(":")[0] == expected
