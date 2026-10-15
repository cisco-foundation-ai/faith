# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Read-only workspace checks and Docker archive metadata."""

import io
import posixpath
import shlex
import tarfile


def read_only_required(config: dict) -> bool:
    """Read the injected policy without accepting truthy strings."""
    value = config.get("read_only_workspace", False)
    if not isinstance(value, bool):
        raise TypeError("read_only_workspace must be a boolean")
    return value


def workspace_check(workspace: str, *, sealed: bool) -> str:
    """Check identity, privileges and workspace paths from the agent's shell."""
    if not workspace.startswith("/") or posixpath.normpath(workspace) != workspace:
        raise ValueError("read_only_workspace requires a canonical absolute workspace")
    if workspace == "/":
        raise ValueError("read_only_workspace cannot protect the filesystem root")
    script = f"""set -eu
test "$(id -u)" != 0
awk '
    /^NoNewPrivs:/ {{ nnp = ($2 == 1) }}
    /^Cap[A-Za-z]+:/ {{ if ($2 !~ /^0+$/) exit 1; caps++ }}
    END {{ if (!nnp || caps != 5) exit 1 }}
' /proc/self/status
workspace={shlex.quote(workspace)}
test "$(realpath -e -- "$workspace")" = "$workspace"
parent=$(dirname -- "$workspace")
while :; do
    test "$(stat -c %u -- "$parent")" = 0
    test ! -w "$parent"
    test "$parent" = / && break
    parent=$(dirname -- "$parent")
done
"""
    if sealed:
        script += """bad=$(find -P "$workspace" \\( ! -user 0 -o \\( ! -type l -a -writable \\) \\) -print -quit)
test -z "$bad"
test -r "$workspace" && test -x "$workspace"
find -P "$workspace" -type l -exec bash -c '
    root=$1; shift
    for link; do
        target=$(realpath -m -- "$link") || exit 1
        case "$target" in "$root"|"$root"/*) ;; *) exit 1 ;; esac
    done
' _ "$workspace" {} +
"""
    return script


def read_only_archive(archive: bytes, root: str) -> bytes:
    """Repack a daemon workspace snapshot with root ownership and no write bits.

    Keep the named root for put_archive(parent): Docker ignores metadata on a
    '.' entry. Special files, duplicate paths, escaping links, and entries beneath
    links are rejected.
    No member is extracted onto the runner's filesystem.
    """
    if not root or root in (".", "..") or "/" in root:
        raise ValueError("Invalid workspace archive root")
    result = io.BytesIO()
    with tarfile.open(fileobj=io.BytesIO(archive)) as source:
        names = _workspace_members(source, root)
        _validate_links(names, root)
        with tarfile.open(fileobj=result, mode="w") as target:
            for name, member in sorted(names.items()):
                _copy_member(
                    source, target, root if name == "." else f"{root}/{name}", member
                )
    return result.getvalue()


def _workspace_members(
    source: tarfile.TarFile, root: str
) -> dict[str, tarfile.TarInfo]:
    names: dict[str, tarfile.TarInfo] = {}
    for member in source.getmembers():
        name = member.name.rstrip("/")
        if name == root:
            relative = "."
        elif name.startswith(f"{root}/"):
            relative = name[len(root) + 1 :]
        else:
            raise ValueError(f"Archive member outside workspace: {name!r}")
        if (
            posixpath.normpath(relative) != relative
            or relative.startswith("/")
            or ".." in relative.split("/")
            or relative in names
        ):
            raise ValueError(f"Unsafe workspace archive member: {name!r}")
        if not (member.isfile() or member.isdir() or member.issym() or member.islnk()):
            raise ValueError(f"Unsupported workspace file: {name!r}")
        names[relative] = member
    if "." not in names or not names["."].isdir():
        raise ValueError("Workspace archive has no directory root")
    return names


def _validate_links(names: dict[str, tarfile.TarInfo], root: str) -> None:
    for name, member in names.items():
        parent = posixpath.dirname(name)
        while parent:
            if parent not in names or not names[parent].isdir():
                raise ValueError(f"Workspace entry beneath a non-directory: {name!r}")
            parent = posixpath.dirname(parent)
        if member.issym():
            target = posixpath.normpath(
                posixpath.join(posixpath.dirname(name), member.linkname)
            )
            if target.startswith("/") or target == ".." or target.startswith("../"):
                raise ValueError(f"Workspace symlink escapes its root: {name!r}")
        if member.islnk():
            target = member.linkname.removeprefix(f"{root}/")
            if target not in names or not names[target].isfile():
                raise ValueError(f"Unsafe workspace hard link: {name!r}")


def _copy_member(
    source: tarfile.TarFile, target: tarfile.TarFile, name: str, member: tarfile.TarInfo
) -> None:
    info = tarfile.TarInfo(name)
    info.mtime = member.mtime
    if member.isdir():
        info.type, info.mode = tarfile.DIRTYPE, 0o555
        target.addfile(info)
    elif member.issym():
        info.type, info.mode = tarfile.SYMTYPE, 0o777
        info.linkname = member.linkname
        target.addfile(info)
    else:
        content = source.extractfile(member)
        if content is None:
            raise ValueError(f"Unreadable workspace file: {name!r}")
        with content:
            info.size = content.seek(0, io.SEEK_END)
            content.seek(0)
            info.mode = 0o555 if member.mode & 0o111 else 0o444
            target.addfile(info, content)
