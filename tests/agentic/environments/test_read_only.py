# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for fail-closed workspace policy and archive metadata."""

# Backend doubles exercise private transport seams, never external services.
# pylint: disable=protected-access

import io
import tarfile
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from faith.agentic.core.task import Task
from faith.agentic.environments._read_only import (
    read_only_archive,
    read_only_required,
    workspace_check,
)
from faith.agentic.environments.docker import DockerEnvironment
from faith.agentic.environments.kubernetes import KubernetesEnvironment
from faith.agentic.registries import validate_read_only_workspace


def _tar(entries: list[tuple[str, bytes, int, str, int]]) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w") as archive:
        root = tarfile.TarInfo("workspace")
        root.type, root.mode = tarfile.DIRTYPE, 0o777
        root.uid, root.gid = 1001, 1001
        archive.addfile(root)
        for name, kind, mode, link, size in entries:
            member = tarfile.TarInfo(name)
            member.type, member.mode = kind, mode
            member.uid, member.gid = 1001, 1001
            member.linkname, member.size = link, size
            archive.addfile(member, io.BytesIO(b"x" * size) if size else None)
    return output.getvalue()


def test_archive_protects_directories_and_preserves_content_and_executable_bits() -> (
    None
):
    archive = _tar(
        [
            ("workspace/repo", tarfile.DIRTYPE, 0o777, "", 0),
            ("workspace/repo/a", tarfile.REGTYPE, 0o666, "", 3),
            ("workspace/repo/run", tarfile.REGTYPE, 0o6755, "", 1),
            ("workspace/link", tarfile.SYMTYPE, 0o777, "repo/a", 0),
            ("workspace/hard", tarfile.LNKTYPE, 0o666, "workspace/repo/a", 0),
        ]
    )
    with tarfile.open(
        fileobj=io.BytesIO(read_only_archive(archive, "workspace"))
    ) as result:
        assert {m.name for m in result} == {
            "workspace",
            "workspace/repo",
            "workspace/repo/a",
            "workspace/repo/run",
            "workspace/link",
            "workspace/hard",
        }
        assert all(m.uid == m.gid == 0 for m in result)
        assert result.getmember("workspace").mode == 0o555
        assert result.getmember("workspace/repo").mode == 0o555
        assert result.getmember("workspace/repo/a").mode == 0o444
        assert result.getmember("workspace/repo/run").mode == 0o555
        assert result.getmember("workspace/link").linkname == "repo/a"
        for name in ("workspace/repo/a", "workspace/hard"):
            content = result.extractfile(name)
            assert content is not None and content.read() == b"xxx"


@pytest.mark.parametrize(
    "entry",
    [
        ("outside/file", tarfile.REGTYPE, 0o644, "", 1),
        ("workspace/../file", tarfile.REGTYPE, 0o644, "", 1),
        ("workspace/a", tarfile.SYMTYPE, 0o777, "/tmp/a", 0),
        ("workspace/a", tarfile.SYMTYPE, 0o777, "../outside", 0),
        ("workspace/a", tarfile.FIFOTYPE, 0o644, "", 0),
        ("workspace/a", tarfile.LNKTYPE, 0o644, "outside", 0),
    ],
)
def test_archive_rejects_unsafe_entries(entry) -> None:
    with pytest.raises(ValueError):
        read_only_archive(_tar([entry]), "workspace")


def test_archive_rejects_writing_beneath_a_link() -> None:
    with pytest.raises(ValueError, match="non-directory"):
        read_only_archive(
            _tar(
                [
                    ("workspace/link", tarfile.SYMTYPE, 0o777, ".", 0),
                    ("workspace/link/file", tarfile.REGTYPE, 0o644, "", 1),
                ]
            ),
            "workspace",
        )


@pytest.mark.parametrize("path", ["/", "relative", "/workspace/../tmp", "/workspace/"])
def test_policy_rejects_noncanonical_workspace_paths(path: str) -> None:
    with pytest.raises(ValueError):
        workspace_check(path, sealed=True)


@pytest.mark.parametrize("value", ["false", "true", None, 1, 0])
def test_backend_policy_flag_is_strict(value) -> None:
    with pytest.raises(TypeError, match="boolean"):
        read_only_required({"read_only_workspace": value})


def test_unsupported_environment_fails_configuration_validation() -> None:
    with patch("faith.agentic.registries.AgenticEnvironmentType") as registry:
        registry.return_value.impl_cls = type("Unsupported", (), {})
        with pytest.raises(ValueError, match="does not support"):
            validate_read_only_workspace("custom", True)
        validate_read_only_workspace("custom", False)


def test_docker_refuses_capability_additions_when_protection_is_required() -> None:
    with pytest.raises(ValueError, match="added capabilities"):
        DockerEnvironment(
            {
                "agent_image": "test",
                "read_only_workspace": True,
                "capabilities": ["DAC_OVERRIDE"],
            },
            client=Mock(),
        )


@pytest.mark.parametrize(
    "environment, config",
    [
        (DockerEnvironment, {"agent_image": "test"}),
        (KubernetesEnvironment, {"warmpool": "test"}),
    ],
)
def test_required_policy_blocks_commands_before_successful_setup(
    environment, config
) -> None:
    env = environment({**config, "read_only_workspace": True}, client=Mock())
    with pytest.raises(RuntimeError, match="setup must succeed"):
        env.run_command("touch unprotected")


def test_kubernetes_recovery_reapplies_the_seal() -> None:
    env = KubernetesEnvironment(
        {"warmpool": "test", "read_only_workspace": True}, client=Mock()
    )
    task = Task(id="test", prompt="test", ground_truth=None)
    with (
        patch.object(env, "_probe_runtime"),
        patch.object(env, "seal_workspace") as seal,
    ):
        env.setup(task)
        env.teardown()
        env.setup(task)
    assert seal.call_count == 2


def _kubernetes_policy_env(status=200, payload=None, probe_code=0, upload_status=403):
    connector = Mock()
    connector.send_request.side_effect = [
        SimpleNamespace(
            status_code=status,
            json=lambda: (
                payload
                if payload is not None
                else {
                    "version": 1,
                    "read_only_workspace": True,
                    "uploads_disabled": True,
                }
            ),
        ),
        SimpleNamespace(status_code=upload_status),
    ]
    env = KubernetesEnvironment(
        {"warmpool": "test", "read_only_workspace": True}, client=Mock()
    )
    env._sandbox = SimpleNamespace(
        connector=connector,
        commands=SimpleNamespace(
            run=Mock(
                return_value=SimpleNamespace(
                    exit_code=probe_code,
                    stdout="probe result",
                    stderr="",
                )
            )
        ),
    )
    return env, connector


@pytest.mark.parametrize("status", [404, 405, 409, 422, 501])
def test_kubernetes_requires_an_explicit_runtime_contract(status: int) -> None:
    env, connector = _kubernetes_policy_env(status=status)
    with pytest.raises(RuntimeError, match="does not support"):
        env.seal_workspace()
    assert connector.send_request.call_count == 1


@pytest.mark.parametrize("payload", [{}, {"version": 2}, {"read_only_workspace": True}])
def test_kubernetes_refuses_partial_or_unknown_guarantees(payload) -> None:
    env, _ = _kubernetes_policy_env(payload=payload)
    with pytest.raises(RuntimeError, match="does not support"):
        env.seal_workspace()


def test_kubernetes_verifies_shell_protection() -> None:
    env, _ = _kubernetes_policy_env(probe_code=1)
    with pytest.raises(RuntimeError, match="verification failed"):
        env.seal_workspace()


def test_kubernetes_verifies_the_upload_bypass_is_closed() -> None:
    env, _ = _kubernetes_policy_env(upload_status=200)
    with pytest.raises(RuntimeError, match="still accepts uploads"):
        env.seal_workspace()


def test_kubernetes_accepts_a_verified_seal() -> None:
    env, connector = _kubernetes_policy_env()
    env.seal_workspace()
    assert connector.send_request.call_args_list[0].args == (
        "POST",
        "faith/workspace/read-only",
    )
    assert connector.send_request.call_args_list[1].args == ("POST", "upload")
