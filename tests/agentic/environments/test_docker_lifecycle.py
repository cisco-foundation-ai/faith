# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Failed-start cleanup uses only the container owned by the setup attempt."""

import os
import uuid
from unittest.mock import Mock, patch

import docker
import pytest
import requests

from faith.agentic.core import Task
from faith.agentic.core.environment import EnvironmentUnavailableError
from faith.agentic.environments.docker import DockerEnvironment


@pytest.mark.parametrize("stage", ["create", "start"])
def test_a_lost_response_removes_the_owned_container(stage: str) -> None:
    client = Mock()
    client.images.list.return_value = [Mock()]
    container = Mock()

    def create(**kwargs):
        container.labels = kwargs["labels"]
        if stage == "create":
            raise requests.ReadTimeout("lost create response")
        return container

    client.containers.create.side_effect = create
    client.containers.get.return_value = container
    container.start.side_effect = requests.ReadTimeout("lost start response")
    env = DockerEnvironment({"agent_image": "test:local"}, client=client)
    with pytest.raises(EnvironmentUnavailableError, match=f"lost {stage} response"):
        env.setup(Task(id="test", prompt="test", ground_truth=None))
    container.remove.assert_called_once_with(force=True)
    client.containers.list.assert_not_called()
    if stage == "create":
        name = client.containers.create.call_args.kwargs["name"]
        client.containers.get.assert_called_once_with(name)
        container.start.assert_not_called()
    else:
        client.containers.get.assert_not_called()
    env.teardown()
    container.stop.assert_not_called()


def test_lost_create_response_never_removes_an_unowned_container() -> None:
    client = Mock()
    client.images.list.return_value = [Mock()]
    client.containers.create.side_effect = requests.ReadTimeout("lost response")
    existing = Mock(labels={"faith.managed": "true", "faith.instance-id": "other"})
    client.containers.get.return_value = existing
    env = DockerEnvironment({"agent_image": "test:local"}, client=client)
    with pytest.raises(EnvironmentUnavailableError):
        env.setup(Task(id="test", prompt="test", ground_truth=None))
    env.teardown()
    existing.remove.assert_not_called()
    existing.stop.assert_not_called()


def test_cleanup_failure_does_not_hide_the_start_error(caplog) -> None:
    client = Mock()
    client.images.list.return_value = [Mock()]
    container = client.containers.create.return_value
    container.start.side_effect = requests.ReadTimeout("original start error")
    container.remove.side_effect = requests.ConnectionError("daemon unavailable")
    env = DockerEnvironment({"agent_image": "test:local"}, client=client)
    with pytest.raises(EnvironmentUnavailableError, match="original start error"):
        env.setup(Task(id="test", prompt="test", ground_truth=None))
    assert "Could not clean up container faith-" in caplog.text
    env.teardown()
    container.stop.assert_called_once_with(timeout=5)


def test_create_still_pulls_a_missing_image() -> None:
    client = Mock()
    client.images.list.return_value = [Mock()]
    container = Mock()
    client.containers.create.side_effect = [
        docker.errors.ImageNotFound("missing"),
        container,
    ]
    env = DockerEnvironment({"agent_image": "test:local"}, client=client)
    env.setup(Task(id="test", prompt="test", ground_truth=None))
    client.images.pull.assert_called_once_with("test:local")
    assert client.containers.create.call_count == 2
    container.start.assert_called_once_with()
    env.teardown()


@pytest.mark.skipif(
    os.environ.get("FAITH_TEST_READ_ONLY") != "1", reason="opt-in Docker integration"
)
@pytest.mark.parametrize("stage", ["create", "start"])
def test_real_docker_lost_response_leaves_no_owned_container(stage: str) -> None:
    task_id = f"startup-test-{uuid.uuid4().hex}"
    client = docker.from_env()
    env = DockerEnvironment(
        {"agent_image": "faith-read-only-agent:test"}, client=client
    )
    original_create = docker.models.containers.ContainerCollection.create
    original_start = docker.models.containers.Container.start

    def create(collection, **kwargs):
        container = original_create(collection, **kwargs)
        if stage == "create":
            raise requests.ReadTimeout("synthetic lost response after real create")
        return container

    def start(container):
        original_start(container)
        raise requests.ReadTimeout("synthetic lost response after real start")

    filters = {"label": f"faith.task-id={task_id}"}
    try:
        with (
            patch.object(
                docker.models.containers.ContainerCollection, "create", create
            ),
            patch.object(docker.models.containers.Container, "start", start),
            pytest.raises(EnvironmentUnavailableError, match="synthetic lost response"),
        ):
            env.setup(Task(id=task_id, prompt="test", ground_truth=None))
        assert client.containers.list(all=True, filters=filters) == []
    finally:
        for container in client.containers.list(all=True, filters=filters):
            container.remove(force=True)
        client.close()
